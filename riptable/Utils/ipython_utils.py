"""This module has side effects on import.
If DisplayOptions.CUSTOM_COMPLETION is True then this will disable IPython Jedi and
use a IPCompleter custom complete.
"""
import re
import sys
from collections import ChainMap
from types import MethodType
from typing import Iterable, List, Optional, Tuple

from IPython import get_ipython
from IPython.core.completer import _FakeJediCompletion
from IPython.utils.generics import complete_object

import riptable

__all__ = ["enable_custom_attribute_completion"]


_PRESERVE_ORDER_SENTINEL: str = "PRESERVE_ORDER_SENTINEL"
_DEBUG: bool = False  # toggle to enable stdout logging


def _is_public(name: str) -> bool:
    """Returns True if a method is not protected, nor private; otherwise False."""
    return not (name.startswith("__") or name.startswith("_"))


def _get_key_names(obj: object) -> List[str]:
    if hasattr(obj, "keys"):
        return obj.keys()
    return []


def _ns_search(name: str) -> Optional[object]:
    """
    IPython specific search of local then global namespace for name.
    If found returns the object, otherwise return None.
    """
    from IPython import get_ipython

    ip = get_ipython()

    # check ipython namespace table
    chain_map = ChainMap(
        ip.ns_table.get("user_local"),
        ip.ns_table.get("user_global"),
        ip.ns_table.get("builtin"),
    )
    if name in chain_map:
        return chain_map.get(name)

    # check python namespace; if ipython namespace is additive to the python namespace then this is redundant
    if name in locals():
        return locals()[name]
    if name in globals():
        return globals()[name]

    return None


# The motivation for this monkey patched completion is to add support for custom completion that preserve ordering
# while remaining backwards compatible with the original IPython completion machinery (see RIP-111 for more details).
# IPython version 7.10.2supports two approaches for custom completion that didn't work for us:
# 1) IPython supports custom completer dispatching for python objects.
#   https://ipython.readthedocs.io/en/stable/api/generated/IPython.utils.generics.html#IPython.utils.generics.complete_object
# 2) IPython supports custom completion functions
#   https://ipython.readthedocs.io/en/stable/api/generated/IPython.core.interactiveshell.html#IPython.core.interactiveshell.InteractiveShell.set_custom_completer
# Approach (1) does post filtering and ordering after the custom completer has run.
# Approach (2) - set_custom_completer attempts to insert the completer in the IPCompleter.matchers property which
# returns a list literal thereby dropping the custom completer insertion.
def _monkey_patched__complete(
    self, *, cursor_line, cursor_pos, line_buffer=None, text=None, full_text=None
) -> Tuple[str, List[str], List[str], Iterable[_FakeJediCompletion]]:
    """

    Like complete but can also returns raw jedi completions as well as the
    origin of the completion text. This could (and should) be made much
    cleaner but that will be simpler once we drop the old (and stateful)
    :any:`complete` API.


    With current provisional API, cursor_pos act both (depending on the
    caller) as the offset in the ``text`` or ``line_buffer``, or as the
    ``column`` when passing multiline strings this could/should be renamed
    but would add extra noise.
    """

    # if the cursor position isn't given, the only sane assumption we can
    # make is that it's at theendregion of the line (the common case)
    if cursor_pos is None:
        cursor_pos = len(line_buffer) if text is None else len(text)

    if self.use_main_ns:
        self.namespace = __main__.__dict__  # noqa: F821

    # if text is either None or an empty string, rely on the line buffer
    if (not line_buffer) and full_text:
        line_buffer = full_text.split("\n")[cursor_line]
    if not text:
        text = self.splitter.split_line(line_buffer, cursor_pos)

    if self.backslash_combining_completions:
        # allow deactivation of these on windows.
        base_text = text if not line_buffer else line_buffer[:cursor_pos]
        latex_text, latex_matches = self.latex_matches(base_text)
        if latex_matches:
            return latex_text, latex_matches, ["latex_matches"] * len(latex_matches), ()
        name_text = ""
        name_matches = []

        # region monkey patched region
        # The latex namespace searches can be more defensive by falling back to a local implementation of these function
        # definitions if they are not found from _ns_search.
        back_latex_name_matches, back_unicode_name_matches = (
            _ns_search("back_latex_name_matches"),
            _ns_search("back_unicode_name_matches"),
        )
        if back_latex_name_matches and back_unicode_name_matches:
            # endregion monkey patched region
            # need to add self.fwd_unicode_match() function here when done
            for meth in (
                self.unicode_name_matches,
                back_latex_name_matches,
                back_unicode_name_matches,
                self.fwd_unicode_match,
            ):
                name_text, name_matches = meth(base_text)
                if name_text:
                    # region monkey patched region
                    # As of IPython 7.10.2, MATCHES_LIMIT is set to 500
                    MATCHES_LIMIT = _ns_search("MATCHES_LIMIT") if _ns_search("MATCHES_LIMIT") else 500
                    # endregion monkey patched region
                    return (
                        name_text,
                        name_matches[:MATCHES_LIMIT],
                        [meth.__qualname__] * min(len(name_matches), MATCHES_LIMIT),
                        (),
                    )

    # If no line buffer is given, assume the input text is all there was
    if line_buffer is None:
        line_buffer = text

    self.line_buffer = line_buffer
    self.text_until_cursor = self.line_buffer[:cursor_pos]

    # Do magic arg matches
    MATCHES_LIMIT = 500  # get this from local
    for matcher in self.magic_arg_matchers:
        matches = list(matcher(line_buffer))[:MATCHES_LIMIT]
        if matches:
            origins = [matcher.__qualname__] * len(matches)
            return text, matches, origins, ()

    # Start with a clean slate of completions
    matches = []

    # FIXME: we should extend our api to return a dict with completions for
    # different types of objects.  The rlcomplete() method could then
    # simply collapse the dict into a list for readline, but we'd have
    # richer completion semantics in other environments.
    completions = ()
    if self.use_jedi:
        if not full_text:
            full_text = line_buffer
        completions = self._jedi_matches(cursor_pos, cursor_line, full_text)

    if self.merge_completions:
        matches = []
        for matcher in self.matchers:
            try:
                matches.extend([(m, matcher.__qualname__) for m in matcher(text)])
            except:
                # Show the ugly traceback if the matcher causes an
                # exception, but do NOT crash the kernel!
                sys.excepthook(*sys.exc_info())
    else:
        for matcher in self.matchers:
            matches = [(m, matcher.__qualname__) for m in matcher(text)]
            if matches:
                break

    seen = set()
    filtered_matches = set()
    for m in matches:
        t, c = m
        if t not in seen:
            filtered_matches.add(m)
            seen.add(t)

    _filtered_matches = []
    if _DEBUG:
        print(
            f"IPCompleter.monkey_patched__complete: check if {text + _PRESERVE_ORDER_SENTINEL} in matches={[m[0] for m in matches]}\n"
        )
    # region monkey patched region
    matches_text = [
        m[0] for m in matches
    ]  # unpack the text part of the list of tuples of text matches and matcher name
    if len(matches) and text + _PRESERVE_ORDER_SENTINEL in matches_text:
        matches.pop()  # remove the PRESERVE_ORDER_SENTINEL
        _filtered_matches = matches
        if _DEBUG:
            print(
                f"IPCompleter.monkey_patched__complete: custom completion path: check matches={[m[0] for m in matches]}\n\tmatches{matches}"
            )
    else:
        completions_sorting_key_fn = _ns_search("completions_sorting_key")
        if not completions_sorting_key_fn:
            # completions_sorting_key is pulled in from IPython 7.10.2 in the event we cannot find this function in the namespace
            def completions_sorting_key(word):
                """key for sorting completions

                This does several things:

                - Demote any completions starting with underscores to the end
                - Insert any %magic and %%cellmagic completions in the alphabetical order
                  by their name
                """
                prio1, prio2 = 0, 0

                if word.startswith("__"):
                    prio1 = 2
                elif word.startswith("_"):
                    prio1 = 1

                if word.endswith("="):
                    prio1 = -1

                if word.startswith("%%"):
                    # If there's another % in there, this is something else, so leave it alone
                    if not "%" in word[2:]:
                        word = word[2:]
                        prio2 = 2
                elif word.startswith("%"):
                    if not "%" in word[1:]:
                        word = word[1:]
                        prio2 = 1

                return prio1, word, prio2

            completions_sorting_key_fn = completions_sorting_key
        _filtered_matches = sorted(filtered_matches, key=lambda x: completions_sorting_key_fn(x[0]))
    # endregion monkey patched region

    custom_res = [(m, "custom") for m in self.dispatch_custom_completer(text) or []]

    _filtered_matches = custom_res or _filtered_matches

    _filtered_matches = _filtered_matches[:MATCHES_LIMIT]
    _matches = [m[0] for m in _filtered_matches]
    origins = [m[1] for m in _filtered_matches]

    self.matches = _matches

    return text, _matches, origins, completions


def _ipython_major_version() -> int:
    try:
        import IPython

        return int(str(IPython.__version__).split(".")[0])
    except Exception:
        return 0


def _monkey_patched__complete_v9(self, *, cursor_line, cursor_pos, line_buffer=None, text=None, full_text=None):
    """
    IPython 9 variant of the monkey-patched ``_complete``.

    IPython 9 replaced the provisional 4-tuple ``_complete`` API with a
    dict of matcher results, and its ``custom_completer_matcher`` already
    marks custom completions as ordered. This wrapper therefore delegates
    to the original ``_complete`` (saved as ``_riptable_original_complete``
    by ``_set__monkey_patched__complete``) and only strips the
    order-preservation sentinel appended by
    ``_attribute_method_completion`` (which would otherwise appear as a
    completion) and marks the affected matcher result as ordered, so the
    keys-first order is kept instead of being re-sorted.
    """
    call_kwargs = dict(
        cursor_line=cursor_line, cursor_pos=cursor_pos, line_buffer=line_buffer, text=text, full_text=full_text
    )
    results = self._riptable_original_complete(**call_kwargs)
    if _python_matcher_empty(results):
        token = _completion_token(self, text, line_buffer, full_text, cursor_pos, cursor_line)
        if token.endswith(".") and token.count(".") >= 2:
            retried = _retry_with_plain_eval(self, call_kwargs)
            if retried is not None:
                results = retried
    sentinel_found = False
    for matcher_id, result in results.items():
        if "jedi" in matcher_id.lower():
            continue
        completions = result.get("completions", [])
        if not isinstance(completions, list):
            try:
                completions = list(completions)
            except TypeError:
                continue
            result["completions"] = completions
        texts = [
            completion if isinstance(completion, str) else getattr(completion, "text", None)
            for completion in completions
        ]
        if not any(isinstance(t, str) and t.endswith(_PRESERVE_ORDER_SENTINEL) for t in texts):
            continue
        result["completions"] = [
            completion
            for completion, text_value in zip(completions, texts)
            if not (isinstance(text_value, str) and text_value.endswith(_PRESERVE_ORDER_SENTINEL))
        ]
        result["ordered"] = True
        sentinel_found = True
    if sentinel_found:
        # The attribute matcher already provides every completion (keys
        # first); Jedi's duplicates would be yielded first by
        # ``completions()`` and displace that order during deduplication.
        # Dropping them here mirrors the legacy ``complete()`` behavior,
        # which arranged results while skipping the Jedi matcher.
        for matcher_id in [key for key in results if "jedi" in key.lower()]:
            del results[matcher_id]
        return results

    # When Jedi is enabled it suppresses the Python matcher, so no
    # sentinel-marked result exists and Jedi alone provides the attribute
    # completions (sorted by Jedi). Reorder those by the object's keys so
    # keys still come first, matching the disabled-Jedi behavior.
    _reorder_jedi_completions_by_keys(self, results, text, line_buffer, full_text, cursor_pos, cursor_line)
    return results


_DOTTED_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*")


def _completion_token(completer, text, line_buffer, full_text, cursor_pos, cursor_line) -> str:
    """Return the completion token for a ``_complete`` call."""
    if text:
        return text
    line = line_buffer
    if not line and full_text:
        try:
            line = full_text.split("\n")[cursor_line]
        except Exception:
            line = ""
    if not line:
        return ""
    try:
        return completer.splitter.split_line(line, cursor_pos)
    except Exception:
        return ""


def _plain_eval_dotted_expr(completer, expr):
    """
    Evaluate a dotted-name expression with plain ``eval``.

    IPython 9's guarded evaluation refuses attribute access on objects
    with a custom ``__getattr__`` (which riptable containers use for
    columns/items), so nested expressions such as ``m.ds_alpha`` evaluate
    to ``not_found`` and yield no completions. Only plain dotted names
    rooted at an object with keys (or ``_ipython_key_completions_``) are
    evaluated this way; anything else returns ``None``. Attribute access
    on those containers is a pure lookup, matching what IPython 8 did
    with plain ``eval`` for every completion.
    """
    if not _DOTTED_NAME_RE.fullmatch(expr):
        return None
    try:
        from IPython.core.completer import not_found
    except Exception:
        return None
    try:
        root = completer._evaluate_expr(expr.split(".", 1)[0])
    except Exception:
        return None
    if root is not_found:
        return None
    if not (hasattr(root, "keys") or hasattr(root, "_ipython_key_completions_")):
        return None
    try:
        namespace = {}
        for ns in (getattr(completer, "global_namespace", None), getattr(completer, "namespace", None)):
            if ns is not None:
                namespace.update(dict(ns))
        return eval(expr, namespace)  # noqa: S307 - dotted names on keyed containers only
    except Exception:
        return None


def _python_matcher_empty(results) -> bool:
    """True when the Python matcher ran but produced no completions."""
    for matcher_id, result in results.items():
        if matcher_id.endswith("python_matcher"):
            completions = result.get("completions", [])
            return isinstance(completions, list) and len(completions) == 0
    return False


def _retry_with_plain_eval(completer, call_kwargs):
    """
    Re-run the original ``_complete`` with plain evaluation for dotted
    names, for one call only. Guarded evaluation refuses nested attribute
    access on riptable containers; the retry falls back to plain eval
    only for dotted names rooted at keyed containers (see
    ``_plain_eval_dotted_expr``) and restores the guarded evaluator
    afterwards, whatever happens.
    """
    try:
        from IPython.core.completer import not_found
    except Exception:
        return None
    original_evaluate = completer._evaluate_expr

    def _patched_evaluate(expr):
        obj = original_evaluate(expr)
        if obj is not_found:
            alt = _plain_eval_dotted_expr(completer, expr)
            return alt if alt is not None else obj
        return obj

    completer._evaluate_expr = _patched_evaluate
    try:
        return completer._riptable_original_complete(**call_kwargs)
    except Exception:
        return None
    finally:
        try:
            del completer._evaluate_expr
        except Exception:
            completer._evaluate_expr = original_evaluate


def _reorder_jedi_completions_by_keys(
    completer, results, text, line_buffer, full_text, cursor_pos, cursor_line
) -> None:
    """Reorder Jedi attribute completions so object keys come first.

    Only applies when the completion token is an attribute access (ends
    with ``.``) on an object with keys. Non-key completions keep Jedi's
    order. Any failure to evaluate the object leaves results untouched.
    """
    token = _completion_token(completer, text, line_buffer, full_text, cursor_pos, cursor_line)
    if not token or not token.endswith("."):
        return
    expr = token[:-1]
    if not expr:
        return
    obj = _plain_eval_dotted_expr(completer, expr)
    if obj is None:
        try:
            from IPython.core.completer import not_found

            obj = completer._evaluate_expr(expr)
            if obj is not_found:
                return
        except Exception:
            return
    try:
        if hasattr(obj, "_ipython_key_completions_"):
            keys = obj._ipython_key_completions_()
        elif hasattr(obj, "keys"):
            keys = obj.keys()
        else:
            return
        keys = [str(key) for key in keys]
        if not keys:
            return
        rank = {key: index for index, key in enumerate(sorted(keys))}
        for matcher_id, result in results.items():
            if "jedi" not in matcher_id.lower():
                continue
            completions = result.get("completions", [])
            if not isinstance(completions, list):
                try:
                    completions = list(completions)
                except TypeError:
                    continue
                result["completions"] = completions

            def _completion_name(completion):
                name = getattr(completion, "name_with_symbols", None)
                if name is None:
                    name = getattr(completion, "name", None)
                if name is None:
                    name = getattr(completion, "text", None)
                if not isinstance(name, str):
                    return None
                return name.rsplit(".", 1)[-1]

            reordered = sorted(
                enumerate(completions),
                key=lambda pair: (rank.get(_completion_name(pair[1]), len(rank)), pair[0]),
            )
            result["completions"] = [completion for _, completion in reordered]
    except Exception:
        return


def _disable_jedi() -> None:
    ip = get_ipython()
    ip.Completer.use_jedi = False


def _set__monkey_patched__complete() -> None:
    ip = get_ipython()
    if _ipython_major_version() >= 9:
        completer = ip.Completer
        if not hasattr(completer, "_riptable_original_complete"):
            completer._riptable_original_complete = completer._complete
        completer._complete = MethodType(_monkey_patched__complete_v9, completer)
    else:
        ip.Completer._complete = MethodType(_monkey_patched__complete, ip.Completer)


def _set_custom_attribute_completion() -> None:
    @complete_object.register(object)
    def _attribute_method_completion(obj: object, existing_completions: List[str]) -> List[str]:
        key_names: List[str] = []  # grab keys if they are available
        if hasattr(obj, "_ipython_key_completions_"):
            key_names = obj._ipython_key_completions_()
        elif hasattr(obj, "keys"):
            key_names = obj.keys()

        make_sorted_list = lambda lst: sorted(list(lst), key=str.casefold)  # Type Callable[[Iterable], List]

        # Return the sorted keys, then sorted existing completions, then a sentinel
        # for the custom completion to signal preserving this ordering.
        key_names = make_sorted_list(key_names)
        existing_public_completions = make_sorted_list(filter(_is_public, existing_completions))
        ordered_completions = {c: None for c in key_names + existing_public_completions}
        matches = list(ordered_completions.keys())
        matches.append(_PRESERVE_ORDER_SENTINEL)
        return matches


# As of 20191218, if `DisplayOptions.CUSTOM_COMPLETION` is toggled on it results in a one-time registration of custom
# attribute completion per IPython session as opposed to supporting deregistration in the same session.
# The following is a list of tasks to make this more extensible and robust:
# - Create a CustomCompleterManager that manages registration and deregistration more cleanly.
# - Investigate using traitlets observers for on the fly toggling.
# - Investigate deregistration of custom completer.
# - Add configuration options to allow for other data structures such pandas DataFrames custom attribute
#   completion ordering.
def enable_custom_attribute_completion() -> None:
    """
    Registers the custom attribute completer and sets our monkey patched complete method to preserve the custom
    attribute completer ordering.
    If an object has keys then the completion results will show the keys then any existing completions
    in case insensitive sorted order.

    Notable side effect - this will disable Jedi support since IPCompleter will
    prefer to use Jedi completions over any custom completer.
    """

    _disable_jedi()

    _set__monkey_patched__complete()

    _set_custom_attribute_completion()


# Note, this module has side effects!
if riptable.TypeRegister.DisplayOptions.CUSTOM_COMPLETION:
    enable_custom_attribute_completion()
