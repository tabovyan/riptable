"""
Utility function for rt.
These functions (may) have dependence on additional libraries and therefore should
_NOT_ be imported in __init__.py or any other such core (like rt_appconfig.py).
"""

__all__ = [
    "dataset_as_pandas_df",
    "dataset_from_pandas_df",
    "fastarray_to_pandas_series",
    "pandas_series_to_riptable",
]

import warnings
from typing import TYPE_CHECKING, Union

import numpy as np

if TYPE_CHECKING:
    # pandas is an optional dependency.
    try:
        import pandas as pd
    except ImportError:
        pass

from .. import TypeRegister, INVALID_DICT
from ..rt_dataset import Dataset
from ..rt_enum import CategoryMode, int_dtype_from_len

# Key under which ``Dataset.to_pandas`` / ``fastarray_to_pandas_series`` stash
# the riptable-side description of a categorical column that pandas itself
# cannot express (see ``_categorical_metadata``). ``Dataset.from_pandas`` /
# ``pandas_series_to_riptable`` read it back to restore the original
# categorical; a frame without it (or with categories that no longer match)
# falls back to a plain single-key conversion.
_RIPTABLE_CATEGORICAL_ATTRS_KEY = "riptable_categorical"
_CATEGORICAL_METADATA_VERSION = 1


def dataset_from_pandas_df(df, tz="UTC"):
    """
    This function is deprecated, please use riptable.Dataset.from_pandas.

    Creates a riptable Dataset from a pandas DataFrame. Pandas categoricals
    and datetime arrays are converted to their riptable counterparts.
    Any timezone-unaware datetime arrays (or those using a timezone not
    recognized by riptable) are localized to the timezone specified by the
    tz parameter.

    Recognized pandas timezones:
        UTC, GMT, US/Eastern, and Europe/Dublin

    Parameters
    ----------
    df: DataFrame
        The pandas DataFrame to be converted
    tz: string
        A riptable-supported timezone ('UTC', 'NYC', 'DUBLIN', 'GMT')

    Returns
    -------
    Dataset

    See Also
    --------
    riptable.Dataset.from_pandas
    riptable.Dataset.to_pandas
    """
    warnings.warn(
        "dataset_from_pandas_df is deprecated and will be removed in future release, "
        "please use riptable.Dataset.from_pandas method",
        FutureWarning,
        stacklevel=2,
    )
    return Dataset.from_pandas(df, tz)


def dataset_as_pandas_df(ds):
    """
    This function is deprecated, please use riptable.Dataset.as_pandas_df method.

    Create a pandas DataFrame from a riptable Dataset.
    Will attempt to preserve single-key categoricals, otherwise will appear as
    an index array. Any bytestrings will be converted to unicode.

    Parameters
    ----------
    ds : Dataset
        The riptable Dataset to be converted.

    Returns
    -------
    DataFrame

    See Also
    --------
    riptable.Dataset.to_pandas
    """
    warnings.warn(
        "dataset_as_pandas_df is deprecated and will be removed in future release, "
        "please use riptable.Dataset.to_pandas method",
        FutureWarning,
        stacklevel=2,
    )
    return ds.to_pandas()


def _pandas_categories_equal(stored, categories) -> bool:
    """Check stored ``pandas_categories`` against the categories of a pandas categorical.

    Only the values are compared (dtypes may differ, e.g. float32 vs float64);
    float NaN categories compare equal. Used to reject stale metadata when a
    user edited the categories after ``to_pandas``.
    """
    try:
        stored_arr = np.asarray(stored)
        cats_arr = np.asarray(categories)
    except Exception:
        return False
    if stored_arr.shape != cats_arr.shape:
        return False
    if stored_arr.size == 0:
        return True
    try:
        # equal_nan only applies to float categories; asking numpy to test
        # NaN-ness of strings/objects raises TypeError.
        if stored_arr.dtype.kind in "fc" or cats_arr.dtype.kind in "fc":
            return bool(np.array_equal(stored_arr, cats_arr, equal_nan=True))
        return bool(np.array_equal(stored_arr, cats_arr))
    except Exception:
        return False


def _categorical_metadata(arr, pandas_categories):
    """Capture the riptable-side description of a categorical for round-tripping.

    Pandas ``Categorical`` only carries an ordered list of categories, integer
    codes (-1 for missing), and an ``ordered`` flag. The following riptable
    structure has no pandas equivalent and is stashed in ``.attrs`` so
    ``Dataset.from_pandas`` can restore it:

    * ``CategoryMode`` (``StringArray`` / ``NumericArray`` / ``Dictionary`` /
      ``IntEnum`` / ``MultiKey``) and ``base_index`` (0 / 1 / ``None``).
    * For ``Dictionary`` / ``IntEnum`` modes: the int -> label mapping and the
      code to use for rows whose code is not in the mapping (pandas can only
      express those as missing).
    * For ``MultiKey`` mode: the per-key category arrays and their names.
    * The category array itself, preserving its dtype (bytes vs unicode,
      int vs float kind), for single-key modes.

    ``pandas_categories`` (the categories as emitted to pandas) is stored as
    well so stale metadata can be detected on the way back.
    """
    mode = arr.category_mode
    metadata = {
        "version": _CATEGORICAL_METADATA_VERSION,
        "mode": int(mode),
        "base_index": arr.base_index,
        "ordered": bool(arr.ordered),
        "pandas_categories": np.asarray(pandas_categories),
    }
    if arr.isenum:
        mapping = dict(arr.category_mapping)
        keys = list(mapping.keys())
        # Rows whose code is not in the mapping become missing in pandas;
        # remember the first such code so it can be restored for them.
        codes = np.asarray(arr)
        invalid_code = 0
        if len(codes) and len(keys):
            unmapped = ~np.isin(codes, np.asarray(keys))
            if unmapped.any():
                invalid_code = int(codes[np.argmax(unmapped)])
        metadata["mapping"] = mapping
        metadata["invalid_code"] = invalid_code
    elif arr.ismultikey:
        metadata["categories"] = {name: np.array(col, copy=True) for name, col in arr.category_dict.items()}
    else:
        metadata["categories"] = np.array(arr.category_array, copy=True)
    return metadata


def _categorical_from_grouping(codes, categories, ordered, base_index):
    """Build a single/multi-key categorical from codes + categories, honouring
    ``ordered`` and ``base_index`` exactly.

    Constructing via ``Categorical(codes, categories)`` derives ``ordered``
    from whether the categories happen to be sorted, so it cannot restore an
    unordered-but-ordered (or sorted-but-unordered) categorical. Going
    through ``Grouping`` with ``_trusted=True`` keeps the requested flags;
    the codes come from a pandas categorical, so they are already in range.
    """
    from ..rt_grouping import Grouping

    grouping = Grouping(
        codes,
        categories=categories,
        ordered=ordered,
        base_index=base_index,
        categorical=True,
        _trusted=True,
    )
    return TypeRegister.Categorical(grouping)


def _restore_categorical(codes, categories, metadata):
    """Rebuild the original categorical from pandas codes/categories + metadata.

    Returns ``None`` when the metadata is missing, malformed, or no longer
    matches the pandas categorical (e.g. the user renamed/reordered the
    categories), in which case the caller falls back to a generic
    single-key conversion.
    """
    if not isinstance(metadata, dict):
        return None
    if metadata.get("version") != _CATEGORICAL_METADATA_VERSION:
        return None
    try:
        mode = CategoryMode(metadata.get("mode"))
    except (TypeError, ValueError):
        return None
    if not _pandas_categories_equal(metadata.get("pandas_categories"), categories):
        return None

    codes = np.asarray(codes)
    if codes.size and codes.dtype.kind not in "iu":
        return None
    ordered = bool(metadata.get("ordered", False))

    if mode in (CategoryMode.Dictionary, CategoryMode.IntEnum):
        mapping = metadata.get("mapping")
        if not isinstance(mapping, dict) or not mapping:
            return None
        keys = list(mapping.keys())
        if len(keys) != len(np.asarray(categories)):
            return None
        if not all(isinstance(k, (int, np.integer)) and not isinstance(k, bool) for k in keys):
            return None
        if not all(isinstance(v, (str, bytes, np.str_, np.bytes_)) for v in mapping.values()):
            return None
        invalid_code = metadata.get("invalid_code", 0)
        if not isinstance(invalid_code, (int, np.integer)) or isinstance(invalid_code, bool):
            return None
        key_arr = np.asarray(keys)
        rt_codes = key_arr[np.maximum(codes, 0)]
        rt_codes = rt_codes.astype(np.int64, copy=False)
        rt_codes[codes < 0] = int(invalid_code)
        return TypeRegister.Categorical(rt_codes, dict(mapping))

    base_index = metadata.get("base_index")
    if base_index not in (0, 1):
        return None
    if base_index == 0 and codes.size and bool((codes < 0).any()):
        # Base-0 categoricals have no invalid bin; a missing value can only
        # be represented by promoting to base-1 (missing -> Filtered).
        base_index = 1
    rt_codes = codes.astype(int_dtype_from_len(len(np.asarray(categories))), copy=False) + base_index

    if mode in (CategoryMode.StringArray, CategoryMode.NumericArray, CategoryMode.Default):
        stored = metadata.get("categories")
        if not isinstance(stored, np.ndarray) or len(stored) != len(np.asarray(categories)):
            return None
        return _categorical_from_grouping(rt_codes, stored, ordered, base_index)

    if mode == CategoryMode.MultiKey:
        stored = metadata.get("categories")
        if not isinstance(stored, dict) or not stored:
            return None
        if not all(isinstance(v, np.ndarray) for v in stored.values()):
            return None
        ncombos = len(next(iter(stored.values())))
        if any(len(v) != ncombos for v in stored.values()) or ncombos != len(np.asarray(categories)):
            return None
        return _categorical_from_grouping(rt_codes, dict(stored), ordered, base_index)

    return None


def pandas_series_to_riptable(
    series: Union["pd.Series", "pd.Categorical"], tz: str = "UTC", categorical_metadata=None
) -> TypeRegister.FastArray:
    import pandas as pd

    dtype = series.dtype
    dtype_kind = dtype.kind
    if hasattr(pd, "CategoricalDtype"):
        iscat = isinstance(dtype, pd.CategoricalDtype)
    else:
        iscat = dtype.num == 100

    iscat = iscat or isinstance(series, pd.Categorical)

    if iscat:
        if isinstance(series, pd.Categorical):
            cat = series
        else:
            cat = series.cat

        codes = np.asarray(cat.codes)
        categories = np.asarray(cat.categories)
        ordered = bool(cat.ordered)

        if categorical_metadata is None:
            # Metadata stashed on the Series itself (e.g. by
            # ``fastarray_to_pandas_series`` or ``Dataset.__init__``) is
            # honored the same way as metadata passed explicitly.
            categorical_metadata = getattr(series, "attrs", {}).get(_RIPTABLE_CATEGORICAL_ATTRS_KEY)
        # Metadata comes from user-editable ``.attrs``; a restore failure
        # must degrade to the generic conversion, never raise.
        try:
            restored = _restore_categorical(codes, categories, categorical_metadata)
        except Exception:
            restored = None
        if restored is not None:
            return restored

        # Generic conversion: pandas missing (-1) becomes the riptable
        # invalid bin 0, unused categories are preserved in order, and the
        # pandas ``ordered`` flag is preserved rather than re-derived from
        # whether the categories happen to be sorted.
        return _categorical_from_grouping(codes + 1, categories, ordered, base_index=1)
    elif hasattr(pd, "Int8Dtype") and isinstance(
        dtype,
        (
            pd.Int8Dtype,
            pd.Int16Dtype,
            pd.Int32Dtype,
            pd.Int64Dtype,
            pd.UInt8Dtype,
            pd.UInt16Dtype,
            pd.UInt32Dtype,
            pd.UInt64Dtype,
        ),
    ):
        sentinel = INVALID_DICT[dtype.numpy_dtype.num]
        return TypeRegister.FastArray(series.fillna(sentinel), dtype=dtype.numpy_dtype)
    elif dtype_kind == "M":
        try:
            _tz = str(dtype.tz)
        except AttributeError:
            _tz = tz
        # Pandas 3 holds datetimes at s/ms/us/ns resolution; DateTimeNano
        # stores nanoseconds, so normalize to ns first (naive values keep
        # their wall time, tz-aware values are converted to UTC instants).
        ns_values = np.asarray(series.to_numpy(dtype="datetime64[ns]"), dtype="i8")
        return TypeRegister.DateTimeNano(ns_values, from_tz="UTC", to_tz=_tz)
    elif dtype_kind == "m":
        ns_values = np.asarray(series.to_numpy(dtype="timedelta64[ns]"), dtype="i8")
        arr = TypeRegister.FastArray(ns_values)
        arr = TypeRegister.TimeSpan(arr)
        arr[arr == pd.NaT.value] = arr.inv
        return arr
    elif dtype_kind == "O":
        if len(series) > 0:
            notnull = np.where(series.notnull())[0]
            all_null = len(notnull) == 0
            first_element = np.nan if all_null else series.iloc[notnull[0]]
            if isinstance(first_element, (int, float, np.number)):
                # An object array with number (int or float) in it probably means there is
                # NaN in it so convert to float64.
                new_col = np.asarray(series, dtype="f8")
            else:
                try:
                    new_col = np.asarray(series, dtype="S")
                except UnicodeEncodeError:
                    new_col = np.asarray(series, dtype="U")
        else:
            new_col = np.asarray(series, dtype="S")
        return TypeRegister.FastArray(new_col)
    else:
        return TypeRegister.FastArray(series)


def fastarray_to_pandas_series(
    arr: Union[TypeRegister.FastArray, TypeRegister.Categorical], unicode: bool = True, use_nullable: bool = True
) -> "pd.Series":
    import pandas as pd

    def _to_unicode_if_string(arr):
        if arr.dtype.char == "S":
            arr = arr.astype("U")
        return arr

    dtype = arr.dtype
    if isinstance(arr, TypeRegister.Categorical):
        original = arr
        if arr.category_mode in (CategoryMode.Default, CategoryMode.StringArray, CategoryMode.NumericArray):
            pass  # already compatible with pandas; no special handling needed
        elif arr.category_mode in (CategoryMode.Dictionary, CategoryMode.MultiKey, CategoryMode.IntEnum):
            # Pandas does not have a notion of a IntEnum, Dictionary, and Multikey category mode.
            # Encode dictionary codes to a monotonically increasing sequence and construct
            # pandas Categorical as if it was a string or numeric array category mode.
            old_category_mode = arr.category_mode
            arr = arr.as_singlekey()
            warnings.warn(
                f"Series  converted from {repr(CategoryMode(old_category_mode))} to {repr(CategoryMode(arr.category_mode))}.",
                stacklevel=2,
            )
        else:
            raise NotImplementedError(
                f"Dataset.to_pandas: Unhandled category mode {repr(CategoryMode(arr.category_mode))}"
            )

        base_index = 0 if arr.base_index is None else arr.base_index
        codes = np.asarray(arr) - base_index
        categories = _to_unicode_if_string(arr.category_array) if unicode else arr.category_array
        pandas_cat = pd.Categorical.from_codes(codes, categories=categories, ordered=bool(arr.ordered))
        out = pd.Series(pandas_cat)
        # Stash what pandas cannot express so the round-trip through
        # ``pandas_series_to_riptable`` restores the original categorical.
        # Best effort: if capturing it fails, the Series is still a valid
        # pandas categorical and converts back generically.
        try:
            out.attrs[_RIPTABLE_CATEGORICAL_ATTRS_KEY] = _categorical_metadata(original, pandas_cat.categories)
        except Exception:
            pass
    elif isinstance(arr, TypeRegister.DateTimeNano):
        utc_datetime = pd.DatetimeIndex(arr._np, tz="UTC")
        tz_datetime = utc_datetime.tz_convert(arr._timezone._to_tz)
        out = pd.Series(tz_datetime)
    elif isinstance(arr, TypeRegister.TimeSpan):
        out = pd.Series(arr._np, dtype="timedelta64[ns]")
    # TODO: riptable.DateSpan doesn't have a counterpart in pandas, what do we want to do?
    elif use_nullable and np.issubdtype(dtype, np.integer):
        # N.B. Has to use the same dtype for `isin` otherwise riptable will convert the dtype
        #      and the invalid value.
        is_invalid = arr.isin(TypeRegister.FastArray([INVALID_DICT[dtype.num]], dtype=dtype))
        # N.B. Have to make a copy of the array to numpy array otherwise pandas seg
        #      fault in DataFrame.
        # NOTE: not all versions of pandas have pd.arrays
        if hasattr(pd, "arrays"):
            arr = pd.arrays.IntegerArray(np.array(arr), mask=is_invalid)
        else:
            arr = np.array(arr)
        out = pd.Series(arr)
    elif unicode and arr.dtype.char == "S":
        out = pd.Series(arr.astype("U"))
    else:
        out = pd.Series(arr)

    out.name = getattr(arr, "_name", None)
    return out
