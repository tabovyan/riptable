"""Start a global IPython shell for the tooling integration tests.

These tests drive completions through ``get_ipython()`` (mirroring IPython's
own test suite, which starts its shell via
``IPython.testing.globalipapp``). Without a running shell, ``get_ipython()``
returns ``None`` and every test that touches the completer fails before
any completion is attempted. This conftest only applies to this directory;
IPython remains an optional dependency and the test modules skip themselves
if it cannot be imported.
"""


def pytest_configure() -> None:
    try:
        from IPython.testing.globalipapp import start_ipython

        start_ipython()
    except Exception:
        pass
