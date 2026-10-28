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
from ..rt_enum import CategoryMode

# Keys under which riptable stores Categorical metadata on pandas objects so a
# riptable -> pandas -> riptable round-trip can restore the original
# ``CategoryMode`` (pandas itself only models a flat list of categories).
_RIPTABLE_CAT_ATTR = "riptable_categorical"
_RIPTABLE_CATS_ATTR = "riptable_categoricals"
_META_VERSION = 1
_NAN_MARKER = "__riptable_nan__"
_BYTES_MARKER = "__riptable_bytes__"


def _encode_scalar(value):
    """Encode a single category value so it can be stored in ``Series.attrs``."""
    if value is None:
        return None
    if isinstance(value, bytes):
        return {_BYTES_MARKER: value.decode("utf-8", "surrogateescape")}
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return {_NAN_MARKER: True}
    if isinstance(value, np.generic):
        return value.item()
    return value


def _decode_scalar(value):
    """Reverse :func:`_encode_scalar`."""
    if isinstance(value, dict):
        if value.get(_NAN_MARKER):
            return float("nan")
        if _BYTES_MARKER in value:
            return value[_BYTES_MARKER].encode("utf-8", "surrogateescape")
    return value


def _encode_values(arr) -> list:
    return [_encode_scalar(v) for v in np.asarray(arr).tolist()]


def _decode_values(values, dtype_str) -> np.ndarray:
    decoded = [_decode_scalar(v) for v in values]
    if dtype_str is None:
        return np.asarray(decoded)
    return np.asarray(decoded, dtype=np.dtype(dtype_str))


def _find_categorical_meta(series):
    """Return the riptable sidecar metadata stored on a pandas Series, if any."""
    attrs = getattr(series, "attrs", None) or {}
    meta = attrs.get(_RIPTABLE_CAT_ATTR)
    if isinstance(meta, dict) and meta.get("version") == _META_VERSION:
        return meta
    # When the Series came from a DataFrame built by ``Dataset.to_pandas`` the
    # per-column metadata lives in ``DataFrame.attrs`` (mirrored onto columns).
    by_column = attrs.get(_RIPTABLE_CATS_ATTR)
    if isinstance(by_column, dict):
        meta = by_column.get(getattr(series, "name", None))
        if isinstance(meta, dict) and meta.get("version") == _META_VERSION:
            return meta
    return None


def _build_categorical_meta(arr) -> dict:
    """Capture everything needed to rebuild a riptable Categorical exactly."""
    mode = arr.category_mode
    meta = {
        "version": _META_VERSION,
        "mode": int(mode),
        "base_index": None if arr.base_index is None else int(arr.base_index),
        "ordered": bool(arr.ordered),
        "codes_dtype": str(np.asarray(arr).dtype),
        "invalid_category": _encode_scalar(arr.invalid_category),
    }
    if mode in (CategoryMode.Dictionary, CategoryMode.IntEnum):
        meta["mapping"] = [[int(code), _encode_scalar(label)] for code, label in arr.category_mapping.items()]
    elif mode == CategoryMode.MultiKey:
        cat_dict = arr.category_dict
        names = list(cat_dict.keys())
        meta["multikey_categories"] = {name: _encode_values(cat_dict[name]) for name in names}
        meta["multikey_dtypes"] = {name: str(np.asarray(cat_dict[name]).dtype) for name in names}
        n_combined = len(cat_dict[names[0]]) if names else 0
        meta["multikey_combined"] = [
            [_encode_scalar(cat_dict[name][i]) for name in names] for i in range(n_combined)
        ]
    else:
        cat_arr = arr.category_array
        meta["categories"] = _encode_values(cat_arr)
        meta["category_dtype"] = str(np.asarray(cat_arr).dtype)
    return meta


def _scalars_match(a, b) -> bool:
    a = _decode_scalar(a)
    b = _decode_scalar(b)
    if isinstance(a, bytes):
        a = a.decode("utf-8", "surrogateescape")
    if isinstance(b, bytes):
        b = b.decode("utf-8", "surrogateescape")
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return bool(a == b) or str(a) == str(b)


def _meta_matches(meta: dict, pandas_categories: np.ndarray, pandas_codes: np.ndarray) -> bool:
    """Check the pandas categorical still matches the sidecar metadata.

    Guards against restoring stale metadata after the user edited the
    pandas categories/codes; on mismatch the plain conversion is used.
    """
    mode = CategoryMode(meta["mode"])
    actual = np.asarray(pandas_categories).tolist()
    if mode in (CategoryMode.Dictionary, CategoryMode.IntEnum):
        expected = [_decode_scalar(label) for _, label in meta["mapping"]]
    elif mode == CategoryMode.MultiKey:
        # The flattened pandas labels cannot be recomputed cheaply here;
        # rely on the code range check below.
        expected = None
    else:
        decoded = [_decode_scalar(v) for v in meta["categories"]]
        keep = meta.get("pandas_keep_positions")
        expected = [decoded[p] for p in keep] if keep else decoded
    if expected is not None:
        if len(actual) != len(expected):
            return False
        if not all(_scalars_match(a, e) for a, e in zip(actual, expected, strict=True)):
            return False
    limit = len(actual)
    if mode == CategoryMode.MultiKey:
        limit = len(meta.get("multikey_combined", []))
    if len(pandas_codes) > 0 and int(np.max(pandas_codes)) >= limit:
        return False
    return True


def _reconstruct_categorical(meta: dict, pandas_codes: np.ndarray):
    """Rebuild the original riptable Categorical from sidecar metadata."""
    from ..rt_grouping import Grouping

    mode = CategoryMode(meta["mode"])
    base_index = meta.get("base_index")
    ordered = bool(meta.get("ordered", False))
    codes_dtype = np.dtype(meta.get("codes_dtype", "int64"))

    if mode in (CategoryMode.Dictionary, CategoryMode.IntEnum):
        pairs = meta["mapping"]
        mapping = {int(code): _decode_scalar(label) for code, label in pairs}
        ordered_codes = [int(code) for code, _ in pairs]
        codes = np.array(
            [ordered_codes[p] if p >= 0 else 0 for p in pandas_codes],
            dtype=codes_dtype,
        )
        return TypeRegister.Categorical(codes, categories=mapping)

    if mode == CategoryMode.MultiKey:
        base = 1 if base_index is None else base_index
        uniques = {
            name: _decode_values(values, meta["multikey_dtypes"][name])
            for name, values in meta["multikey_categories"].items()
        }
        codes = (np.asarray(pandas_codes, dtype=np.int64) + base).astype(codes_dtype)
        grouping = Grouping(codes, uniques, base_index=base, categorical=True, _trusted=True)
        return TypeRegister.Categorical(grouping)

    # Single-key modes (StringArray / NumericArray / Default).
    base = 1 if base_index is None else base_index
    categories = _decode_values(meta["categories"], meta.get("category_dtype"))
    pandas_codes = np.asarray(pandas_codes, dtype=np.int64)
    keep_positions = meta.get("pandas_keep_positions")
    if keep_positions:
        # Some original categories (e.g. a float NaN category) cannot be held
        # by pandas and were dropped; map pandas positions back to originals.
        keep = np.asarray(keep_positions, dtype=np.int64)
        orig_positions = np.where(pandas_codes >= 0, keep[np.clip(pandas_codes, 0, None)], -1)
    else:
        orig_positions = pandas_codes
    codes = orig_positions + base
    if base == 0:
        # pandas uses -1 for missing; a base-0 categorical has no filtered bin,
        # so this can only occur if the pandas frame was edited by hand.
        codes[codes < 0] = 0
    codes = codes.astype(codes_dtype, copy=False)
    invalid = _decode_scalar(meta.get("invalid_category"))
    grouping = Grouping(
        codes,
        categories=categories,
        base_index=base,
        ordered=ordered,
        categorical=True,
        _trusted=True,
    )
    return TypeRegister.Categorical(grouping, invalid=invalid)


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


def pandas_series_to_riptable(series: Union["pd.Series", "pd.Categorical"], tz: str = "UTC") -> TypeRegister.FastArray:
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
        categories = cat.categories.to_numpy() if hasattr(cat.categories, "to_numpy") else np.asarray(cat.categories)

        # Restore the original riptable categorical when sidecar metadata from
        # ``Dataset.to_pandas`` / ``fastarray_to_pandas_series`` is present.
        meta = _find_categorical_meta(series) if hasattr(series, "attrs") else None
        if meta is not None and _meta_matches(meta, categories, codes):
            return _reconstruct_categorical(meta, codes)

        # Plain pandas categorical: go through the Categorical constructor's
        # pandas fast track, which preserves the categories and ordering.
        # (The pandas missing code -1 becomes riptable's filtered bin 0.)
        return TypeRegister.Categorical(cat)
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
        return TypeRegister.DateTimeNano(np.asarray(series, dtype="i8"), from_tz="UTC", to_tz=_tz)
    elif dtype_kind == "m":
        arr = TypeRegister.FastArray(series, dtype="i8")
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
            single = arr  # already compatible with pandas; no special handling needed
        elif arr.category_mode in (CategoryMode.Dictionary, CategoryMode.MultiKey, CategoryMode.IntEnum):
            # Pandas does not have a notion of a IntEnum, Dictionary, and Multikey category mode.
            # Encode dictionary codes to a monotonically increasing sequence and construct
            # pandas Categorical as if it was a string or numeric array category mode.
            # The original mode is stored in the Series attrs (see ``_build_categorical_meta``)
            # so ``Dataset.from_pandas`` can restore it on the way back.
            old_category_mode = arr.category_mode
            single = arr.as_singlekey()
            warnings.warn(
                f"Series  converted from {repr(CategoryMode(old_category_mode))} to {repr(CategoryMode(single.category_mode))}.",
                stacklevel=2,
            )
        else:
            raise NotImplementedError(
                f"Dataset.to_pandas: Unhandled category mode {repr(CategoryMode(arr.category_mode))}"
            )

        base_index = 0 if single.base_index is None else single.base_index
        codes = np.asarray(single) - base_index
        categories = _to_unicode_if_string(single.category_array) if unicode else single.category_array

        # pandas categories cannot be null: drop null categories (e.g. a float
        # NaN category) and send their values to the pandas missing code (-1).
        categories_arr = np.asarray(categories)
        try:
            null_mask = pd.isna(categories_arr)
        except (TypeError, ValueError):
            null_mask = np.zeros(len(categories_arr), dtype=bool)
        if not isinstance(null_mask, np.ndarray) or null_mask.shape != categories_arr.shape:
            null_mask = np.zeros(len(categories_arr), dtype=bool)
        keep_positions = [int(i) for i in np.nonzero(~null_mask)[0]]
        if len(keep_positions) != len(categories_arr):
            pos_map = np.full(len(categories_arr), -1, dtype=np.int64)
            pos_map[keep_positions] = np.arange(len(keep_positions), dtype=np.int64)
            codes = np.where(codes >= 0, pos_map[np.clip(codes, 0, None)], -1)
            categories = categories_arr[keep_positions]
        else:
            categories = categories_arr

        cat_dtype = pd.CategoricalDtype(categories=list(categories), ordered=bool(original.ordered))
        out = pd.Series(pd.Categorical.from_codes(np.asarray(codes), dtype=cat_dtype))
        meta = _build_categorical_meta(original)
        if original.category_mode not in (CategoryMode.Dictionary, CategoryMode.IntEnum, CategoryMode.MultiKey):
            meta["pandas_keep_positions"] = keep_positions
        out.attrs[_RIPTABLE_CAT_ATTR] = meta
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
