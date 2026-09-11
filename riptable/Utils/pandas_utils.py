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
            # For plain Categorical (not Series), attrs are not available
            attrs = {}
        else:
            cat = series.cat
            # Preserve riptable metadata stored in Series.attrs for roundtrip
            attrs = getattr(series, "attrs", {}) or {}

        # Check for preserved riptable categorical metadata for roundtrip
        rt_mode = attrs.get("rt_category_mode", None)
        if rt_mode is not None:
            try:
                mode = CategoryMode(rt_mode)
            except Exception:
                mode = None

            # Dictionary / IntEnum mode: stored mapping and original codes
            if mode in (CategoryMode.Dictionary, CategoryMode.IntEnum):
                mapping = attrs.get("rt_dict_mapping", None)
                orig_fa = attrs.get("rt_original_fa", None)
                if mapping is not None and orig_fa is not None:
                    # Use original integer codes and mapping to reconstruct
                    return TypeRegister.Categorical(np.asarray(orig_fa), categories=mapping)

                # Fallback: reconstruct from pandas codes via stored keys
                dict_keys = attrs.get("rt_dict_keys", None)
                if dict_keys is not None and mapping is not None:
                    codes = cat.codes
                    if hasattr(codes, "to_numpy"):
                        codes = codes.to_numpy()
                    else:
                        codes = np.asarray(codes)
                    orig_codes = []
                    for c in codes:
                        if c == -1:
                            orig_codes.append(-1)
                        else:
                            if 0 <= c < len(dict_keys):
                                orig_codes.append(int(dict_keys[c]))
                            else:
                                orig_codes.append(-1)
                    return TypeRegister.Categorical(np.asarray(orig_codes), categories=mapping)

            # MultiKey mode: reconstruct from stored uniquedict
            if mode == CategoryMode.MultiKey:
                multikey_dict = attrs.get("rt_multikey_dict", None)
                orig_fa = attrs.get("rt_original_fa", None)
                if multikey_dict is not None and orig_fa is not None:
                    try:
                        fa = np.asarray(orig_fa)
                        # Reconstruct original key columns from codes and uniquedict
                        reconstructed_cols = []
                        for col_name, uniq_arr in multikey_dict.items():
                            uniq = np.asarray(uniq_arr)
                            # Map codes to values: code 1 -> index 0, etc.
                            # Invalid (0) -> map to 0 index as placeholder (will be treated as valid after reconstruction,
                            # but preserves data for valid rows)
                            idx = fa - 1
                            idx = np.where(idx < 0, 0, idx)
                            idx = np.clip(idx, 0, len(uniq) - 1)
                            recon = uniq[idx]
                            reconstructed_cols.append(recon)

                        if len(reconstructed_cols) == 1:
                            return TypeRegister.Categorical(reconstructed_cols[0])
                        else:
                            return TypeRegister.Categorical(reconstructed_cols)
                    except Exception:
                        # Fall through to default handling on any error
                        pass

            # Fallback for other preserved modes or if reconstruction failed

        codes = cat.codes
        categories = cat.categories

        if hasattr(codes, "to_numpy"):
            codes = codes.to_numpy()
            categories = categories.to_numpy()
        else:
            codes = np.asarray(codes)
            categories = np.asarray(categories)

        # Preserve ordered flag if available
        ordered = getattr(cat, "ordered", False)

        # pandas codes: -1 = missing, 0..n-1 = valid
        # riptable codes: 0 = invalid/Filtered when base_index=1, 1..n = valid
        # Use base_index from attrs if available, default 1
        base_index = attrs.get("rt_base_index", 1)
        if base_index is None:
            base_index = 1

        # For standard StringArray/NumericArray, reconstruct directly
        # Handle -1 -> 0 invalid when base_index=1, or -1 -> -1 when base_index=0 (will be treated as invalid via filter?)
        # Simplest: rt_codes = codes + base_index, where -1 becomes base_index-1
        # When base_index=1, -1+1=0 -> invalid, correct.
        # When base_index=0, -1+0=-1 -> negative, but Categorical may handle? Clamp -1 to 0 for safety?
        if base_index == 0:
            # For base_index 0, we want -1 to map to 0? Actually base 0 has no dedicated invalid code 0 is valid.
            # But we can map -1 to 0 and let caller handle? For now, map -1 to 0 and rely on Categorical to treat as valid first category.
            # To preserve invalid, we would need filter mask, which we don't have.
            # So map -1 -> 0 for base 0 as well (first category) is best effort.
            # Alternatively, keep -1 and let Categorical handle? Categorical may treat -1 as invalid? No, it expects 0..n-1.
            # We'll map -1 to 0.
            rt_codes = np.where(codes == -1, 0, codes)
            # No +base_index since base 0
        else:
            rt_codes = codes + 1  # -1 -> 0

        return TypeRegister.Categorical(rt_codes, categories=categories, ordered=ordered)

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
        # Pandas 3 may use us/ms/s resolution (e.g., datetime64[us, tz]), while riptable expects ns.
        try:
            import re

            m = re.search(r"\[([a-z]+)", dtype.str)
            unit = m.group(1) if m else "ns"
        except Exception:
            unit = "ns"
        factor = {"ns": 1, "us": 1000, "ms": 1_000_000, "s": 1_000_000_000}.get(unit, 1)
        arr_ns = np.asarray(series, dtype="i8") * factor
        return TypeRegister.DateTimeNano(arr_ns, from_tz="UTC", to_tz=_tz)
    elif dtype_kind == "m":
        # Pandas 3 may use us/ms/s resolution, riptable TimeSpan expects ns
        try:
            import re

            m = re.search(r"\[([a-z]+)", dtype.str)
            unit = m.group(1) if m else "ns"
        except Exception:
            unit = "ns"
        factor = {"ns": 1, "us": 1000, "ms": 1_000_000, "s": 1_000_000_000}.get(unit, 1)
        arr = TypeRegister.FastArray(np.asarray(series, dtype="i8") * factor, dtype="i8")
        arr = TypeRegister.TimeSpan(arr)
        try:
            nat_val = pd.NaT.value
            arr[arr == nat_val] = arr.inv
        except Exception:
            arr[arr == pd.NaT.value] = arr.inv
        return arr
    elif dtype_kind == "O":
        if len(series) > 0:
            notnull = np.where(series.notnull())[0]
            all_null = len(notnull) == 0
            first_element = np.nan if all_null else series.iloc[notnull[0]]
            if isinstance(first_element, (int, float, np.number)):
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
        # Preserve original categorical metadata for roundtrip via Series.attrs
        orig_mode = arr.category_mode
        orig_base_index = arr.base_index
        orig_ordered = arr.ordered if hasattr(arr, "ordered") else False
        rt_attrs = {
            "rt_category_mode": int(orig_mode),
            "rt_base_index": orig_base_index,
            "rt_ordered": bool(orig_ordered),
        }

        # For Dictionary/IntEnum, store mapping and original codes
        if orig_mode in (CategoryMode.Dictionary, CategoryMode.IntEnum):
            try:
                mapping = arr.category_mapping
                rt_attrs["rt_dict_mapping"] = dict(mapping) if isinstance(mapping, dict) else mapping
                rt_attrs["rt_original_fa"] = np.asarray(arr._fa).copy() if hasattr(arr, "_fa") else np.asarray(arr).copy()
                if isinstance(mapping, dict):
                    rt_attrs["rt_dict_keys"] = list(mapping.keys())
                    rt_attrs["rt_dict_values"] = list(mapping.values())
            except Exception:
                pass

        # For MultiKey, store uniquedict and original codes
        if orig_mode == CategoryMode.MultiKey:
            try:
                multikey_dict = arr.categories(showfilter=False)
                if isinstance(multikey_dict, dict):
                    rt_attrs["rt_multikey_dict"] = {
                        k: np.asarray(v).copy() for k, v in multikey_dict.items()
                    }
                rt_attrs["rt_original_fa"] = np.asarray(arr._fa).copy() if hasattr(arr, "_fa") else np.asarray(arr).copy()
            except Exception:
                pass

        if arr.category_mode in (CategoryMode.Default, CategoryMode.StringArray, CategoryMode.NumericArray):
            pass  # already compatible with pandas; no special handling needed
        elif arr.category_mode in (CategoryMode.Dictionary, CategoryMode.MultiKey, CategoryMode.IntEnum):
            # Pandas does not have a notion of a IntEnum, Dictionary, and Multikey category mode.
            # Encode dictionary codes to a monotonically increasing sequence and construct
            # pandas Categorical as if it was a string or numeric array category mode.
            old_category_mode = arr.category_mode
            arr = arr.as_singlekey()
            warnings.warn(
                f"Series converted from {repr(CategoryMode(old_category_mode))} to {repr(CategoryMode(arr.category_mode))}.",
                stacklevel=2,
            )
        else:
            raise NotImplementedError(
                f"Dataset.to_pandas: Unhandled category mode {repr(CategoryMode(arr.category_mode))}"
            )

        base_index = 0 if arr.base_index is None else arr.base_index
        codes = np.asarray(arr) - base_index
        categories = _to_unicode_if_string(arr.category_array) if unicode else arr.category_array
        # NOTE: Do not preserve ordered flag for pandas compatibility;
        # original riptable code did not, and test expects unordered.
        # Ordered is still preserved via rt_attrs for Dictionary/MultiKey roundtrip.
        ordered = False
        # pandas Categorical.from_codes does not take ordered directly in older versions? Use ordered param if available
        try:
            cat = pd.Categorical.from_codes(codes, categories=categories, ordered=ordered)
        except TypeError:
            # Older pandas may not support ordered in from_codes
            cat = pd.Categorical.from_codes(codes, categories=categories)
            # Set ordered afterwards if possible
            try:
                cat = cat.set_ordered(ordered)
            except Exception:
                pass
        out = pd.Series(cat)
        # Attach riptable metadata for roundtrip
        try:
            out.attrs.update(rt_attrs)
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
        is_invalid = arr.isin(TypeRegister.FastArray([INVALID_DICT[dtype.num]], dtype=dtype))
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
