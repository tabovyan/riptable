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
    "_extract_categorical_meta",
    "_reconstruct_categorical_from_meta",
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


def _get_rt_categorical_meta_from_series(series) -> Union[dict, None]:
    """Extract riptable categorical metadata from pandas Series attrs if present."""
    # Metadata is stored in Series.attrs['_rt_cat_meta'] for direct Series conversion
    # For DataFrame conversion, metadata lives in DataFrame.attrs, handled in Dataset.from_pandas
    if hasattr(series, "attrs"):
        meta = series.attrs.get("_rt_cat_meta")
        if meta is not None:
            return meta
    return None


def _reconstruct_categorical_from_meta(codes_pandas, categories_pandas, meta: dict):
    """Reconstruct riptable Categorical from stored metadata and pandas codes/categories."""
    # meta contains: mode, ordered, base_index, codes (rt codes), categories_data, invalid_category
    # For roundtrip preservation we use stored rt codes and categories_data directly
    mode = CategoryMode(meta["mode"])
    ordered = meta.get("ordered", False)
    base_index = meta.get("base_index", 1)
    rt_codes = meta.get("rt_codes")
    categories_data = meta.get("categories_data")
    invalid_category = meta.get("invalid_category")

    if rt_codes is not None:
        # Use stored codes for perfect roundtrip
        if mode == CategoryMode.Dictionary or mode == CategoryMode.IntEnum:
            # categories_data is int->str dict
            # rt_codes are the original int keys (e.g., 0,2,32)
            # Reconstruct via Categorical(codes, categories=dict)
            return TypeRegister.Categorical(rt_codes, categories=categories_data)
        elif mode == CategoryMode.MultiKey:
            # categories_data is dict of arrays (uniquedict)
            # rt_codes are indices (1-based)
            # For multikey, Categorical(codes, categories=uniquedict) expects codes as indices?
            # Actually multikey categorical is constructed from dict + codes handling is via Grouping.
            # Simplest: reconstruct via grouping metadata: we have codes and uniquedict.
            # Use internal fast path: create Categorical via _from_categorical with grouping?
            # Instead, use Categorical with values as codes and categories as uniquedict,
            # but need to handle base_index.
            # For multikey, the constructor Categorical(values, categories) where categories is dict
            # does not support passing codes + uniquedict directly. We need to use grouping.
            # Workaround: create a temporary categorical from uniquedict and then map codes?
            # Alternative: store expanded logic - use codes and uniquedict to build via Grouping.
            # Let's use the stored rt_codes and uniquedict to reconstruct via internal API:
            # We can create Categorical from codes + categories dict using the same path as dict mode?
            # For multikey, we need to handle differently: use Dataset-like reconstruction.
            # Simplest: if rt_codes and uniquedict are stored, we can reconstruct by creating
            # a Categorical that has those codes as underlying array and uniquedict as categories.
            # The public API for multikey is Categorical(dict_of_arrays) for unique creation,
            # but for reconstruction with existing codes, we need to use low-level Grouping.
            from ..rt_grouping import Grouping

            # categories_data is dict of FastArray
            # rt_codes is FastArray of indices (1-based)
            # Build grouping from codes + uniquedict
            # We can use Categorical via grouping: create grouping then categorical
            # For multikey, grouping holds uniquedict and ikey = codes
            # Use internal constructor
            # To avoid complexity, we store also the original categorical's grouping unique arrays
            # and reconstruct by creating a Categorical from the dict and then setting its ikey?
            # Simpler: use the generic path: Categorical(rt_codes, categories=categories_data)
            # For multikey, categories_data is a dict, but Categorical.__new__ with values as codes
            # and categories as dict will treat values as indices? Let's try: for multikey, values
            # should be dict, not codes. So we need custom.
            # Instead, we will reconstruct via low-level: create a Grouping object from codes and uniquedict
            # then create Categorical from grouping.
            grouping = Grouping(
                rt_codes,
                categories=categories_data,
                base_index=base_index if base_index is not None else 1,
                categorical=True,
            )
            return TypeRegister.Categorical(grouping)
        else:  # StringArray, NumericArray, Default
            # rt_codes are 1-based indices (0 invalid)
            # categories_data is array of categories
            return TypeRegister.Categorical(rt_codes, categories=categories_data, ordered=ordered, base_index=base_index if base_index is not None else 1)
    # Fallback: generic reconstruction from pandas codes+categories (no stored rt_codes)
    ordered_pandas = False
    # pandas 3: Categorical has ordered attribute
    # We already have codes_pandas and categories_pandas
    # This path is for generic pandas categoricals without rt metadata
    return TypeRegister.Categorical(codes_pandas + 1, categories=categories_pandas, ordered=ordered)


def pandas_series_to_riptable(series: Union["pd.Series", "pd.Categorical"], tz: str = "UTC") -> TypeRegister.FastArray:
    import pandas as pd

    # First check for riptable metadata in Series attrs (direct Series roundtrip)
    rt_meta = _get_rt_categorical_meta_from_series(series)
    if rt_meta is not None:
        # We have stored metadata, use it for perfect roundtrip
        # Still need pandas codes/categories for fallback, but prefer stored
        # Extract pandas categorical for codes (if available)
        dtype = series.dtype
        if isinstance(dtype, pd.CategoricalDtype) or isinstance(series, pd.Categorical):
            if isinstance(series, pd.Categorical):
                cat = series
            else:
                cat = series.cat
            codes = cat.codes
            categories = cat.categories
            if hasattr(codes, "to_numpy"):
                codes = codes.to_numpy()
                categories = categories.to_numpy()
            else:
                codes = np.asarray(codes)
                categories = np.asarray(categories)
            return _reconstruct_categorical_from_meta(codes, categories, rt_meta)

    dtype = series.dtype
    dtype_kind = dtype.kind
    if hasattr(pd, "CategoricalDtype"):
        iscat = isinstance(dtype, pd.CategoricalDtype)
    else:
        iscat = dtype.num == 100

    iscat = iscat or isinstance(series, pd.Categorical)

    if iscat:
        # Use fast path: pass pandas Categorical directly to riptable Categorical
        # This preserves ordered flag and handles categories correctly (including pandas 3 ArrowStringArray)
        if isinstance(series, pd.Categorical):
            # Direct Categorical object
            return TypeRegister.Categorical(series)
        else:
            # Series with categorical dtype: use its values (which is pandas Categorical)
            # series.values is pandas Categorical for categorical dtype
            try:
                pd_cat = series.values
                if isinstance(pd_cat, pd.Categorical):
                    return TypeRegister.Categorical(pd_cat)
            except Exception:
                pass
            # Fallback: extract codes/categories manually
            cat = series.cat
            codes = cat.codes
            categories = cat.categories
            ordered = bool(getattr(cat, "ordered", False))

            # check for newer version of pandas
            if hasattr(codes, "to_numpy"):
                codes = codes.to_numpy()
                categories = categories.to_numpy()
            else:
                codes = np.asarray(codes)
                categories = np.asarray(categories)

            # Pandas uses -1 for NA, riptable uses 0 for invalid and 1-based indexing
            # Preserve ordered flag
            return TypeRegister.Categorical(codes + 1, categories=categories, ordered=ordered)
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
        # Pandas 3 defaults to datetime64[us] (microseconds) while riptable uses ns.
        # Convert to ns if needed.
        try:
            # Try to get resolution from dtype (pandas 3)
            # dtype is e.g., datetime64[us, UTC] or datetime64[ns]
            dtype_str = str(dtype)
            if "[us" in dtype_str:
                # microseconds -> nanoseconds (x1000)
                arr = np.asarray(series, dtype="i8") * 1000
            elif "[ms" in dtype_str:
                arr = np.asarray(series, dtype="i8") * 1_000_000
            elif "[s" in dtype_str:
                arr = np.asarray(series, dtype="i8") * 1_000_000_000
            else:
                arr = np.asarray(series, dtype="i8")
        except Exception:
            arr = np.asarray(series, dtype="i8")
        return TypeRegister.DateTimeNano(arr, from_tz="UTC", to_tz=_tz)
    elif dtype_kind == "m":
        # Pandas 3 defaults to timedelta64[us], convert to ns if needed
        try:
            dtype_str = str(dtype)
            if "[us" in dtype_str:
                arr_np = np.asarray(series, dtype="i8") * 1000
            elif "[ms" in dtype_str:
                arr_np = np.asarray(series, dtype="i8") * 1_000_000
            elif "[s" in dtype_str:
                arr_np = np.asarray(series, dtype="i8") * 1_000_000_000
            else:
                arr_np = np.asarray(series, dtype="i8")
        except Exception:
            arr_np = np.asarray(series, dtype="i8")
        arr = TypeRegister.FastArray(arr_np, dtype="i8")
        arr = TypeRegister.TimeSpan(arr)
        # pd.NaT.value is ns, need to handle us case
        try:
            nat_val = pd.NaT.value
            # If series was us, nat_val in us is same int? Actually pd.NaT.value is ns (iNaT = -9223372036854775808)
            # For us, the invalid sentinel in riptable is same (it uses i8 min)
            arr[arr == nat_val] = arr.inv
        except Exception:
            pass
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


def _extract_categorical_meta(arr) -> dict:
    """Extract metadata from riptable Categorical for roundtrip preservation."""
    mode = arr.category_mode
    # ordered flag: Categorical may have _ordered or grouping.isordered
    ordered = False
    try:
        ordered = bool(arr._ordered) if arr._ordered is not None else False
    except AttributeError:
        pass
    if not ordered:
        try:
            ordered = bool(arr.grouping.isordered)
        except Exception:
            ordered = False

    base_index = arr.base_index
    # rt_codes: underlying integer array (1-based with 0 invalid, or raw codes for dict modes)
    try:
        rt_codes = arr._fa.copy() if hasattr(arr, "_fa") else np.asarray(arr).copy()
        # Ensure FastArray type for consistency, but store as numpy for attrs pickling
        # Convert to numpy array for storage (attrs may hold FastArray but numpy is safer)
        rt_codes = np.asarray(rt_codes)
    except Exception:
        rt_codes = np.asarray(arr)

    categories_data = None
    invalid_category = None
    try:
        invalid_category = arr._categories_wrap._invalid_category
    except Exception:
        invalid_category = None

    if mode in (CategoryMode.StringArray, CategoryMode.NumericArray, CategoryMode.Default):
        try:
            categories_data = np.asarray(arr.category_array)
        except Exception:
            categories_data = None
    elif mode in (CategoryMode.Dictionary, CategoryMode.IntEnum):
        # int -> str mapping
        try:
            # grouping._enum holds the mapping
            enum_obj = arr.grouping._enum
            # _int_to_str_dict is dict[int, str]
            categories_data = dict(enum_obj._int_to_str_dict)
        except Exception:
            # fallback: try category_dict which for dict mode returns {'key_0': array of strings}
            # but we need int->str mapping; if not available, leave None
            categories_data = None
    elif mode == CategoryMode.MultiKey:
        try:
            # uniquedict: dict of arrays
            uniquedict = arr.category_dict
            # Convert each FastArray to numpy for storage
            categories_data = {k: np.asarray(v) for k, v in uniquedict.items()}
        except Exception:
            categories_data = None

    meta = {
        "mode": int(mode),
        "ordered": bool(ordered),
        "base_index": base_index,
        "rt_codes": rt_codes,
        "categories_data": categories_data,
        "invalid_category": invalid_category,
    }
    return meta


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
        # Preserve original categorical for metadata extraction before any conversion
        original_cat = arr
        original_mode = arr.category_mode

        # Extract metadata for roundtrip preservation (store even for modes that need conversion)
        try:
            rt_meta = _extract_categorical_meta(original_cat)
        except Exception:
            rt_meta = None

        if arr.category_mode in (CategoryMode.Default, CategoryMode.StringArray, CategoryMode.NumericArray):
            pass  # already compatible with pandas; no special handling needed
        elif arr.category_mode in (CategoryMode.Dictionary, CategoryMode.IntEnum):
            # For pandas display, we want human-readable strings, but as_singlekey() is lossy/buggy
            # for dicts with non-contiguous codes (e.g., {0:USA,2:IRL,32:JPN}).
            # Instead, manually expand using the int->str mapping.
            old_category_mode = arr.category_mode
            try:
                # Use metadata to expand correctly
                enum_obj = original_cat.grouping._enum
                int_to_str = enum_obj._int_to_str_dict
                # rt_codes are the original integer codes (e.g., [0,2,32])
                # Expand to string array
                rt_codes = np.asarray(original_cat._fa) if hasattr(original_cat, "_fa") else np.asarray(original_cat)
                # Map each code to string, handling invalid (code not in dict -> nan)
                expanded = []
                for code in rt_codes:
                    # code may be numpy int
                    code_int = int(code) if not isinstance(code, (str, bytes)) else code
                    s = int_to_str.get(code_int, None)
                    if s is None:
                        expanded.append(np.nan)
                    else:
                        expanded.append(s)
                # Create a temporary categorical from expanded strings for pandas conversion
                # Use FastArray to create categorical, then use it for pandas codes/categories
                # We create a string array categorical: categories are unique strings
                expanded_arr = np.array(expanded, dtype=object)
                # Use pandas to create categorical directly from expanded strings for display
                # We'll bypass the arr variable and directly create pandas categorical later
                # To keep code path uniform, we set arr to a categorical made from expanded strings
                # But we need category_array and codes for pandas.
                # Simplest: create a new riptable Categorical from expanded strings
                # (this will be StringArray mode)
                temp_cat = TypeRegister.Categorical(expanded_arr)
                arr = temp_cat
                warnings.warn(
                    f"Series converted from {repr(CategoryMode(old_category_mode))} to {repr(CategoryMode(arr.category_mode))} for pandas display; "
                    f"original mode preserved in attrs for roundtrip via riptable.",
                    stacklevel=2,
                )
            except Exception as e:
                # Fallback to as_singlekey if manual expansion fails
                try:
                    old_mode = arr.category_mode
                    arr = arr.as_singlekey()
                    warnings.warn(
                        f"Series converted from {repr(CategoryMode(old_mode))} to {repr(CategoryMode(arr.category_mode))} for pandas display; "
                        f"original mode preserved in attrs for roundtrip via riptable. (fallback due to {e})",
                        stacklevel=2,
                    )
                except Exception:
                    pass
        elif arr.category_mode == CategoryMode.MultiKey:
            old_category_mode = arr.category_mode
            # For MultiKey, expand to tuples or joined strings for display
            # Use as_singlekey as fallback, but try to produce readable representation
            try:
                # For multikey, the expanded representation is tuples of values
                # We'll represent as strings joined by ' ' for pandas display
                # This is lossy for display but roundtrip is preserved via attrs
                # Use the original categorical's expand_array logic? For multikey, expand_array
                # is not straightforward. Use category_dict + codes to build tuples.
                uniquedict = original_cat.category_dict
                codes = np.asarray(original_cat._fa) if hasattr(original_cat, "_fa") else np.asarray(original_cat)
                base_idx = original_cat.base_index if original_cat.base_index is not None else 1
                # codes are 1-based indices into uniquedict arrays
                # Build expanded tuples
                expanded_tuples = []
                # uniquedict values are FastArrays
                uniq_arrays = [np.asarray(v) for v in uniquedict.values()]
                for code in codes:
                    idx = int(code) - base_idx
                    if idx < 0 or idx >= len(uniq_arrays[0]):
                        expanded_tuples.append(np.nan)
                    else:
                        tup = tuple(arr[idx] for arr in uniq_arrays)
                        # Join as string for pandas categorical (pandas can handle tuples as categories, but use string for simplicity)
                        # Use tuple itself as category (pandas supports object categories)
                        expanded_tuples.append(tup)
                # Create pandas categorical from tuples (object dtype)
                # We'll directly create pandas categorical later, so set arr to a dummy that will be overridden
                # Instead, we will handle pandas creation specially for multikey below
                # For now, set a flag and handle after
                # Use a placeholder: create categorical from string representations
                str_reprs = [str(t) if not (isinstance(t, float) and np.isnan(t)) else np.nan for t in expanded_tuples]
                temp_cat = TypeRegister.Categorical(str_reprs)
                arr = temp_cat
                warnings.warn(
                    f"Series converted from {repr(CategoryMode(old_category_mode))} to {repr(CategoryMode(arr.category_mode))} for pandas display; "
                    f"original mode preserved in attrs for roundtrip via riptable.",
                    stacklevel=2,
                )
            except Exception as e:
                try:
                    old_mode = arr.category_mode
                    arr = arr.as_singlekey()
                    warnings.warn(
                        f"Series converted from {repr(CategoryMode(old_mode))} to {repr(CategoryMode(arr.category_mode))} for pandas display; "
                        f"original mode preserved in attrs for roundtrip via riptable. (fallback due to {e})",
                        stacklevel=2,
                    )
                except Exception:
                    pass
        else:
            raise NotImplementedError(
                f"Dataset.to_pandas: Unhandled category mode {repr(CategoryMode(arr.category_mode))}"
            )

        base_index = 0 if arr.base_index is None else arr.base_index
        # pandas ordered flag: preserve from original categorical
        ordered = False
        try:
            # original_cat may have ordered flag
            if original_cat._ordered is not None:
                ordered = bool(original_cat._ordered)
            else:
                ordered = bool(original_cat.grouping.isordered)
        except Exception:
            ordered = False

        codes = np.asarray(arr) - base_index
        # Handle invalid codes: riptable uses 0 for invalid (when base_index=1), pandas uses -1
        # Our subtraction already converts 0 -> -base_index, but for base_index=1, 0-1 = -1 correct.
        # For base_index=0, invalid handling is different; but we keep as is for now.
        # Clip invalid handling: ensure -1 for invalid
        # For base_index=1, codes = [0,1,2] -1 => [-1,0,1] which matches pandas expectation (-1 invalid)
        categories = _to_unicode_if_string(arr.category_array) if unicode else arr.category_array
        # pandas 3: Categorical.from_codes supports ordered parameter
        try:
            pandas_cat = pd.Categorical.from_codes(codes, categories=categories, ordered=ordered)
        except TypeError:
            # Older pandas may not support ordered in from_codes, fallback
            pandas_cat = pd.Categorical.from_codes(codes, categories=categories)
            # Try to set ordered via constructor if possible
            if ordered:
                try:
                    pandas_cat = pd.Categorical(categories[codes], categories=categories, ordered=True)
                except Exception:
                    pass
        out = pd.Series(pandas_cat)

        # Store riptable metadata in Series attrs for direct Series roundtrip
        # (Note: Series attrs do not survive DataFrame construction, so Dataset.to_pandas
        # will also store metadata in DataFrame.attrs)
        if rt_meta is not None:
            try:
                out.attrs["_rt_cat_meta"] = rt_meta
            except Exception:
                # attrs may not be writable in some pandas versions, ignore
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
