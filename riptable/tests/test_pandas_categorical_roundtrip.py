import warnings
from enum import IntEnum

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

import riptable as rt
from riptable import Categorical, Dataset, FastArray
from riptable.Utils.pandas_utils import (
    _RIPTABLE_CATEGORICAL_ATTRS_KEY,
    fastarray_to_pandas_series,
    pandas_series_to_riptable,
)
from riptable.rt_enum import CategoryMode


def roundtrip(cat):
    ds = Dataset({"c": cat})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        pdf = ds.to_pandas()
    return Dataset.from_pandas(pdf)["c"], pdf


def assert_categorical_equal(orig, back):
    assert isinstance(back, Categorical)
    assert back.category_mode == orig.category_mode
    assert back.base_index == orig.base_index
    assert back.ordered == orig.ordered
    assert_array_equal(np.asarray(back), np.asarray(orig))
    if orig.ismultikey:
        assert list(back.category_dict.keys()) == list(orig.category_dict.keys())
        for key in orig.category_dict:
            assert_array_equal(back.category_dict[key], orig.category_dict[key])
    elif orig.isenum:
        assert back.category_mapping == orig.category_mapping
        assert back.isenum
    else:
        assert_array_equal(back.category_array, orig.category_array)
        assert back.category_array.dtype == orig.category_array.dtype


def test_string_base1_with_unused_category_and_invalid():
    orig = Categorical([0, 1, 2, 3, 0], ["a", "b", "c", "d"])
    back, pdf = roundtrip(orig)
    assert pdf["c"].dtype.name == "category"
    assert list(pdf["c"].cat.categories) == ["a", "b", "c", "d"]
    assert list(pdf["c"].cat.codes) == [-1, 0, 1, 2, -1]
    assert orig.ordered
    assert pdf["c"].cat.ordered
    assert_categorical_equal(orig, back)


def test_string_base0():
    orig = Categorical(["a", "b", "a"], categories=["a", "b"], base_index=0)
    back, pdf = roundtrip(orig)
    assert list(pdf["c"].cat.codes) == [0, 1, 0]
    assert_categorical_equal(orig, back)


def test_numeric_categories():
    orig_int = Categorical([4, 5, 4, 6])
    back_int, _ = roundtrip(orig_int)
    assert_categorical_equal(orig_int, back_int)

    orig_float = Categorical([1.5, 2.5, 1.5])
    back_float, _ = roundtrip(orig_float)
    assert_categorical_equal(orig_float, back_float)


def test_dictionary_sparse_codes_and_unmapped_code():
    mapping = {0: "USA", 2: "IRL", 4: "GBR", 8: "AUS", 16: "CHN", 32: "JPN"}
    orig = Categorical([0, 2, 32, 2, 99], categories=mapping)
    back, pdf = roundtrip(orig)
    assert list(pdf["c"].cat.categories) == ["USA", "IRL", "GBR", "AUS", "CHN", "JPN"]
    assert list(pdf["c"].cat.codes) == [0, 1, 5, 1, -1]
    assert_categorical_equal(orig, back)


def test_intenum_codes_preserved():
    class Level(IntEnum):
        DEBUG = 10
        INFO = 20
        WARNING = 30

    orig = Categorical([10, 20, 10, 0, 30], Level)
    back, _ = roundtrip(orig)
    assert_categorical_equal(orig, back)


def test_multikey_categories_preserved():
    orig = Categorical([FastArray(["AAPL", "MSFT", "AAPL", "MSFT"]), FastArray([1, 1, 2, 1])])
    back, pdf = roundtrip(orig)
    assert list(pdf["c"].cat.categories) == ["AAPL_1", "MSFT_1", "AAPL_2"]
    assert_categorical_equal(orig, back)


def test_ordered_flag_roundtrips():
    # Sorted categories with ordered=True: pandas and riptable must both report ordered.
    orig = Categorical(["a", "b", "a"], categories=["a", "b"], ordered=True)
    assert orig.ordered
    back, pdf = roundtrip(orig)
    assert pdf["c"].cat.ordered
    assert_categorical_equal(orig, back)

    # A pandas-native ordered categorical with unsorted categories keeps its order.
    pdf = pd.DataFrame({"c": pd.Categorical(["x", "y", None, "x"], categories=["y", "x", "z"], ordered=True)})
    ds = Dataset.from_pandas(pdf)
    cat = ds["c"]
    assert isinstance(cat, Categorical)
    assert cat.ordered
    assert_array_equal(np.asarray(cat), [2, 1, 0, 2])
    assert_array_equal(cat.category_array, np.array(["y", "x", "z"], dtype="S"))


def test_fallback_without_metadata_still_converts_values():
    orig = Categorical([0, 2, 32, 2], categories={0: "USA", 2: "IRL", 4: "GBR", 8: "AUS", 16: "CHN", 32: "JPN"})
    ds = Dataset({"c": orig})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        pdf = ds.to_pandas()
    assert _RIPTABLE_CATEGORICAL_ATTRS_KEY in pdf.attrs
    del pdf.attrs[_RIPTABLE_CATEGORICAL_ATTRS_KEY]
    back = Dataset.from_pandas(pdf)["c"]
    # Without metadata the mapping is lost, but the displayed values survive.
    assert back.category_mode == CategoryMode.StringArray
    assert_array_equal(back.expand_array, orig.as_singlekey().expand_array)


def test_fallback_when_pandas_categories_edited():
    orig = Categorical(["a", "b", "a"], categories=["a", "b"])
    ds = Dataset({"c": orig})
    pdf = ds.to_pandas()
    pdf["c"] = pdf["c"].cat.rename_categories(["A", "B"])
    back = Dataset.from_pandas(pdf)["c"]
    assert isinstance(back, Categorical)
    assert_array_equal(back.expand_array, np.array(["A", "B", "A"], dtype="S"))


def test_series_roundtrip_preserves_dictionary_mode():
    orig = Categorical([0, 2, 32], categories={0: "USA", 2: "IRL", 32: "JPN"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        series = fastarray_to_pandas_series(orig)
    assert _RIPTABLE_CATEGORICAL_ATTRS_KEY in series.attrs
    back = pandas_series_to_riptable(series)
    assert_categorical_equal(orig, back)


def test_dataset_constructor_roundtrip_preserves_dictionary_mode():
    orig = Categorical([0, 2, 32], categories={0: "USA", 2: "IRL", 32: "JPN"})
    ds = Dataset({"c": orig})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        pdf = ds.to_pandas()
    back = Dataset(pdf)["c"]
    assert_categorical_equal(orig, back)


def test_datetime_us_resolution_from_pandas():
    # Pandas 3 defaults to microsecond resolution; DateTimeNano needs nanoseconds.
    df = pd.DataFrame({"t": pd.Series(pd.date_range("20190101", periods=3, tz="US/Eastern"))})
    assert str(df["t"].dtype).startswith("datetime64[us")
    ds = Dataset.from_pandas(df)
    assert isinstance(ds["t"], rt.DateTimeNano)
    assert list(ds["t"].yyyymmdd) == [20190101, 20190102, 20190103]
    assert ds["t"]._timezone._to_tz == "America/New_York"


def test_timedelta_us_resolution_from_pandas():
    df = pd.DataFrame({"t": pd.Series(pd.to_timedelta(["1 days", "2 days"]))})
    ds = Dataset.from_pandas(df)
    assert isinstance(ds["t"], rt.TimeSpan)
    assert_array_equal(
        np.asarray(ds["t"], dtype=np.int64),
        [86400_000_000_000, 172800_000_000_000],
    )
