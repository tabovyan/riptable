"""Round-trip preservation of riptable Categoricals through pandas.

riptable has several ``CategoryMode``s (StringArray, NumericArray, Dictionary,
IntEnum, MultiKey) while pandas only models a flat category list. These tests
check that ``Dataset.to_pandas`` -> ``Dataset.from_pandas`` restores the
original categorical (mode, categories/mapping, codes, base index, ordering)
using the metadata stored in ``DataFrame.attrs``, and that a plain pandas
categorical still converts sensibly when no metadata is present.
"""

import unittest
import warnings
from enum import IntEnum

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from riptable import Categorical, Dataset, FastArray
from riptable.rt_enum import CategoryMode


def _roundtrip(ds, key):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = ds.to_pandas()
    return Dataset.from_pandas(df)[key]


class TestPandasCategoricalRoundtrip(unittest.TestCase):
    def test_string_array_roundtrip(self):
        ds = Dataset({"x": Categorical(["b", "a", "b", "c", "a"])})
        result = _roundtrip(ds, "x")
        self.assertIsInstance(result, Categorical)
        self.assertEqual(result.category_mode, CategoryMode.StringArray)
        assert_array_equal(result._fa, ds.x._fa)
        assert_array_equal(result.category_array, ds.x.category_array)
        assert_array_equal(result.expand_array, ds.x.expand_array)

    def test_base_zero_roundtrip(self):
        ds = Dataset({"x": Categorical(["b", "a", "b", "c"], base_index=0)})
        result = _roundtrip(ds, "x")
        self.assertEqual(result.base_index, 0)
        assert_array_equal(result._fa, ds.x._fa)
        assert_array_equal(result.expand_array, ds.x.expand_array)

    def test_numeric_array_roundtrip(self):
        ds = Dataset({"x": Categorical([4, 5, 4, 6, 5])})
        result = _roundtrip(ds, "x")
        self.assertEqual(result.category_mode, CategoryMode.NumericArray)
        assert_array_equal(result._fa, ds.x._fa)
        assert_array_equal(result.category_array, ds.x.category_array)

    def test_dictionary_roundtrip_preserves_mapping_and_codes(self):
        country_map = {0: "USA", 2: "IRL", 4: "GBR", 32: "JPN"}
        ds = Dataset({"x": Categorical([0, 2, 32, 2, 0], categories=country_map)})
        self.assertIn(ds.x.category_mode, (CategoryMode.Dictionary, CategoryMode.IntEnum))
        result = _roundtrip(ds, "x")
        self.assertEqual(result.category_mode, ds.x.category_mode)
        self.assertEqual(result.category_mapping, country_map)
        assert_array_equal(result._fa, ds.x._fa)

    def test_intenum_roundtrip(self):
        class Color(IntEnum):
            RED = 10
            GREEN = 20
            BLUE = 30

        ds = Dataset({"x": Categorical([10, 20, 10, 30], categories=Color)})
        result = _roundtrip(ds, "x")
        self.assertTrue(result.isenum)
        assert_array_equal(result._fa, ds.x._fa)
        self.assertEqual(result.category_mapping, ds.x.category_mapping)

    def test_multikey_roundtrip(self):
        ds = Dataset(
            {"x": Categorical([FastArray(["a", "b", "a", "b"]), FastArray([1, 2, 1, 1])])}
        )
        self.assertEqual(ds.x.category_mode, CategoryMode.MultiKey)
        result = _roundtrip(ds, "x")
        self.assertEqual(result.category_mode, CategoryMode.MultiKey)
        for expected_arr, result_arr in zip(ds.x.expand_array, result.expand_array, strict=True):
            assert_array_equal(result_arr, expected_arr)

    def test_filtered_values_roundtrip(self):
        filt = FastArray([True, False, True, True])
        ds = Dataset({"x": Categorical(FastArray(["a", "b", "c", "b"]), filter=filt)})
        result = _roundtrip(ds, "x")
        assert_array_equal(result._fa, ds.x._fa)
        assert_array_equal(result.isfiltered(), ds.x.isfiltered())

    def test_float_nan_category_roundtrip(self):
        ds = Dataset({"x": Categorical(FastArray([1.0, np.nan, 2.0, 1.0]))})
        result = _roundtrip(ds, "x")
        self.assertEqual(result.category_mode, CategoryMode.NumericArray)
        assert_array_equal(result.expand_array, ds.x.expand_array)

    def test_plain_pandas_categorical_keeps_ordering(self):
        df = pd.DataFrame({"x": pd.Categorical(["b", "a", "b"], categories=["b", "a"], ordered=True)})
        ds = Dataset.from_pandas(df)
        self.assertIsInstance(ds.x, Categorical)
        self.assertTrue(ds.x.ordered)
        assert_array_equal(ds.x.category_array.astype("U"), np.array(["b", "a"]))

    def test_roundtrip_without_metadata_falls_back_to_values(self):
        ds = Dataset({"x": Categorical(["b", "a", "b", "c"])})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = ds.to_pandas()
        df.attrs.clear()
        result = Dataset.from_pandas(df)["x"]
        self.assertIsInstance(result, Categorical)
        assert_array_equal(result.expand_array, ds.x.expand_array)


if __name__ == "__main__":
    tester = unittest.main()
