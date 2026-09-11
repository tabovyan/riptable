"""Validation script for categorical roundtrip preservation (riptable -> pandas -> riptable)"""
import pandas as pd
print(f"pandas {pd.__version__}")
import riptable as rt
import numpy as np
from riptable import Categorical, Dataset, FastArray

def test_case(name, cat):
    print(f"\n=== {name} ===")
    print(f"orig mode {cat.category_mode} base_index {cat.base_index} ordered {cat.ordered}")
    ds = Dataset({'Col': cat})
    df = ds.to_pandas()
    print(f"df dtype {df['Col'].dtype} attrs {bool(df.attrs.get('_rt_categorical_meta'))}")
    ds2 = Dataset.from_pandas(df)
    c2 = ds2['Col']
    mode_match = c2.category_mode == cat.category_mode
    fa_match = np.array_equal(np.asarray(cat._fa) if hasattr(cat, '_fa') else np.asarray(cat),
                              np.asarray(c2._fa) if hasattr(c2, '_fa') else np.asarray(c2))
    expand_match = np.array_equal(cat.as_singlekey().expand_array, c2.as_singlekey().expand_array)
    print(f"roundtrip mode {c2.category_mode} mode_match={mode_match} fa_match={fa_match} expand_match={expand_match}")
    assert mode_match, f"Mode mismatch: {cat.category_mode} vs {c2.category_mode}"
    assert expand_match, "Expand array mismatch"
    print("PASS")
    return True

# 1. StringArray
test_case("StringArray", Categorical(['a','b','c','a','b']))

# 2. Dictionary (sparse int mapping)
country_map = {0: "USA", 2: "IRL", 4: "GBR", 8: "AUS", 16: "CHN", 32: "JPN"}
test_case("Dictionary", Categorical([0, 2, 32, 0], categories=country_map))

# 3. IntEnum
from enum import IntEnum
class MyEnum(IntEnum):
    a = 0; b = 1; c = 2
test_case("IntEnum", Categorical([0,1,2,0], MyEnum))

# 4. MultiKey
test_case("MultiKey", Categorical([FastArray([0,1,2,0]), FastArray([10,20,30,10])]))

# 5. NumericArray
test_case("NumericArray", Categorical([1.0, 2.0, 3.0, 1.0], [1.0, 2.0, 3.0]))

# 6. Ordered
c_ord = Categorical(['a','b','c'], ordered=True)
test_case("Ordered", c_ord)
# also check ordered flag preserved
ds = Dataset({'Ord': c_ord})
df = ds.to_pandas()
assert df['Ord'].cat.ordered == True, "pandas ordered flag not preserved"
ds2 = Dataset.from_pandas(df)
assert ds2['Ord'].ordered == True, "riptable ordered flag not preserved after roundtrip"
print("\nOrdered flag preserved correctly")

print("\nAll 6 test cases PASSED")
