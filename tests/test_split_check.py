"""Small test for the SplitChecker, on two tiny hand-made tables."""

import pandas as pd

from src.data_loader import load_config
from src.split_check import SplitChecker


def test_split_check_toy():
    """The same data gives a PSI of 0, and a clear change in one column gives a large shift."""
    checker = SplitChecker(load_config())
    train = pd.DataFrame({
        "price_euro": [float(price) for price in range(100, 200)],
        "payment_method": ["visa"] * 50 + ["mastercard"] * 50,
    })

    same = checker.psi_table(train, train.copy(), ["price_euro", "payment_method"])
    assert (same["psi"] == 0).all()
    assert (same["status"] == "stable").all()

    # In the test part, every price is higher than any training price.
    test = train.copy()
    test["price_euro"] = test["price_euro"] + 1000
    shifted = checker.psi_table(train, test, ["price_euro", "payment_method"]).set_index("column")
    assert shifted.loc["price_euro", "status"] == "large shift"
    assert shifted.loc["payment_method", "status"] == "stable"
