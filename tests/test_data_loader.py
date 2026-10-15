"""Small tests for the DataLoader and the DataQualityChecker, on a tiny hand-made table."""

import pandas as pd
import pytest

from src.data_loader import DataLoader, load_config
from src.data_quality import DataQualityChecker


def make_toy_attempts() -> pd.DataFrame:
    """Build three clean payment attempts with all 17 columns."""
    return pd.DataFrame({
        "transaction_id": [1, 2, 3],
        "creation_date": ["2022-01-01", "2022-01-01", "2022-01-02"],
        "ip_id": ["ip1", "ip2", "ip3"],
        "credit_card_id": ["card1", "card2", "card3"],
        "booker_country": ["Germany", "Spain", "France"],
        "card_issuing_country": ["Germany", "Spain", "France"],
        "hotel_country": ["Germany", "Spain", "France"],
        "hotel_id": [10, 11, 12],
        "language": ["German", "Spanish", "French"],
        "price_euro": [500.0, 800.0, 1200.0],
        "book_window": [10, 20, 30],
        "length_of_stay": [2, 3, 4],
        "email_id": ["mail1", "mail2", "mail3"],
        "email_domain": ["@a.com", "@b.com", "@c.com"],
        "auth_result": ["AUTHORISED", "REFUSED", "AUTHORISED"],
        "property_age": [12, 24, 36],
        "payment_method": ["visa", "visa", "mastercard"],
    })


def test_data_checks_toy():
    """A missing column stops the run, and the quality report finds odd rows without changing them."""
    config = load_config()
    loader = DataLoader(config)
    checker = DataQualityChecker(config)

    # A missing column is a blocking error with a clear message.
    without_price = make_toy_attempts().drop(columns="price_euro")
    with pytest.raises(ValueError, match="price_euro"):
        loader.check_structure(without_price)

    # One zero-night stay and one negative book window.
    attempts = make_toy_attempts()
    attempts.loc[0, "length_of_stay"] = 0
    attempts.loc[1, "book_window"] = -5
    loader.check_structure(attempts)
    attempts = loader.add_day_column(attempts)
    before = attempts.copy()

    report = checker.quality_report(attempts).set_index("id")

    assert report.loc["Q5", "rows"] == 1
    assert report.loc["Q5", "status"] == "WARNING"
    assert report.loc["Q4", "rows"] == 1
    assert report.loc["Q4", "status"] == "WARNING"
    pd.testing.assert_frame_equal(attempts, before)
