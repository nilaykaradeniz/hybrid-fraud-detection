"""Test that the features do not look into the future, on the real data."""

import pandas as pd

from src.data_loader import DataLoader, load_config
from src.features import FeatureBuilder


def test_features_no_future():
    """Days 1 to 20 get the same features with and without the later days in the table."""
    config = load_config()
    loader = DataLoader(config)
    df = loader.load()
    train_df, _ = loader.split(df)
    builder = FeatureBuilder(config).fit(train_df)
    first_days = df[df["day"] <= 20]

    built_on_all_days = builder.transform(df).loc[first_days.index]
    built_on_first_days = builder.transform(first_days)

    pd.testing.assert_frame_equal(built_on_all_days, built_on_first_days)
