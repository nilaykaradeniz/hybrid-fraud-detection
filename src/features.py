"""Build the features: numbers that describe each attempt and its history.

A feature is one column that the model or a rule can use, for example
"how many times was this card used today". Most features count earlier
attempts with the same card, email, IP or property in a time window.

A feature of an attempt on day d only uses rows from day d or earlier.
It never counts over the whole month, because that would see the future.
The auth_result of the attempt itself is never a feature: the decline
counts only look at the days before.

Known limits: the data has no time of day, so a same-day count also
includes attempts that came later on that day. And at the start of the
month the windows are shorter, because there are no days before day 1.
"""

import numpy as np
import pandas as pd

# For an attempt on day d, a window runs from day "d - start" to day "d - end".
WINDOWS = {
    "same day": (0, 0),
    "last 7 days": (6, 0),
    "7 days before": (7, 1),
    "previous day": (1, 1),
}

# The short name of each ID in the feature names.
ID_PREFIXES = {"card": "credit_card_id", "email": "email_id", "ip": "ip_id", "property": "hotel_id"}

# Every column that FeatureBuilder builds or passes on: name, meaning, time window and who uses it.
FEATURES = [
    ("card_attempts_day", "Attempts with the same card", "same day", "model, rules"),
    ("email_attempts_day", "Attempts with the same email", "same day", "model"),
    ("ip_attempts_day", "Attempts from the same IP", "same day", "model"),
    ("property_attempts_day", "Attempts at the same property", "same day", "model, rules"),
    ("card_attempts_7d", "Attempts with the same card", "last 7 days", "model"),
    ("email_attempts_7d", "Attempts with the same email", "last 7 days", "model"),
    ("ip_attempts_7d", "Attempts from the same IP", "last 7 days", "model"),
    ("property_attempts_7d", "Attempts at the same property", "last 7 days", "model"),
    ("ip_distinct_cards_day", "Different cards used from the same IP", "same day", "model, rules"),
    ("card_distinct_emails_7d", "Different emails used with the same card", "last 7 days", "model, rules"),
    ("email_distinct_cards_7d", "Different cards used with the same email", "last 7 days", "model, rules"),
    ("card_declines_7d_before", "Declined attempts with the same card", "7 days before", "model"),
    ("ip_declines_7d_before", "Declined attempts from the same IP", "7 days before", "model"),
    ("property_declines_7d_before", "Declined attempts at the same property", "7 days before", "model, rules"),
    ("property_age", "Months since the property was listed", "this attempt", "model, rules"),
    ("price_euro", "Value of the reservation in EUR", "this attempt", "model"),
    ("book_window", "Days between the reservation and the check-in", "this attempt", "model, rules"),
    ("length_of_stay", "Nights of the stay", "this attempt", "model, rules"),
    ("is_abroad", "1 if booker_country is not hotel_country", "this attempt", "model"),
    ("ip_distinct_emails_day", "Different emails used from the same IP", "same day", "rules"),
    ("card_hard_declines_prev_day", "Hard declines (REFUSED or FRAUD) with the same card", "previous day", "rules"),
    ("ip_hard_declines_prev_day", "Hard declines (REFUSED or FRAUD) from the same IP", "previous day", "rules"),
    ("card_property_attempts_7d", "Attempts with the same card at the same property", "last 7 days", "rules"),
    ("email_property_attempts_7d", "Attempts with the same email at the same property", "last 7 days", "rules"),
    ("card_country_mismatch", "booker_country is not card_issuing_country", "this attempt", "rules"),
    ("language_mismatch", "The country of the site language is not booker_country", "this attempt", "rules"),
    ("high_price", "price_euro is at or above the high price limit of the training days", "this attempt", "rules"),
    ("value_band", "low, medium or high, cut at two price points of the training days", "this attempt",
     "decision policy"),
]

# The 19 columns that go into the Isolation Forest. card_country_mismatch and language_mismatch
# are left out, so that the model stays independent of the soft rules R6 and R7.
MODEL_FEATURES = [name for name, _, _, used_by in FEATURES if "model" in used_by]


class FeatureBuilder:
    """Learn the price limits on the training days and build all feature columns.

    Args:
        config: The settings from config.yaml, as a dictionary.
    """

    def __init__(self, config: dict):
        """Keep the settings that this class needs."""
        settings = config["features"]
        self.declined_results = settings["declined_results"]
        self.hard_declined_results = settings["hard_declined_results"]
        self.value_band_percentiles = settings["value_band_percentiles"]
        self.high_price_percentile = settings["high_price_percentile"]
        self.language_country = settings["language_country"]
        self.value_band_cuts = None
        self.high_price_limit = None

    def fit(self, train_df: pd.DataFrame) -> "FeatureBuilder":
        """Learn the price limits from the training days.

        A percentile is the value below which a given share of the data
        falls. The two value band cut points are the 33rd and 66th
        percentile of price_euro, and the high price limit is the 90th
        (all three numbers are in the config).

        Args:
            train_df: The attempts of the training days.

        Returns:
            The builder itself, so that fit(train_df).transform(df) works.
        """
        prices = train_df["price_euro"]
        self.value_band_cuts = [float(cut) for cut in np.percentile(prices, self.value_band_percentiles)]
        self.high_price_limit = float(np.percentile(prices, self.high_price_percentile))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build all feature columns.

        Call this once on all days and split afterwards. If the training
        and the test days are transformed separately, the first test days
        lose their history from the last training days, and the counts are
        wrong.

        Args:
            df: The attempts of all days, one row per payment attempt.

        Returns:
            A copy of df with the new columns of feature_table().

        Raises:
            ValueError: If fit() was not called first.
        """
        if self.value_band_cuts is None:
            raise ValueError("Call fit() on the training days before transform().")
        features = df.copy()
        self._add_attempt_counts(features)
        self._add_distinct_counts(features)
        self._add_decline_counts(features)
        self._add_mismatches(features)
        self._add_price_columns(features)
        return features

    @staticmethod
    def feature_table() -> pd.DataFrame:
        """List every feature column with its meaning, its time window and who uses it."""
        return pd.DataFrame(FEATURES, columns=["column", "meaning", "time window", "used by"])

    def _add_attempt_counts(self, features: pd.DataFrame) -> None:
        """Add how often the same card, email, IP or property was used. Many uses in a short time can be an attack."""
        for prefix, id_col in ID_PREFIXES.items():
            features[f"{prefix}_attempts_day"] = self._count(features, [id_col], "same day")
            features[f"{prefix}_attempts_7d"] = self._count(features, [id_col], "last 7 days")
        for prefix in ["card", "email"]:
            keys = [ID_PREFIXES[prefix], "hotel_id"]
            features[f"{prefix}_property_attempts_7d"] = self._count(features, keys, "last 7 days")

    def _add_distinct_counts(self, features: pd.DataFrame) -> None:
        """Add how many different cards or emails come with one ID. Many of them point to stolen identities."""
        features["ip_distinct_cards_day"] = self._count_distinct(features, "ip_id", "credit_card_id", "same day")
        features["ip_distinct_emails_day"] = self._count_distinct(features, "ip_id", "email_id", "same day")
        features["card_distinct_emails_7d"] = self._count_distinct(
            features, "credit_card_id", "email_id", "last 7 days")
        features["email_distinct_cards_7d"] = self._count_distinct(
            features, "email_id", "credit_card_id", "last 7 days")

    def _add_decline_counts(self, features: pd.DataFrame) -> None:
        """Add how many earlier attempts with the same ID were declined.

        "Declined" is REFUSED, FRAUD or ERROR, and "hard declined" is
        REFUSED or FRAUD (both lists are in the config).
        """
        declined = features["auth_result"].isin(self.declined_results)
        hard_declined = features["auth_result"].isin(self.hard_declined_results)
        for prefix in ["card", "ip", "property"]:
            features[f"{prefix}_declines_7d_before"] = self._count(
                features, [ID_PREFIXES[prefix]], "7 days before", only=declined)
        for prefix in ["card", "ip"]:
            features[f"{prefix}_hard_declines_prev_day"] = self._count(
                features, [ID_PREFIXES[prefix]], "previous day", only=hard_declined)

    def _add_mismatches(self, features: pd.DataFrame) -> None:
        """Add the columns that compare two countries of the same attempt."""
        booker_country = features["booker_country"]
        features["is_abroad"] = (booker_country != features["hotel_country"]).astype(int)
        features["card_country_mismatch"] = booker_country != features["card_issuing_country"]
        features["language_mismatch"] = features["language"].map(self.language_country) != booker_country

    def _add_price_columns(self, features: pd.DataFrame) -> None:
        """Add high_price and value_band, from the limits that fit() learned. A price on a limit goes up."""
        prices = features["price_euro"]
        low_cut, high_cut = self.value_band_cuts
        features["high_price"] = prices >= self.high_price_limit
        features["value_band"] = "medium"
        features.loc[prices < low_cut, "value_band"] = "low"
        features.loc[prices >= high_cut, "value_band"] = "high"

    def _count(self, df: pd.DataFrame, keys: list[str], window: str, only: pd.Series = None) -> np.ndarray:
        """Count the attempts with the same keys in the window of each attempt.

        Args:
            df: The attempts of all days.
            keys: The columns that must be the same, for example ["credit_card_id"].
            window: A name from WINDOWS, for example "last 7 days".
            only: True for the rows to count (for example the declined ones). All rows if empty.

        Returns:
            One number per attempt, in the same order as df.
        """
        rows = df if only is None else df[only]
        spread = self._spread_over_days(rows[keys + ["day"]], window)
        counts = spread.groupby(keys + ["day"]).size()
        return self._look_up(df, keys, counts)

    def _count_distinct(self, df: pd.DataFrame, key: str, other: str, window: str) -> np.ndarray:
        """Count the different values of "other" that come with the same key in the window of each attempt."""
        pairs = df[[key, other, "day"]].drop_duplicates()
        counts = self._spread_over_days(pairs, window).groupby([key, "day"])[other].nunique()
        return self._look_up(df, [key], counts)

    @staticmethod
    def _spread_over_days(rows: pd.DataFrame, window: str) -> pd.DataFrame:
        """Copy each row to every day whose window holds it.

        With "last 7 days", a row of day 3 is copied to days 3 to 9. A count
        per day on the copies is then the count over the window. Rows are
        only copied forward, so no day gets a row from its future.
        """
        start, end = WINDOWS[window]
        return pd.concat([rows.assign(day=rows["day"] + days_later) for days_later in range(end, start + 1)])

    @staticmethod
    def _look_up(df: pd.DataFrame, keys: list[str], counts: pd.Series) -> np.ndarray:
        """Give each attempt the count of its own keys and day, or 0 if there is none."""
        found = df[keys + ["day"]].merge(counts.rename("count").reset_index(), on=keys + ["day"], how="left")
        return found["count"].fillna(0).astype(int).to_numpy()
