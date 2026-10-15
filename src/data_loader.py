"""Load the payment attempts and stop the run if the data is broken.

This is the first step of the pipeline. Before we build anything on the
data, we must know that the file is complete. The data is synthetic
transaction data from a fraud analytics case study.

The other two data classes build on the loaded data:
DataQualityChecker (data_quality.py) and DataProfiler (data_profile.py).
"""

import pandas as pd
import yaml

# The meaning of each column, taken from the column dictionary in the brief.
COLUMN_MEANINGS = {
    "transaction_id": "Unique ID of the payment attempt",
    "creation_date": "Day the reservation was created (no time of day)",
    "ip_id": "ID of the IP address the booker used",
    "credit_card_id": "ID of the card used for the payment",
    "booker_country": "Country the reservation was made from",
    "card_issuing_country": "Country where the card was issued",
    "hotel_country": "Country of the property",
    "hotel_id": "ID of the property listed on the platform",
    "language": "Language of the website the booker used",
    "price_euro": "Total value of the reservation in EUR",
    "book_window": "Days between the reservation date and the check-in date",
    "length_of_stay": "Nights between check-in and check-out",
    "email_id": "ID of the booker's email address",
    "email_domain": "Domain of the booker's email address",
    "auth_result": "Answer of the payment provider to this attempt",
    "property_age": "Months since the property was listed, at the time of the reservation",
    "payment_method": "Payment method used",
}

EXPECTED_COLUMNS = list(COLUMN_MEANINGS)

# The ID columns, with the plural word we use for them in tables and reports.
ID_NAMES = {"credit_card_id": "cards", "email_id": "emails", "ip_id": "IPs", "hotel_id": "properties"}


def load_config(path: str = "config.yaml") -> dict:
    """Read the project settings.

    Every threshold and setting lives in one file, so a change needs no
    code edit. The pipeline, the notebook and the tests all read it here.

    Args:
        path: The settings file.

    Returns:
        The settings, as a dictionary.
    """
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def text_columns(df: pd.DataFrame) -> list[str]:
    """Name the original columns that hold text, such as a country or an ID."""
    return [col for col in EXPECTED_COLUMNS if pd.api.types.is_string_dtype(df[col])]


class DataLoader:
    """Read the raw data, run the blocking checks and split it by time.

    The DataLoader never changes or removes a row. It only adds a "day"
    column, so that later steps can count attempts per day.

    Args:
        config: The settings from config.yaml, as a dictionary.
    """

    def __init__(self, config: dict):
        """Keep the settings that this class needs."""
        self.path = config["data"]["path"]
        self.quality = config["quality"]
        self.train_last_day = config["split"]["train_last_day"]
        self.passed_checks: list[str] = []

    # ------------------------------------------------------------------
    # Loading and blocking checks
    # ------------------------------------------------------------------

    def load(self) -> pd.DataFrame:
        """Read the data file, run the blocking checks and add a day column.

        The sentences of the checks that passed are kept in passed_checks,
        so the notebook can show them without running the checks again.

        Returns:
            One row per payment attempt, with creation_date as a date and the
            new "day" column.

        Raises:
            ValueError: If one of the blocking checks fails.
        """
        df = pd.read_csv(self.path)
        self.passed_checks = self.check_structure(df)
        return self.add_day_column(df)

    @staticmethod
    def add_day_column(df: pd.DataFrame) -> pd.DataFrame:
        """Turn creation_date into a date and add the day of the month.

        Args:
            df: The raw data, with creation_date as text.

        Returns:
            The same table, with creation_date as a date and a new "day"
            column (1 to 31).
        """
        df["creation_date"] = pd.to_datetime(df["creation_date"])
        df["day"] = df["creation_date"].dt.day
        return df

    def check_structure(self, df: pd.DataFrame) -> list[str]:
        """Run the five blocking checks. Stop the run if one of them fails.

        These checks catch problems that would make every later step wrong,
        such as a missing column or an unknown payment result. Each check is
        a small method that gives back one sentence when it passes.

        Args:
            df: The raw data, as read from the file.

        Returns:
            One sentence per check that passed, to show in the notebook.

        Raises:
            ValueError: With a clear message that names the failed check.
        """
        # The column check runs first, because the other checks need the columns.
        return [
            self._check_columns(df),
            self._check_missing_values(df),
            self._check_unique_ids(df),
            self._check_types(df),
            self._check_auth_results(df),
        ]

    @staticmethod
    def _check_columns(df: pd.DataFrame) -> str:
        """Check that every expected column is in the file."""
        missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Check 'all columns' failed: missing columns {missing_columns}")
        return f"All {len(EXPECTED_COLUMNS)} columns exist."

    @staticmethod
    def _check_missing_values(df: pd.DataFrame) -> str:
        """Check that no cell is missing, and that no text cell holds nothing or only spaces."""
        missing_values = int(df[EXPECTED_COLUMNS].isna().sum().sum())
        empty_texts = 0
        for col in text_columns(df):
            empty_texts += int((df[col].str.strip() == "").sum())
        if missing_values > 0 or empty_texts > 0:
            raise ValueError(
                f"Check 'no missing values' failed: {missing_values} missing values "
                f"and {empty_texts} empty text values"
            )
        return "There are no missing values and no empty text values."

    @staticmethod
    def _check_unique_ids(df: pd.DataFrame) -> str:
        """Check that no transaction_id is used twice."""
        duplicated_ids = int(df["transaction_id"].duplicated().sum())
        if duplicated_ids > 0:
            raise ValueError(f"Check 'unique transaction_id' failed: {duplicated_ids} repeated IDs")
        return "Every transaction_id is unique."

    def _check_types(self, df: pd.DataFrame) -> str:
        """Check that dates are dates and that number columns hold numbers."""
        dates = pd.to_datetime(df["creation_date"], errors="coerce")
        if dates.isna().any():
            raise ValueError("Check 'types' failed: creation_date has values that are not dates")

        prices = pd.to_numeric(df["price_euro"], errors="coerce")
        if prices.isna().any():
            raise ValueError("Check 'types' failed: price_euro has values that are not numbers")

        whole_number_columns = self.quality["whole_number_columns"]
        for col in whole_number_columns:
            values = pd.to_numeric(df[col], errors="coerce")
            if values.isna().any() or (values % 1 != 0).any():
                raise ValueError(f"Check 'types' failed: {col} has values that are not whole numbers")
        return (
            "creation_date is a date, price_euro is numeric, and "
            + ", ".join(whole_number_columns)
            + " are whole numbers."
        )

    def _check_auth_results(self, df: pd.DataFrame) -> str:
        """Check that auth_result holds only the answers we know."""
        known_results = self.quality["auth_results"]
        unknown_results = sorted(set(df["auth_result"]) - set(known_results))
        if unknown_results:
            raise ValueError(f"Check 'known auth_result values' failed: unknown values {unknown_results}")
        return f"auth_result has only the {len(known_results)} known values."

    # ------------------------------------------------------------------
    # Train and test split
    # ------------------------------------------------------------------

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split the attempts by time into training days and test days.

        We learn from the past and test on the future, as in real life.
        Days 1 to train_last_day (14 in the config) are the training days,
        the rest are the test days.

        Args:
            df: The loaded data, with its "day" column.

        Returns:
            Two tables: the training rows and the test rows.
        """
        is_train = df["day"] <= self.train_last_day
        return df[is_train], df[~is_train]
