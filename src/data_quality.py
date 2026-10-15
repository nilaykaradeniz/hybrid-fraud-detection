"""Check the quality of the loaded data and report what is odd.

The blocking checks in DataLoader stop the run when the file is broken.
The checks here are softer: they never stop the run and never change the
data. They tell the reader what looks odd, so that we can explain it.
"""

import pandas as pd

from src.data_loader import EXPECTED_COLUMNS, ID_NAMES, text_columns

COUNTRY_COLUMNS = ["booker_country", "card_issuing_country", "hotel_country"]
# The IDs that belong to the booker.
BOOKER_ID_COLUMNS = ["ip_id", "credit_card_id", "email_id"]
REPORT_COLUMNS = ["check", "what we check", "result", "rows", "share of rows (%)", "status", "action"]


class DataQualityChecker:
    """Run the ten data quality checks Q1 to Q10.

    Each check is a small method that returns one row of the report.
    In this data no row needs a fix, and removing odd rows could also
    remove fraud signals.

    Args:
        config: The settings from config.yaml, as a dictionary.
    """

    def __init__(self, config: dict):
        """Keep the settings that this class needs."""
        self.quality = config["quality"]
        self.language_country = config["features"]["language_country"]

    def quality_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the ten data quality checks and return one row per check.

        The status of a check is:
        - OK: the check found nothing,
        - NOTE: not an error, but the reader needs to know it,
        - WARNING: it looks like a data error. We keep the rows and explain them.

        Args:
            df: The loaded data (output of DataLoader.load()).

        Returns:
            A table with the columns: check, what we check, result, rows,
            share of rows (%), status and action.
        """
        checks = [
            self._check_dates(df),
            self._check_time_of_day(df),
            self._check_repeated_attempts(df),
            self._check_value_ranges(df),
            self._check_zero_night_stays(df),
            self._check_text_values(df),
            self._check_known_values(df),
            self._check_one_value_per_id(df),
            self._check_card_id_for_every_method(df),
            self._check_ids_from_many_countries(df),
        ]
        report = pd.DataFrame(checks)
        report.insert(0, "id", [f"Q{number}" for number in range(1, len(checks) + 1)])
        report["share of rows (%)"] = (100 * report["rows"] / len(df)).round(1)
        return report[["id"] + REPORT_COLUMNS]

    @staticmethod
    def ids_from_many_countries(df: pd.DataFrame) -> pd.DataFrame:
        """List the IPs, cards and emails that come with more than one booker_country.

        A normal booker has one country, so an ID with many countries is a
        trace of an attack. Check Q10 counts these IDs, and the notebook
        shows the full list.

        Args:
            df: The loaded data.

        Returns:
            One row per ID, with its type (ip_id, credit_card_id or email_id),
            the ID, the number of booker countries and the number of attempts.
        """
        tables = []
        for id_col in BOOKER_ID_COLUMNS:
            per_id = df.groupby(id_col).agg(
                booker_countries=("booker_country", "nunique"),
                attempts=("transaction_id", "count"),
            )
            per_id = per_id[per_id["booker_countries"] > 1].sort_values("attempts", ascending=False)
            per_id = per_id.reset_index().rename(columns={id_col: "id"})
            per_id.insert(0, "id type", id_col)
            tables.append(per_id)
        return pd.concat(tables, ignore_index=True)

    @staticmethod
    def _report_row(check: str, what: str, result: str, rows: int, found: bool,
                    status_if_found: str, action_if_found: str, action_if_ok: str = "None") -> dict:
        """Put the parts of one check into one row of the quality report.

        If the check found nothing, the status is OK and the action is
        action_if_ok. Otherwise the status and the action are the "if found" ones.
        """
        return {
            "check": check,
            "what we check": what,
            "result": result,
            "rows": int(rows),
            "status": status_if_found if found else "OK",
            "action": action_if_found if found else action_if_ok,
        }

    def _check_dates(self, df: pd.DataFrame) -> dict:
        """Q1: all dates are in one month, and every day of that month has attempts."""
        dates = df["creation_date"]
        months = dates.dt.to_period("M")
        main_month = months.mode()[0]
        rows_outside = int((months != main_month).sum())
        attempts_per_day = df.groupby(dates.dt.date).size()
        missing_days = main_month.days_in_month - df.loc[months == main_month, "day"].nunique()

        result = (
            f"{dates.min().day} {dates.min():%B %Y} to {dates.max().day} {dates.max():%B %Y}, "
            f"{attempts_per_day.size} days, "
            f"{attempts_per_day.min():,} to {attempts_per_day.max():,} attempts per day"
        )
        if missing_days > 0:
            result += f"; {missing_days} days without attempts"
        return self._report_row(
            "Dates", "All dates are in one month, and every day has attempts", result, rows_outside,
            found=rows_outside > 0 or missing_days > 0, status_if_found="WARNING",
            action_if_found="Keep. Read the daily counts and the day split with care",
        )

    def _check_time_of_day(self, df: pd.DataFrame) -> dict:
        """Q2: does creation_date also have a time of day?

        Known limit: a time of exactly midnight looks the same as "no time".
        """
        dates = df["creation_date"]
        has_time = bool((dates != dates.dt.normalize()).any())
        return self._report_row(
            "Time of day", "creation_date has a time",
            "Dates have a time" if has_time else "Dates only, no time",
            0 if has_time else len(df), found=not has_time, status_if_found="NOTE",
            action_if_found="None. Same-day counts also include later attempts of that day",
        )

    def _check_repeated_attempts(self, df: pd.DataFrame) -> dict:
        """Q3: rows that are the same in every column except transaction_id."""
        columns = [col for col in EXPECTED_COLUMNS if col != "transaction_id"]
        repeated = int(df.duplicated(subset=columns, keep=False).sum())
        return self._report_row(
            "Repeated attempts", "Rows that are the same in every column except transaction_id",
            f"{repeated:,} repeated rows", repeated, found=repeated > 0, status_if_found="WARNING",
            action_if_found="Keep. Repeated attempts can be a fraud signal, so we do not drop them",
        )

    def _check_value_ranges(self, df: pd.DataFrame) -> dict:
        """Q4: prices are above 0, and day, night and month counts are not negative."""
        out_of_range = pd.Series(False, index=df.index)
        for col in self.quality["positive_columns"]:
            out_of_range |= df[col] <= 0
        for col in self.quality["non_negative_columns"]:
            out_of_range |= df[col] < 0
        rows = int(out_of_range.sum())

        result = (
            f"{rows} rows out of range. "
            f"Price {df['price_euro'].min():,.0f} to {df['price_euro'].max():,.0f} EUR, "
            f"book window {df['book_window'].min()} to {df['book_window'].max()} days, "
            f"stay {df['length_of_stay'].min()} to {df['length_of_stay'].max()} nights, "
            f"property age {df['property_age'].min()} to {df['property_age'].max()} months"
        )
        return self._report_row(
            "Value ranges", "price_euro > 0, and book_window, length_of_stay and property_age >= 0",
            result, rows, found=rows > 0, status_if_found="WARNING",
            action_if_found="Keep. Look at these rows by hand before the rules and the model use them",
        )

    def _check_zero_night_stays(self, df: pd.DataFrame) -> dict:
        """Q5: reservations with 0 nights. A real stay has at least one night."""
        rows = int((df["length_of_stay"] == 0).sum())
        return self._report_row(
            "Zero-night stays", "length_of_stay == 0", f"{rows:,} attempts", rows,
            found=rows > 0, status_if_found="WARNING",
            action_if_found="Keep. Rule R8 uses them",
        )

    def _check_text_values(self, df: pd.DataFrame) -> dict:
        """Q6: text values with spaces at the start or end, or the same value in another case.

        For example "visa" and "Visa" would be two values for one payment method.
        """
        problem_rows = pd.Series(False, index=df.index)
        problem_columns = []
        for col in text_columns(df):
            trimmed = df[col].str.strip()
            has_spaces = df[col] != trimmed
            spellings_per_value = trimmed.groupby(trimmed.str.lower()).transform("nunique")
            has_case_variant = spellings_per_value > 1
            if (has_spaces | has_case_variant).any():
                problem_columns.append(col)
            problem_rows |= has_spaces | has_case_variant

        rows = int(problem_rows.sum())
        return self._report_row(
            "Text values", "Spaces at the start or end, or the same value in a different case",
            f"Found in {problem_columns}" if problem_columns else "None", rows,
            found=rows > 0, status_if_found="WARNING",
            action_if_found="Keep. Clean the spelling before counting per value",
        )

    def _check_known_values(self, df: pd.DataFrame) -> dict:
        """Q7: every language is in the language map, and the three country columns share one list of names."""
        unknown_language = ~df["language"].isin(list(self.language_country))
        country_sets = [set(df[col]) for col in COUNTRY_COLUMNS]
        shared_countries = set.intersection(*country_sets)
        all_countries = set.union(*country_sets)
        unshared_country = pd.Series(False, index=df.index)
        for col in COUNTRY_COLUMNS:
            unshared_country |= ~df[col].isin(shared_countries)

        rows = int((unknown_language | unshared_country).sum())
        languages = df["language"].nunique()
        result = (
            f"{languages} languages, {languages - df.loc[unknown_language, 'language'].nunique()} in the map; "
            f"{len(all_countries)} countries, {len(shared_countries)} used in all three columns"
        )
        return self._report_row(
            "Known values",
            "Every language is in the language map, and the three country columns use the same names",
            result, rows, found=rows > 0, status_if_found="WARNING",
            action_if_found="Keep. Add the missing names to the language map",
            action_if_ok="None. R7 can check every attempt",
        )

    def _check_one_value_per_id(self, df: pd.DataFrame) -> dict:
        """Q8: an ID always comes with the same detail.

        A card has one issuing country and one payment method, an email has one
        domain, and a property has one country. Two values for one ID would
        point to an error in the data.
        """
        pairs = [
            ("credit_card_id", "card_issuing_country"),
            ("credit_card_id", "payment_method"),
            ("email_id", "email_domain"),
            ("hotel_id", "hotel_country"),
        ]
        problem_rows = pd.Series(False, index=df.index)
        conflicts = []
        for id_col, detail_col in pairs:
            values_per_id = df.groupby(id_col)[detail_col].transform("nunique")
            conflict_ids = df.loc[values_per_id > 1, id_col].nunique()
            if conflict_ids > 0:
                conflicts.append(f"{conflict_ids} {id_col} with more than one {detail_col}")
            problem_rows |= values_per_id > 1

        rows = int(problem_rows.sum())
        return self._report_row(
            "One value per ID",
            "Each card has one issuing country and payment method, each email one domain, each property one country",
            "; ".join(conflicts) if conflicts else "No conflicts", rows,
            found=rows > 0, status_if_found="WARNING",
            action_if_found="Keep. This data cannot show which value is right",
        )

    def _check_card_id_for_every_method(self, df: pd.DataFrame) -> dict:
        """Q9: payments that are not made with a card (such as PayPal) still have a card ID."""
        non_card_methods = self.quality["non_card_payment_methods"]
        non_card = df["payment_method"].str.lower().isin(non_card_methods)
        with_card_id = non_card & df["credit_card_id"].notna() & df["card_issuing_country"].notna()
        rows = int(with_card_id.sum())
        return self._report_row(
            "Card ID for every payment method",
            "credit_card_id and card_issuing_country are also filled for " + " and ".join(non_card_methods),
            f"{rows:,} {' or '.join(non_card_methods)} attempts with a card ID", rows,
            found=rows > 0, status_if_found="NOTE",
            action_if_found="Keep. We treat credit_card_id as the ID of the card or account behind every payment",
        )

    def _check_ids_from_many_countries(self, df: pd.DataFrame) -> dict:
        """Q10: IPs, cards and emails that come with more than one booker_country (see ids_from_many_countries)."""
        many_countries = self.ids_from_many_countries(df)
        parts = []
        rows_with_listed_id = pd.Series(False, index=df.index)
        for id_col in BOOKER_ID_COLUMNS:
            listed = many_countries[many_countries["id type"] == id_col]
            parts.append(f"{len(listed)} {ID_NAMES[id_col]} ({listed['attempts'].sum():,} attempts)")
            rows_with_listed_id |= df[id_col].isin(listed["id"])

        rows = int(rows_with_listed_id.sum())
        return self._report_row(
            "IDs with more than one booker country", "IPs, cards and emails seen with more than one booker_country",
            ", ".join(parts), rows, found=rows > 0, status_if_found="NOTE",
            action_if_found="Keep. This is not a data error but a fraud signal",
        )
