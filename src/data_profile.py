"""Describe the loaded data with a few simple tables.

These tables help the reader understand the data before any processing:
what each column means, how big the data is, how the payment provider
answered, and how many attempts there are per day.
"""

import pandas as pd

from src.data_loader import COLUMN_MEANINGS, EXPECTED_COLUMNS, ID_NAMES


class DataProfiler:
    """Build the description tables that the notebook shows first.

    The profiler only reads the data.

    Args:
        config: The settings from config.yaml, as a dictionary.
    """

    def __init__(self, config: dict):
        """Keep the settings that this class needs."""
        self.declined_results = config["features"]["declined_results"]

    @staticmethod
    def column_profile(df: pd.DataFrame) -> pd.DataFrame:
        """Describe each column: what it means, its type and its values.

        Args:
            df: The loaded data.

        Returns:
            One row per original column, with its meaning, type, number of
            distinct values, an example value, and the min and max of number columns.
        """
        rows = []
        for col in EXPECTED_COLUMNS:
            is_number = pd.api.types.is_numeric_dtype(df[col])
            is_date = pd.api.types.is_datetime64_any_dtype(df[col])
            # The min and max of an ID number say nothing, so we leave them out.
            show_range = is_number and col not in ["transaction_id", "hotel_id"]
            example = df[col].iloc[0]
            rows.append({
                "column": col,
                "meaning": COLUMN_MEANINGS[col],
                "type": str(df[col].dtype),
                "distinct values": df[col].nunique(),
                "example": f"{example:%Y-%m-%d}" if is_date else example,
                "min": df[col].min() if show_range else "",
                "max": df[col].max() if show_range else "",
            })
        return pd.DataFrame(rows)

    @staticmethod
    def dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
        """Give the size of the data, and show how often each ID comes back.

        In normal traffic almost every card, email and IP is new, so an ID that
        comes back is worth a look.

        Args:
            df: The loaded data.

        Returns:
            A table with one row for attempts, one for days, and one each for
            cards, emails, IPs and properties. For the four ID rows it also
            shows how many IDs are used in more than one attempt, and how many
            attempts they cover.
        """
        rows = [
            {"item": "attempts", "count": len(df)},
            {"item": "days", "count": df["day"].nunique()},
        ]
        for id_col, name in ID_NAMES.items():
            attempts_per_id = df[id_col].value_counts()
            reused = attempts_per_id[attempts_per_id > 1]
            rows.append({
                "item": f"distinct {name}",
                "count": len(attempts_per_id),
                "used in more than one attempt": len(reused),
                "attempts covered by these": int(reused.sum()),
            })
        summary = pd.DataFrame(rows)
        for col in ["used in more than one attempt", "attempts covered by these"]:
            summary[col] = summary[col].astype("Int64")
        return summary

    @staticmethod
    def auth_result_mix(df: pd.DataFrame) -> pd.DataFrame:
        """Count each answer of the payment provider.

        Args:
            df: The loaded data.

        Returns:
            One row per auth_result value, with the number of attempts and the
            share of all attempts in percent, largest first.
        """
        counts = df["auth_result"].value_counts()
        return pd.DataFrame({
            "auth_result": counts.index,
            "attempts": counts.values,
            "share (%)": (100 * counts.values / len(df)).round(1),
        })

    @staticmethod
    def split_summary(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
        """Give the size of the two parts of the time split.

        Args:
            train_df: The attempts of the training days.
            test_df: The attempts of the test days.

        Returns:
            One row per part, with its days, the attempts and the number of days.
        """
        rows = []
        for name, part in [("train", train_df), ("test", test_df)]:
            rows.append({
                "split": f"{name} (days {part['day'].min()} to {part['day'].max()})",
                "attempts": len(part),
                "days": part["day"].nunique(),
            })
        return pd.DataFrame(rows)

    def decline_share_by(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Count attempts and declines for each value of one column.

        "Declined" means the payment was not approved: auth_result is one of
        the declined results in the config (REFUSED, FRAUD or ERROR).
        A value with a much higher decline share than the others is worth a
        look. It can be an attack, or a technical problem with one payment
        method or one market.

        Args:
            df: The loaded data.
            column: The column to group by, for example "payment_method",
                "booker_country", "email_domain" or "day".

        Returns:
            One row per value of the column, with the attempts, the declines
            and the decline share in percent.
        """
        is_declined = df["auth_result"].isin(self.declined_results)
        table = pd.DataFrame({
            "attempts": df.groupby(column).size(),
            "declines": is_declined.groupby(df[column]).sum(),
        })
        table["decline share (%)"] = (100 * table["declines"] / table["attempts"]).round(1)
        return table.reset_index()

    def daily_overview(self, df: pd.DataFrame) -> pd.DataFrame:
        """Count attempts and declines per day.

        Args:
            df: The loaded data.

        Returns:
            One row per day (1 to 31), with the attempts, the declines and
            the decline share in percent.
        """
        return self.decline_share_by(df, "day")
