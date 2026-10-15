"""Apply the fraud rules to the attempts.

A rule is a simple condition, for example "the same card was used more
than 3 times today". An attempt that meets the condition gets an alert.
The rules, their thresholds, their groups and their actions come from
config.yaml:

- a hard rule decides alone, with its own action,
- a soft rule only raises a flag, and the risk score decides what happens.

Rule R13 is different. It has no threshold but a hotlist: the cards,
emails, IPs and properties of the attacks that were confirmed on the
training days. The list is built once, after the last training day, so
R13 cannot fire on the training days.
"""

import pandas as pd

from src.data_loader import ID_NAMES

R13_COLUMNS = ["R13_card_or_email", "R13_ip_or_property"]


class RuleEngine:
    """Add one True/False alert column per rule, and summarise the alerts.

    Args:
        config: The settings from config.yaml, as a dictionary.
    """

    def __init__(self, config: dict):
        """Keep the settings that this class needs."""
        self.rules = config["rules"]
        self.train_last_day = config["split"]["train_last_day"]
        self.hotlist = None

    def fit(self, train_labeled_df: pd.DataFrame) -> "RuleEngine":
        """Build the hotlist of rule R13 from the clusters of the training days.

        Args:
            train_labeled_df: The training days with their own label
                (output of InvestigationLabeler.label() on the training days).

        Returns:
            The engine itself, so that fit(train_labeled_df).apply(df) works.
        """
        in_cluster = train_labeled_df[train_labeled_df["label"] == 1]
        self.hotlist = {id_col: set(in_cluster[id_col]) for id_col in ID_NAMES}
        return self

    def apply(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Add the alert columns of all rules.

        Args:
            features_df: The attempts of all days with their features
                (output of FeatureBuilder.transform()).

        Returns:
            A copy of features_df with one True/False column per rule (R1,
            R2, and so on). R13 has two columns: R13_card_or_email and
            R13_ip_or_property. It fires if either is True.

        Raises:
            ValueError: If fit() was not called first.
        """
        if self.hotlist is None:
            raise ValueError("Call fit() on the labeled training days before apply().")
        alerts = features_df.copy()
        for rule, fires in self._threshold_rules(alerts).items():
            alerts[rule] = fires

        after_training = alerts["day"] > self.train_last_day
        on_list = {id_col: alerts[id_col].isin(listed) for id_col, listed in self.hotlist.items()}
        alerts["R13_card_or_email"] = after_training & (on_list["credit_card_id"] | on_list["email_id"])
        alerts["R13_ip_or_property"] = after_training & (on_list["ip_id"] | on_list["hotel_id"])
        return alerts

    def rule_columns(self, group: str) -> list[str]:
        """Name the alert columns of one group of rules.

        Args:
            group: "hard" or "soft".

        Returns:
            The column names, in the order of the config.
        """
        columns = []
        for rule, settings in self.rules.items():
            if settings["group"] == group:
                columns += R13_COLUMNS if rule == "R13" else [rule]
        return columns

    def rule_summary(self, alerts_df: pd.DataFrame, train_label: pd.Series) -> pd.DataFrame:
        """Count the alerts of each rule and how many of them are fraud.

        Precision is the share of the alerts that are fraud. On the training
        days it uses the label of the training days, because this number
        decides if a rule is hard or soft. On the test days the whole-month
        label says which alerts are fraud.

        Args:
            alerts_df: The output of apply(), with the whole-month "label" column.
            train_label: The label of the training days, one value per training attempt.

        Returns:
            One row per alert column: its group, the alerts on the training
            days (in total and per day), the precision on the training days
            in percent, and the alerts, fraud and not-fraud counts on the
            test days.
        """
        is_train = alerts_df["day"] <= self.train_last_day
        train, test = alerts_df[is_train], alerts_df[~is_train]
        train_is_fraud = train_label.loc[train.index] == 1
        rows = []
        for group in ["hard", "soft"]:
            for column in self.rule_columns(group):
                train_alerts = int(train[column].sum())
                train_fraud = int((train[column] & train_is_fraud).sum())
                precision = round(100 * train_fraud / train_alerts, 1) if train_alerts else None
                test_alerts = int(test[column].sum())
                test_fraud = int((test[column] & (test["label"] == 1)).sum())
                rows.append({
                    "rule": column,
                    "group": group,
                    "alerts (training days)": train_alerts,
                    "alerts per training day": round(train_alerts / train["day"].nunique(), 1),
                    "precision (training days, %)": precision,
                    "alerts (test days)": test_alerts,
                    "fraud (test days)": test_fraud,
                    "not fraud (test days)": test_alerts - test_fraud,
                })
        return pd.DataFrame(rows)

    def _threshold_rules(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        """Give the condition of every rule except R13, with the thresholds of the config."""
        r = self.rules
        hard_declines_limit = r["R2"]["hard_declines_prev_day_at_least"]
        per_ip_limit = r["R3"]["distinct_cards_or_emails_per_ip_day_above"]
        same_property_limit = r["R5"]["same_property_attempts_7d_above"]
        linked_limit = r["R11"]["linked_identities_7d_above"]
        return {
            "R1": df["card_attempts_day"] > r["R1"]["card_attempts_day_above"],
            "R2": (df["card_hard_declines_prev_day"] >= hard_declines_limit)
                  | (df["ip_hard_declines_prev_day"] >= hard_declines_limit),
            "R3": (df["ip_distinct_cards_day"] > per_ip_limit) | (df["ip_distinct_emails_day"] > per_ip_limit),
            "R4": (df["property_age"] < r["R4"]["property_age_below"])
                  & (df["book_window"] <= r["R4"]["book_window_at_most"]),
            "R5": (df["card_property_attempts_7d"] > same_property_limit)
                  | (df["email_property_attempts_7d"] > same_property_limit),
            "R9": df["property_attempts_day"] > r["R9"]["property_attempts_day_above"],
            "R10": df["property_declines_7d_before"] >= r["R10"]["property_declines_7d_before_at_least"],
            "R11": (df["card_distinct_emails_7d"] > linked_limit) | (df["email_distinct_cards_7d"] > linked_limit),
            "R6": df["card_country_mismatch"],
            "R7": df["language_mismatch"],
            "R8": df["length_of_stay"] == r["R8"]["length_of_stay_equals"],
            "R12": (df["book_window"] <= r["R12"]["book_window_at_most"]) & df["high_price"],
        }
