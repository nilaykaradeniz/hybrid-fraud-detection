"""Measure the two setups on the test days, and check the model, the label and the split.

The hybrid system and the rules-only baseline are compared on the test
days. "Fraud" here always means label 1 of the investigation label of
the whole month, whatever the payment provider answered. A "good
reservation" is a test-day attempt with label 0.

The comparison tables work on the output of DecisionPolicy.decide().
The tables about the label only need the labeled attempts, and the
tables about the risk score only need the scored attempts. The
Evaluator only reads them and learns nothing from the test days. The one
model in this file (label_check_auc) is a check of the FRAUD answer of
the payment provider. It is not part of the system.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.decision import FRICTION_ACTIONS
from src.labels import AUTHORISED

# The two setups and the column that holds their action.
SETUPS = {"rules only": "rules_only_action", "hybrid": "hybrid_action"}
# Precision and recall are counted per attempt and per identity.
UNIT_COLUMNS = {"attempt": "transaction_id", "card": "credit_card_id", "IP": "ip_id", "property": "hotel_id"}
CAUGHT = "fraud value caught (EUR)"
FRAUD = "FRAUD"
LABEL_CHECK_CATEGORIES = [
    "booker_country", "card_issuing_country", "hotel_country", "language", "payment_method", "email_domain",
]
LABEL_CHECK_NUMBERS = ["price_euro", "book_window", "length_of_stay", "property_age"]


class Evaluator:
    """Build the result tables of the project.

    Args:
        config: The settings from config.yaml, as a dictionary.
    """

    def __init__(self, config: dict):
        """Keep the settings that this class needs."""
        settings = config["evaluation"]
        self.top_shares = settings["top_shares"]
        self.threshold_check_value_share = settings["threshold_check_value_share"]
        self.label_check_folds = settings["label_check_folds"]
        self.tradeoff_percentiles = settings["tradeoff_percentiles"]
        self.train_last_day = config["split"]["train_last_day"]
        self.bootstrap_rounds = config["split_check"]["bootstrap_rounds"]
        self.random_state = config["model"]["random_state"]
        self.young_property_months = config["rules"]["R4"]["property_age_below"]

    # ------------------------------------------------------------------
    # The comparison of the two setups
    # ------------------------------------------------------------------

    def compare_setups(self, decisions_df: pd.DataFrame) -> pd.DataFrame:
        """Give the main comparison: one row per metric, one column per setup, on the test days.

        The metrics are: the friction actions and the share of them that
        fell on good reservations (false positive share), the good
        reservations with friction (count, share of all good reservations
        and EUR), the fraud value caught (EUR, its authorised part, and as
        a share of what the rules alone catch), the manual reviews per day,
        precision and recall per attempt, card, IP and property, and the
        soft-only alerts per day.

        The false positive share is 100% minus the precision per attempt.
        It is not the "false positive rate" of a textbook, which is the
        share of all good reservations that get friction. Both are in the table.

        Precision is the share of flagged units that are fraud, and recall
        is the share of fraud units that are flagged. An attempt is flagged
        if it got friction. A card, IP or property is fraud if one of its
        test-day attempts is fraud, and flagged if one of them got friction.

        Args:
            decisions_df: The attempts of all days (output of DecisionPolicy.decide()).

        Returns:
            The table, with the columns metric, rules only and hybrid.
        """
        test = self._test_days(decisions_df)
        table = pd.DataFrame({name: self._setup_metrics(test, column) for name, column in SETUPS.items()})
        share_name = "fraud value caught, share of the rules-only value (%)"
        table.loc[share_name] = 100 * table.loc[CAUGHT] / table.loc[CAUGHT, "rules only"]
        order = list(table.index)
        order.insert(order.index(CAUGHT) + 1, order.pop())
        return table.loc[order].round(1).rename_axis("metric").reset_index()

    def reviews_per_day(self, decisions_df: pd.DataFrame) -> pd.DataFrame:
        """Count the manual reviews of each test day, for both setups."""
        test = self._test_days(decisions_df)
        counts = {name: (test[column] == "manual_review").groupby(test["day"]).sum() for name, column in SETUPS.items()}
        return pd.DataFrame(counts).reset_index()

    def segment_view(self, decisions_df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Show the comparison per segment, for example per value band or per property country.

        Args:
            decisions_df: The attempts of all days (output of DecisionPolicy.decide()).
            column: The column that defines the segments, "value_band" or "hotel_country".

        Returns:
            One row per segment of the test days: its fraud attempts and
            authorised fraud value, and for each setup the good reservations
            with friction (count and EUR) and the fraud value caught.
        """
        test = self._test_days(decisions_df)
        is_fraud = test["label"] == 1
        is_authorised = test["auth_result"] == AUTHORISED
        table = pd.DataFrame({
            "fraud attempts": is_fraud.groupby(test[column]).sum(),
            "authorised fraud value (EUR)": self._value_per_segment(test, column, is_fraud & is_authorised),
        })
        for name, action_column in SETUPS.items():
            friction = test[action_column].isin(FRICTION_ACTIONS)
            good_with_friction = friction & ~is_fraud
            table[f"{name}: good reservations with friction"] = good_with_friction.groupby(test[column]).sum()
            table[f"{name}: good value with friction (EUR)"] = self._value_per_segment(test, column, good_with_friction)
            table[f"{name}: fraud value caught (EUR)"] = self._value_per_segment(test, column, friction & is_fraud)
        return table.round(0).astype(int).reset_index()

    def properties_at_risk(self, decisions_df: pd.DataFrame) -> pd.DataFrame:
        """List every property with at least one fraud attempt, with a recommended action.

        This table uses all days, not only the test days: a property that
        was used in an attack on the training days is still at risk.

        Returns:
            One row per property: its fraud attempts, their value and the
            authorised part of it in EUR, the property age in months and the
            action. A property younger than the R4 age limit (3 months) is a
            likely fake listing, an older one may be a real property that
            fraudsters used.
        """
        fraud = decisions_df[decisions_df["label"] == 1]
        authorised_value = fraud["price_euro"].where(fraud["auth_result"] == AUTHORISED, 0)
        table = fraud.groupby("hotel_id").agg(
            hotel_country=("hotel_country", "first"),
            fraud_attempts=("transaction_id", "count"),
            fraud_value=("price_euro", "sum"),
            property_age=("property_age", "first"),
        )
        table["authorised_value"] = authorised_value.groupby(fraud["hotel_id"]).sum()
        table["recommended action"] = "hold the payout and verify the owner"
        is_young = table["property_age"] < self.young_property_months
        table.loc[is_young, "recommended action"] = "suspend the listing and hold the payout"
        table = table.round({"fraud_value": 0, "authorised_value": 0}).rename(columns={
            "fraud_attempts": "fraud attempts", "fraud_value": "fraud value (EUR)",
            "authorised_value": "authorised value (EUR)", "property_age": "property age (months)",
        })
        return table.sort_values("fraud value (EUR)", ascending=False).reset_index()

    # ------------------------------------------------------------------
    # The risk score
    # ------------------------------------------------------------------

    @staticmethod
    def score_metrics(attempts_df: pd.DataFrame) -> dict:
        """Measure how well the risk score ranks fraud above normal attempts.

        ROC AUC is the chance that a random fraud attempt has a higher score
        than a random normal attempt: 0.5 is a coin flip, 1.0 is perfect.
        PR AUC (average precision) looks only at the top of the ranking, so
        it is the harder test when fraud is rare.

        Args:
            attempts_df: The attempts to measure, with "label" and "risk_score".

        Returns:
            The two numbers, as a dictionary.
        """
        return {
            "ROC AUC": roc_auc_score(attempts_df["label"], attempts_df["risk_score"]),
            "PR AUC": average_precision_score(attempts_df["label"], attempts_df["risk_score"]),
        }

    def score_metric_intervals(self, decisions_df: pd.DataFrame) -> pd.DataFrame:
        """Give ROC AUC and PR AUC on the test days, each with a 95% interval.

        The interval comes from a bootstrap: we draw the test-day attempts
        again with replacement, bootstrap_rounds (500) times, and measure
        each draw. 95% of the draws fall inside the interval. A narrow
        interval means that the test days are long enough to trust the number.

        Known limits: the interval shows the effect of which attempts are in
        the test days. It does not show the effect of the random seed of the
        model. And the draws treat the attempts as independent, but the fraud
        attempts of the test days come from one attack. With one attack, the
        real uncertainty is larger than the interval.

        Returns:
            One row per metric, with its value and the low and high end of the interval.
        """
        test = self._test_days(decisions_df)[["label", "risk_score"]]
        random = np.random.RandomState(self.random_state)
        draws = pd.DataFrame([
            self.score_metrics(test.sample(len(test), replace=True, random_state=random))
            for _ in range(self.bootstrap_rounds)
        ])
        table = pd.DataFrame({
            "value": self.score_metrics(test),
            "95% interval, low": draws.quantile(0.025),
            "95% interval, high": draws.quantile(0.975),
        })
        return table.round(3).rename_axis("metric").reset_index()

    def new_pattern_attempts(self, decisions_df: pd.DataFrame) -> pd.DataFrame:
        """Keep the "new-pattern" attempts.

        A new-pattern attempt is a fraud attempt on the test days whose
        card was first seen after the training days. The hotlist cannot
        know such a card.
        """
        first_day_of_card = decisions_df.groupby("credit_card_id")["day"].transform("min")
        is_new_pattern = (decisions_df["label"] == 1) & (first_day_of_card > self.train_last_day)
        return decisions_df[is_new_pattern]

    def new_pattern_result(self, decisions_df: pd.DataFrame) -> pd.DataFrame:
        """Check if the model finds fraud that uses cards it has never seen.

        Returns:
            One row per share in top_shares (5% and 1%): how many of the
            new-pattern attempts are in that top share of the risk scores of
            all test-day attempts.
        """
        test = self._test_days(decisions_df)
        new_pattern = self.new_pattern_attempts(decisions_df)
        rows = []
        for share in self.top_shares:
            cut = test["risk_score"].quantile(1 - share)
            rows.append({
                "new-pattern attempts": len(new_pattern),
                "top share of test scores (%)": 100 * share,
                "new-pattern attempts in it (%)": round(100 * (new_pattern["risk_score"] >= cut).mean(), 1),
            })
        return pd.DataFrame(rows)

    def threshold_check(self, decisions_df: pd.DataFrame, train_label: pd.Series, band_cut: float) -> pd.DataFrame:
        """Compare two ways to set the cut point of the risk score.

        The fraud-based cut uses the fraud of the training days: we sort
        the fraud alerts by risk score, highest first, and take the score
        at which they hold 95% of their total value (the share is in the
        config). The band cut is the first cut point of the risk bands,
        which comes from the scores of all training attempts.

        Args:
            decisions_df: The attempts of all days (output of DecisionPolicy.decide()).
            train_label: The label of the training days, one value per training attempt.
            band_cut: The first cut point of the risk bands (RiskModel.band_cuts[0]).

        Returns:
            One row per cut, with its score and the share of the new-pattern
            attempts at or above it. A cut that the new attack stays below
            would not have caught it.
        """
        train = decisions_df[decisions_df["day"] <= self.train_last_day]
        fraud_alerts = train[(train["alert_type"] != "none") & (train_label.loc[train.index] == 1)]
        by_score = fraud_alerts.sort_values("risk_score", ascending=False)
        value_so_far = by_score["price_euro"].cumsum()
        reaches_share = value_so_far >= self.threshold_check_value_share * by_score["price_euro"].sum()
        fraud_cut = by_score.loc[reaches_share, "risk_score"].iloc[0]

        new_pattern_scores = self.new_pattern_attempts(decisions_df)["risk_score"]
        cuts = {"fraud-based cut": fraud_cut, "band cut (95th percentile of all training scores)": band_cut}
        rows = [
            {"cut": name, "risk score": round(cut, 3),
             "new-pattern attempts at or above it (%)": round(100 * (new_pattern_scores >= cut).mean(), 1)}
            for name, cut in cuts.items()
        ]
        return pd.DataFrame(rows)

    def cut_point_tradeoff(self, decisions_df: pd.DataFrame, band_cut: float) -> pd.DataFrame:
        """Show what other cut points for the medium risk band would do on the test days.

        The risk score only decides on attempts without a hard alert. So
        the other cut points are percentiles (in the config) of the risk
        scores of the training attempts without a hard alert. For each
        cut, an attempt gets friction if it has a hard alert, or if it
        has only soft alerts and its score is at or above the cut.

        A lower cut catches more of the fraud that the hard rules miss,
        and it also sends more good customers to an extra check.

        Args:
            decisions_df: The attempts of all days (output of DecisionPolicy.decide()).
            band_cut: The cut point in use (RiskModel.band_cuts[0]).

        Returns:
            One row per cut: its risk score, the share of the fraud value
            caught, the good reservations with friction per test day and
            the false positive share.
        """
        train = decisions_df[decisions_df["day"] <= self.train_last_day]
        test = self._test_days(decisions_df)
        scores_without_hard_alert = train.loc[train["alert_type"] != "hard", "risk_score"]
        cuts = {"cut point in use": band_cut}
        for percentile in self.tradeoff_percentiles:
            cuts[f"percentile {percentile} without hard alerts"] = np.percentile(scores_without_hard_alert, percentile)

        is_fraud = test["label"] == 1
        soft_only = test["alert_type"] == "soft only"
        rows = []
        for name, cut in cuts.items():
            friction = (test["alert_type"] == "hard") | (soft_only & (test["risk_score"] >= cut))
            good_with_friction = (friction & ~is_fraud).sum()
            caught = test.loc[friction & is_fraud, "price_euro"].sum()
            rows.append({
                "cut point": name,
                "risk score": round(cut, 3),
                "fraud value caught (%)": round(100 * caught / test.loc[is_fraud, "price_euro"].sum(), 1),
                "good reservations with friction per day": round(good_with_friction / test["day"].nunique(), 1),
                "false positive share (%)": round(100 * good_with_friction / friction.sum(), 1),
            })
        return pd.DataFrame(rows)

    def score_distribution(self, decisions_df: pd.DataFrame) -> pd.DataFrame:
        """Give the risk score of each attempt with its split and label, ready for a histogram."""
        table = decisions_df[["risk_score", "label"]].copy()
        table.insert(0, "split", "test")
        table.loc[decisions_df["day"] <= self.train_last_day, "split"] = "train"
        return table

    # ------------------------------------------------------------------
    # The label
    # ------------------------------------------------------------------

    @staticmethod
    def daily_cluster_attempts(decisions_df: pd.DataFrame) -> pd.DataFrame:
        """Count the attempts of each cluster per day (one row per day, one column per cluster)."""
        in_cluster = decisions_df[decisions_df["label"] == 1]
        table = pd.crosstab(in_cluster["day"], in_cluster["cluster_id"])
        table.columns = [f"cluster {cluster}" for cluster in table.columns]
        return table.reset_index()

    @staticmethod
    def fraud_answer_rate_by(decisions_df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Give the share of FRAUD answers of the payment provider for each value of one column.

        If the FRAUD answer followed a pattern, some values would have a
        much higher share than others.

        Args:
            decisions_df: The attempts of all days.
            column: The column to group by. "price_decile" cuts price_euro into ten equal groups.

        Returns:
            One row per value, with the attempts and the FRAUD share in percent.
        """
        groups = decisions_df[column] if column != "price_decile" else pd.qcut(
            decisions_df["price_euro"], 10, labels=range(1, 11)).rename("price_decile")
        is_fraud_answer = decisions_df["auth_result"] == FRAUD
        table = pd.DataFrame({
            "attempts": is_fraud_answer.groupby(groups, observed=True).size(),
            "FRAUD share (%)": (100 * is_fraud_answer.groupby(groups, observed=True).mean()).round(2),
        })
        return table.reset_index()

    @staticmethod
    def auth_results_by_label(decisions_df: pd.DataFrame) -> pd.DataFrame:
        """Count each answer of the payment provider inside and outside the clusters.

        This shows how many FRAUD answers fall inside the attack clusters,
        and what the provider answered to the attack attempts.
        """
        in_cluster = decisions_df["label"].map({1: "in a cluster", 0: "not in a cluster"})
        return pd.crosstab(decisions_df["auth_result"], in_cluster).reset_index()

    def label_check_auc(self, decisions_df: pd.DataFrame) -> float:
        """Test if a supervised model can learn the FRAUD answer of the payment provider.

        A gradient boosting model is trained on the raw columns to predict
        the FRAUD answer, with 5-fold cross-validation: each attempt is
        predicted by a model that did not see it.

        Known limit: the folds are random, so the same days are in the
        training and the test folds. This makes the result optimistic. If
        the score is still close to 0.5, the FRAUD answer has no pattern
        that a model could learn.

        Returns:
            The ROC AUC of the predictions.
        """
        inputs = decisions_df[LABEL_CHECK_CATEGORIES + LABEL_CHECK_NUMBERS].copy()
        for column in LABEL_CHECK_CATEGORIES:
            inputs[column] = inputs[column].astype("category")
        is_fraud_answer = (decisions_df["auth_result"] == FRAUD).astype(int)
        model = HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=self.random_state)
        folds = StratifiedKFold(n_splits=self.label_check_folds, shuffle=True, random_state=self.random_state)
        predictions = cross_val_predict(model, inputs, is_fraud_answer, cv=folds, method="predict_proba")[:, 1]
        return float(roc_auc_score(is_fraud_answer, predictions))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _test_days(self, decisions_df: pd.DataFrame) -> pd.DataFrame:
        """Keep the attempts of the test days."""
        return decisions_df[decisions_df["day"] > self.train_last_day]

    @staticmethod
    def _value_per_segment(test: pd.DataFrame, column: str, chosen: pd.Series) -> pd.Series:
        """Sum the price_euro of the chosen attempts for each value of the segment column."""
        return test["price_euro"].where(chosen, 0).groupby(test[column]).sum()

    @staticmethod
    def _setup_metrics(test: pd.DataFrame, action_column: str) -> dict:
        """Compute the metrics of one setup on the test days."""
        friction = test[action_column].isin(FRICTION_ACTIONS)
        is_fraud = test["label"] == 1
        soft_only = test["alert_type"] == "soft only"
        good_with_friction = test[friction & ~is_fraud]
        caught = test[friction & is_fraud]
        test_days = test["day"].nunique()
        metrics = {
            "friction actions": int(friction.sum()),
            "false positive share (%)": 100 * len(good_with_friction) / friction.sum(),
            "good reservations with friction": len(good_with_friction),
            "good reservations with friction, share of all good reservations (%)":
                100 * len(good_with_friction) / (~is_fraud).sum(),
            "good reservations with friction (EUR)": good_with_friction["price_euro"].sum(),
            CAUGHT: caught["price_euro"].sum(),
            "authorised part of the fraud value caught (EUR)":
                caught.loc[caught["auth_result"] == AUTHORISED, "price_euro"].sum(),
            "manual reviews per day": (test[action_column] == "manual_review").sum() / test_days,
        }
        for unit, id_column in UNIT_COLUMNS.items():
            unit_is_fraud = is_fraud.groupby(test[id_column]).any()
            unit_is_flagged = friction.groupby(test[id_column]).any()
            flagged_fraud = (unit_is_fraud & unit_is_flagged).sum()
            metrics[f"precision per {unit} (%)"] = 100 * flagged_fraud / unit_is_flagged.sum()
            metrics[f"recall per {unit} (%)"] = 100 * flagged_fraud / unit_is_fraud.sum()
        metrics["soft-only alerts per day"] = soft_only.sum() / test_days
        metrics["soft-only alerts with friction per day"] = (soft_only & friction).sum() / test_days
        return metrics
