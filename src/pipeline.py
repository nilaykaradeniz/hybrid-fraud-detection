"""Run the whole project in one call, from the data file to the result tables.

The pipeline has seven steps, in this order:

1. load the data, check it and split it by time (DataLoader, with
   DataQualityChecker, DataProfiler and SplitChecker),
2. find the attack clusters and label their attempts (InvestigationLabeler),
3. build the feature columns (FeatureBuilder),
4. apply the rules (RuleEngine),
5. fit the risk model and score every attempt (RiskModel),
6. choose one action per attempt for both setups (DecisionPolicy),
7. measure the two setups on the test days (Evaluator).

Each run can save a "run folder" with the settings, the main tables and
the model, so that a result can be checked later and the PSI of the
columns can be followed from run to run.

Run it from the project folder with: python -m src.pipeline
"""

import copy
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.data_loader import DataLoader, load_config
from src.data_profile import DataProfiler
from src.data_quality import DataQualityChecker
from src.decision import DecisionPolicy
from src.evaluation import Evaluator
from src.features import MODEL_FEATURES, FeatureBuilder
from src.labels import InvestigationLabeler
from src.risk_model import RiskModel
from src.rules import RuleEngine
from src.split_check import SplitChecker

DECISION_FILE_COLUMNS = [
    "transaction_id", "day", "rules_fired", "risk_score", "risk_band", "value_band",
    "hybrid_action", "rules_only_action", "hold_payout",
]


@dataclass
class PipelineResults:
    """The main tables of one run.

    Attributes:
        train_last_day: The last training day of this run.
        quality_report: The data quality report, one row per check.
        split_summary: The days and attempts of the training and the test part.
        psi_raw_columns: The PSI of the raw columns, training days against test days.
        psi_model_features: The same check for the model features.
        clusters: The clusters of the whole month, one row per cluster.
        train_clusters: The clusters that the training days alone show.
        rule_summary: The alerts of each rule and how many of them are fraud.
        decisions: One row per attempt of all days, with every column that the steps added.
        comparison: The main comparison of the two setups on the test days.
        score_metrics: ROC AUC and PR AUC of the risk score on the test days.
        band_cuts: The two risk scores where the medium and the high risk band start.
        run_folder: The folder of the saved run, or nothing if the run was not saved.
    """

    train_last_day: int
    quality_report: pd.DataFrame
    split_summary: pd.DataFrame
    psi_raw_columns: pd.DataFrame
    psi_model_features: pd.DataFrame
    clusters: pd.DataFrame
    train_clusters: pd.DataFrame
    rule_summary: pd.DataFrame
    decisions: pd.DataFrame
    comparison: pd.DataFrame
    score_metrics: dict
    band_cuts: list
    run_folder: str = None


class FraudPipeline:
    """Call the classes of the project in the right order.

    The pipeline has no fraud logic of its own. It only passes the output
    of one step to the next step.

    Args:
        config_path: The settings file.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """Read the settings."""
        self.config_path = config_path
        self.config = load_config(config_path)

    def run(self, train_last_day: int = None, save_outputs: bool = True,
            new_ip_per_fraud_attempt: bool = False) -> PipelineResults:
        """Run the seven steps and collect the main tables.

        Args:
            train_last_day: The last training day. Leave it empty to use the day of the config (14).
            save_outputs: If True, save a run folder.
            new_ip_per_fraud_attempt: If True, run the stress test of _give_fraud_attempts_new_ips().

        Returns:
            The main tables of the run.
        """
        config = self._config_with_split_day(train_last_day)
        loader = DataLoader(config)
        labeler = InvestigationLabeler(config)
        df = loader.load()
        labeled = labeler.label(df)
        if new_ip_per_fraud_attempt:
            df, labeled = self._give_fraud_attempts_new_ips(df, labeled)
        train_df, test_df = loader.split(df)
        train_labeled = labeler.label(train_df)

        features = FeatureBuilder(config).fit(train_df).transform(labeled)
        is_train = features["day"] <= config["split"]["train_last_day"]
        engine = RuleEngine(config).fit(train_labeled)
        alerts = engine.apply(features)
        model = RiskModel(config).fit(alerts[is_train])
        decisions = DecisionPolicy(config).decide(model.score(alerts))

        evaluator = Evaluator(config)
        split_checker = SplitChecker(config)
        results = PipelineResults(
            train_last_day=config["split"]["train_last_day"],
            quality_report=DataQualityChecker(config).quality_report(df),
            split_summary=DataProfiler(config).split_summary(train_df, test_df),
            psi_raw_columns=split_checker.psi_table(train_df, test_df, config["split_check"]["raw_columns"]),
            psi_model_features=split_checker.psi_table(features[is_train], features[~is_train], MODEL_FEATURES),
            clusters=labeler.describe_clusters(labeled),
            train_clusters=labeler.describe_clusters(train_labeled),
            rule_summary=engine.rule_summary(alerts, train_labeled["label"]),
            decisions=decisions,
            comparison=evaluator.compare_setups(decisions),
            score_metrics=evaluator.score_metrics(decisions[~is_train]),
            band_cuts=model.band_cuts,
        )
        if save_outputs:
            rule_columns = engine.rule_columns("hard") + engine.rule_columns("soft")
            results.run_folder = self._save_run(results, model, rule_columns)
        return results

    def split_day_sensitivity(self) -> pd.DataFrame:
        """Run the pipeline with other split days, to see if the results depend on day 14.

        The days to try are in the config (sensitivity_last_days: 10, 14
        and 18). Each run learns everything again from its own training
        days: the price limits, the hotlist of rule R13, the model and the
        risk bands. No run folder is saved.

        Known limit: each row has other test days, so the rows show if the
        story holds, not which split day is best.

        Returns:
            One row per split day: the training days, the test days, the
            clusters in the training days, the fraud attempts on the test
            days, ROC AUC and PR AUC, the false positive share of both
            setups, and the share of the fraud value that the hybrid system
            still catches.
        """
        rows = []
        for day in self.config["split_check"]["sensitivity_last_days"]:
            results = self.run(train_last_day=day, save_outputs=False)
            comparison = results.comparison.set_index("metric")
            test = results.decisions[results.decisions["day"] > day]
            rows.append({
                "training days": f"1 to {day}",
                "test days": f"{day + 1} to {results.decisions['day'].max()}",
                "clusters in the training days": len(results.train_clusters),
                "fraud attempts on the test days": int(test["label"].sum()),
                "ROC AUC": round(results.score_metrics["ROC AUC"], 3),
                "PR AUC": round(results.score_metrics["PR AUC"], 3),
                "false positive share, rules only (%)": comparison.loc["false positive share (%)", "rules only"],
                "false positive share, hybrid (%)": comparison.loc["false positive share (%)", "hybrid"],
                "fraud value caught, hybrid (% of rules only)":
                    comparison.loc["fraud value caught, share of the rules-only value (%)", "hybrid"],
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _config_with_split_day(self, train_last_day: int = None) -> dict:
        """Give a copy of the settings, with another last training day if one is given."""
        config = copy.deepcopy(self.config)
        if train_last_day is not None:
            config["split"]["train_last_day"] = train_last_day
        return config

    @staticmethod
    def _give_fraud_attempts_new_ips(df: pd.DataFrame, labeled: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Stress test: change the data as if the attackers used a new IP for every attempt.

        In the real data one attack sends all its attempts from one IP, and
        the hard rules catch it through this IP. Here every fraud attempt
        gets its own IP, so no rule, feature or hotlist can link the
        attempts by IP. The label stays the one of the real data: we still
        know which attempts are fraud, but the system has to find them
        without the IP.

        Args:
            df: The loaded data.
            labeled: The same attempts with the label of the real data.

        Returns:
            Both tables with the changed ip_id column.
        """
        own_ip = "new_ip_" + df["transaction_id"].astype(str)
        ip_id = df["ip_id"].where(labeled["label"] == 0, own_ip)
        return df.assign(ip_id=ip_id), labeled.assign(ip_id=ip_id)

    def _save_run(self, results: PipelineResults, model: RiskModel, rule_columns: list[str]) -> str:
        """Write the run folder: the settings, the quality report, the two PSI tables, the metrics, the decisions and the model.

        Returns:
            The path of the new folder, outputs/run_<date_time>.
        """
        folder = Path(self.config["outputs"]["folder"]) / f"run_{datetime.now():%Y%m%d_%H%M%S}"
        folder.mkdir(parents=True)
        shutil.copy(self.config_path, folder / "config.yaml")
        results.quality_report.to_csv(folder / "quality_report.csv", index=False)
        results.psi_raw_columns.to_csv(folder / "psi_raw_columns.csv", index=False)
        results.psi_model_features.to_csv(folder / "psi_model_features.csv", index=False)
        with open(folder / "metrics.json", "w", encoding="utf-8") as file:
            json.dump(self._metrics(results), file, indent=2)
        self._decisions_file(results.decisions, rule_columns).to_csv(folder / "decisions.csv", index=False)
        model.save(folder / "model.joblib")
        return str(folder)

    @staticmethod
    def _metrics(results: PipelineResults) -> dict:
        """Collect the key numbers of a run for metrics.json."""
        comparison = results.comparison.set_index("metric")
        return {
            "train_last_day": results.train_last_day,
            "split": results.split_summary.to_dict("records"),
            "cluster attempts, whole month": results.clusters["attempts"].tolist(),
            "cluster attempts, training days": results.train_clusters["attempts"].tolist(),
            "risk score, test days": {name: round(float(value), 3) for name, value in results.score_metrics.items()},
            "risk band cut points": [round(cut, 3) for cut in results.band_cuts],
            "rules only": comparison["rules only"].to_dict(),
            "hybrid": comparison["hybrid"].to_dict(),
        }

    @staticmethod
    def _decisions_file(decisions: pd.DataFrame, rule_columns: list[str]) -> pd.DataFrame:
        """Keep the decision columns of each attempt, and name the rules that fired in one text column.

        Returns:
            One row per attempt of all days. "rules_fired" holds the rule
            names with a space between them, for example "R6 R12", and is
            empty if no rule fired.
        """
        table = decisions.copy()
        table["rules_fired"] = ""
        for column in rule_columns:
            table.loc[table[column], "rules_fired"] += column + " "
        table["rules_fired"] = table["rules_fired"].str.strip()
        return table[DECISION_FILE_COLUMNS]


def main() -> None:
    """Run the pipeline and print the data quality report and the main comparison."""
    results = FraudPipeline().run()
    print("Data quality report")
    print(results.quality_report.drop(columns="what we check").to_string(index=False))
    print("\nComparison of the two setups on the test days")
    print(results.comparison.to_string(index=False))
    print(f"\nRun folder: {results.run_folder}")


if __name__ == "__main__":
    main()
