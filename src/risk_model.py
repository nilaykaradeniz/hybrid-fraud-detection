"""Give each attempt a risk score, and explain where the score comes from.

The model is an Isolation Forest. It needs no labels: it learns what
the attempts of the training days look like, and gives a high score to
an attempt that is easy to separate from the others. So the risk score
says "how unusual is this attempt", not "is this fraud".

The score is turned into three risk bands (low, medium, high). The cut
points come from the scores of the training days, the same way a team
sets its review capacity: the top 1% is high, the next 4% is medium.

SHAP explains one score: it says how much each feature pushed the score
up or down. SHAP learns nothing, it only asks the fitted model.

Known limit: the training days hold attack attempts too, and the model
sees them as part of the data. With labels from real investigations, we
could leave them out.
"""

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import IsolationForest

from src.features import MODEL_FEATURES


class RiskModel:
    """Fit the Isolation Forest on the training days, then score and explain attempts.

    Args:
        config: The settings from config.yaml, as a dictionary.
    """

    def __init__(self, config: dict):
        """Keep the settings that this class needs and set up the model."""
        settings = config["model"]
        self.model = IsolationForest(
            n_estimators=settings["n_estimators"],
            random_state=settings["random_state"],
            n_jobs=settings["n_jobs"],
        )
        self.band_percentiles = settings["band_percentiles"]
        self.shap_sample_size = settings["shap_sample_size"]
        self.random_state = settings["random_state"]
        self.band_cuts = None

    def fit(self, train_features_df: pd.DataFrame) -> "RiskModel":
        """Fit the model and learn the risk band cut points, both on the training days.

        Args:
            train_features_df: The attempts of the training days with their features.

        Returns:
            The model itself, so that fit(train_features_df).score(df) works.
        """
        self.model.fit(train_features_df[MODEL_FEATURES])
        train_scores = self._risk_scores(train_features_df)
        self.band_cuts = [float(cut) for cut in np.percentile(train_scores, self.band_percentiles)]
        return self

    def score(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Add the risk score and the risk band of each attempt.

        Args:
            features_df: The attempts with their features, from any day.

        Returns:
            A copy of features_df with "risk_score" (higher means more
            unusual) and "risk_band": low below the first cut point, medium
            from the first cut point on, high from the second cut point on.

        Raises:
            ValueError: If fit() was not called first.
        """
        if self.band_cuts is None:
            raise ValueError("Call fit() on the training days before score().")
        medium_cut, high_cut = self.band_cuts
        scored = features_df.copy()
        scored["risk_score"] = self._risk_scores(features_df)
        scored["risk_band"] = "low"
        scored.loc[scored["risk_score"] >= medium_cut, "risk_band"] = "medium"
        scored.loc[scored["risk_score"] >= high_cut, "risk_band"] = "high"
        return scored

    def explain(self, features_df: pd.DataFrame) -> shap.Explanation:
        """Explain the scores of some attempts with SHAP.

        For an Isolation Forest, SHAP explains the average path length in
        the trees: the number of splits that it takes to separate the
        attempt. A short path means an unusual attempt, so the raw SHAP
        values are negative for risky features. We flip the sign and call
        the result "risk contribution": a positive value raises the risk.

        Known limit: the contributions of one attempt add up to its flipped
        path length. That number ranks the attempts in the same order as
        the risk score, but it is not the same number.

        Args:
            features_df: The attempts to explain, with their features.

        Returns:
            A shap.Explanation with one risk contribution per attempt and
            model feature. It can go straight into shap.plots.waterfall().
        """
        model_inputs = features_df[MODEL_FEATURES]
        explainer = shap.TreeExplainer(self.model)
        base_value = -float(np.ravel(explainer.expected_value)[0])
        return shap.Explanation(
            values=-explainer.shap_values(model_inputs),
            base_values=np.full(len(model_inputs), base_value),
            data=model_inputs.to_numpy(),
            feature_names=MODEL_FEATURES,
        )

    def global_importance(self, alerts_df: pd.DataFrame) -> pd.DataFrame:
        """Rank the features by how much they move the risk of alerted attempts.

        The table is built on a random sample of the given attempts
        (shap_sample_size in the config), to keep it fast.

        Args:
            alerts_df: The attempts to look at, for example the rule alerts of the test days.

        Returns:
            One row per model feature with its mean absolute risk
            contribution, largest first.
        """
        sample_size = min(len(alerts_df), self.shap_sample_size)
        sample = alerts_df.sample(sample_size, random_state=self.random_state)
        contributions = np.abs(self.explain(sample).values).mean(axis=0)
        table = pd.DataFrame({"feature": MODEL_FEATURES, "mean absolute risk contribution": contributions.round(3)})
        return table.sort_values("mean absolute risk contribution", ascending=False, ignore_index=True)

    def save(self, path: str) -> None:
        """Save the fitted model to a file, so that a later run can load it with joblib."""
        joblib.dump(self.model, path)

    def _risk_scores(self, features_df: pd.DataFrame) -> np.ndarray:
        """Give the risk score of each attempt. We flip the sign of the model's score, so higher is more unusual."""
        return -self.model.score_samples(features_df[MODEL_FEATURES])
