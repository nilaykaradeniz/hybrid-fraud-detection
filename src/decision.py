"""Turn the rule alerts and the risk score into one action per attempt.

The actions, from light to strong, are: allow, allow_and_monitor,
step_up (an extra check for the customer), manual_review and block.
"Friction" means step_up, manual_review or block: the customer or an
analyst has extra work. The other two actions release the attempt.

We decide twice, so that the two setups can be compared:

- hybrid: a hard rule decides alone. For an attempt with only soft
  alerts, the risk band and the value band decide.
- rules only: the same, but every attempt with only soft alerts gets a
  step-up check. This is the baseline without the model.
"""

import numpy as np
import pandas as pd

FRICTION_ACTIONS = ["step_up", "manual_review", "block"]


class DecisionPolicy:
    """Choose the action of each attempt in the hybrid system and in the rules-only baseline.

    Args:
        config: The settings from config.yaml, as a dictionary.
    """

    def __init__(self, config: dict):
        """Keep the settings that this class needs, and read the action of each hard rule."""
        settings = config["decision"]
        self.action_order = settings["action_order"]
        self.soft_policy = settings["soft_policy"]
        self.rules_only_soft_action = settings["rules_only_soft_action"]

        self.hard_actions = {}
        self.hold_payout_columns = []
        self.soft_columns = []
        for rule, rule_settings in config["rules"].items():
            if rule_settings["group"] == "soft":
                self.soft_columns.append(rule)
            elif rule == "R13":
                self.hard_actions["R13_card_or_email"] = rule_settings["action_card_or_email"]
                self.hard_actions["R13_ip_or_property"] = rule_settings["action_ip_or_property"]
            else:
                self.hard_actions[rule] = rule_settings["action"]
            if rule_settings.get("hold_payout"):
                self.hold_payout_columns.append(rule)

    def decide(self, scored_df: pd.DataFrame) -> pd.DataFrame:
        """Add the action of each attempt for both setups.

        Args:
            scored_df: The attempts with their alert columns, risk_band and
                value_band (output of RiskModel.score()).

        Returns:
            A copy of scored_df with four new columns: "alert_type" (hard,
            soft only or none), "hybrid_action", "rules_only_action" and
            "hold_payout" (True if a rule that holds the payout to the
            property fired).
        """
        decisions = scored_df.copy()
        hard_action = self._strongest_hard_action(decisions)
        has_hard_alert = hard_action.notna()
        soft_only = decisions[self.soft_columns].any(axis=1) & ~has_hard_alert

        decisions["alert_type"] = "none"
        decisions.loc[soft_only, "alert_type"] = "soft only"
        decisions.loc[has_hard_alert, "alert_type"] = "hard"

        lightest_action = self.action_order[0]
        decisions["hybrid_action"] = hard_action.fillna(lightest_action)
        decisions.loc[soft_only, "hybrid_action"] = self._soft_action(decisions[soft_only])
        decisions["rules_only_action"] = hard_action.fillna(lightest_action)
        decisions.loc[soft_only, "rules_only_action"] = self.rules_only_soft_action
        decisions["hold_payout"] = decisions[self.hold_payout_columns].any(axis=1)
        return decisions

    def _strongest_hard_action(self, df: pd.DataFrame) -> pd.Series:
        """Give the strongest action of the hard rules that fired, or nothing if none fired."""
        strongest = np.full(len(df), -1)
        for column, action in self.hard_actions.items():
            strength = self.action_order.index(action)
            strongest = np.where(df[column], np.maximum(strongest, strength), strongest)
        actions = [self.action_order[strength] if strength >= 0 else None for strength in strongest]
        return pd.Series(actions, index=df.index, dtype=object)

    def _soft_action(self, soft_only_df: pd.DataFrame) -> list[str]:
        """Look up the action for each soft-only alert in the policy table of the config.

        The table has one row for high-value attempts and one for the
        others, and one column per risk band.
        """
        actions = []
        for value_band, risk_band in zip(soft_only_df["value_band"], soft_only_df["risk_band"]):
            policy_row = "high_value" if value_band == "high" else "other_value"
            actions.append(self.soft_policy[policy_row][risk_band])
        return actions
