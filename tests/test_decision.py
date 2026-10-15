"""Small test for the DecisionPolicy, on a tiny hand-made table."""

import pandas as pd

from src.data_loader import load_config
from src.decision import DecisionPolicy


def test_decision_policy():
    """A soft-only alert with low risk and high value is released, and a hard rule gets its own action."""
    policy = DecisionPolicy(load_config())
    alert_columns = list(policy.hard_actions) + policy.soft_columns
    attempts = pd.DataFrame(False, index=range(5), columns=alert_columns)
    attempts["risk_band"] = "low"
    attempts["value_band"] = "high"
    attempts.loc[0, "R6"] = True           # soft only
    attempts.loc[1, "R1"] = True           # hard: step_up
    attempts.loc[2, ["R1", "R2"]] = True   # two hard rules: the stronger action (block) wins
    attempts.loc[3, ["R4", "R6"]] = True   # hard and soft: the hard rule decides, and R4 holds the payout
    # Row 4 has no alert.

    decisions = policy.decide(attempts)

    assert decisions["hybrid_action"].tolist() == ["allow_and_monitor", "step_up", "block", "manual_review", "allow"]
    assert decisions["rules_only_action"].tolist() == ["step_up", "step_up", "block", "manual_review", "allow"]
    assert decisions["alert_type"].tolist() == ["soft only", "hard", "hard", "hard", "none"]
    assert decisions["hold_payout"].tolist() == [False, False, False, True, False]
