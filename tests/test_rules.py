"""Small test for the RuleEngine, on a tiny hand-made table."""

from src.data_loader import DataLoader, load_config
from src.features import FeatureBuilder
from src.labels import InvestigationLabeler
from src.rules import RuleEngine
from tests.test_data_loader import make_toy_attempts


def test_rules_toy():
    """R1 fires on every attempt of a card with 4 attempts on one day, and on no attempt of a card with 3."""
    config = load_config()
    # Seven copies of one attempt on the same day: four with card_a and three with card_b.
    attempts = make_toy_attempts().iloc[[0] * 7].reset_index(drop=True)
    attempts["transaction_id"] = range(1, 8)
    attempts["credit_card_id"] = ["card_a"] * 4 + ["card_b"] * 3
    attempts = DataLoader.add_day_column(attempts)

    features = FeatureBuilder(config).fit(attempts).transform(attempts)
    engine = RuleEngine(config).fit(InvestigationLabeler(config).label(attempts))
    alerts = engine.apply(features)

    assert alerts.loc[alerts["credit_card_id"] == "card_a", "R1"].all()
    assert not alerts.loc[alerts["credit_card_id"] == "card_b", "R1"].any()
