"""Small test for the InvestigationLabeler, on a tiny hand-made table."""

from src.data_loader import DataLoader, load_config
from src.labels import InvestigationLabeler
from tests.test_data_loader import make_toy_attempts


def test_labeler_toy():
    """Two attempts that share an IP form one cluster, and a lonely attempt gets label 0."""
    labeler = InvestigationLabeler(load_config())
    attempts = DataLoader.add_day_column(make_toy_attempts())
    attempts.loc[1, "ip_id"] = "ip1"

    labeled = labeler.label(attempts)

    assert labeled["label"].tolist() == [1, 1, 0]
    assert labeled["cluster_id"].tolist()[:2] == [1, 1]
    assert labeled["cluster_id"].isna().tolist() == [False, False, True]

    clusters = labeler.describe_clusters(labeled)
    assert len(clusters) == 1
    assert clusters.loc[0, "attempts"] == 2
    assert clusters.loc[0, "distinct IPs"] == 1
    assert clusters.loc[0, "distinct cards"] == 2
