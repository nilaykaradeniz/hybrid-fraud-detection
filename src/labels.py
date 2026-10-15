"""Build the investigation label: which attempts belong to an attack.

The data has no confirmed fraud label, so we build our own "truth" for
the evaluation. We link the attempts that share a card, an email or an
IP. A group of linked attempts is called a cluster. In this data normal
attempts never share one of these IDs, and every cluster with more than
one attempt was reviewed and confirmed as an attack. So the label is
simply: the attempt is in a cluster.

The label is never used to train the model.

Known limit: the label means "shares a card, email or IP with another
attempt". So any rule or feature that counts repeated cards, emails or
IPs agrees with the label by design. This raises the precision of those
rules and the ROC AUC of the model. Only the property rules R4, R9 and
R10 are independent tests.
"""

import networkx as nx
import pandas as pd

from src.data_loader import ID_NAMES

# Two attempts are linked if they share one of these IDs.
LINK_COLUMNS = ["credit_card_id", "email_id", "ip_id"]
AUTHORISED = "AUTHORISED"


class InvestigationLabeler:
    """Find the clusters of linked attempts and label their attempts as fraud.

    Args:
        config: The settings from config.yaml, as a dictionary.
    """

    def __init__(self, config: dict):
        """Keep the settings that this class needs."""
        self.min_cluster_size = config["labels"]["min_cluster_size"]
        self.auth_results = config["quality"]["auth_results"]

    def label(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add the cluster and the label of each attempt.

        The clusters are built from the rows of df alone. Call this on the
        whole month for the evaluation label, and on the training days for
        the label that the rule precision and the hotlist of rule R13 use.

        Args:
            df: The attempts to link, one row per payment attempt.

        Returns:
            A copy of df with two new columns: "cluster_id" (1 for the
            largest cluster, 2 for the next one, empty if the attempt is in
            no cluster) and "label" (1 if the attempt is in a cluster, else 0).
        """
        cluster_id = pd.Series(pd.NA, index=df.index, dtype="Int64")
        for number, attempts in enumerate(self._find_clusters(df), start=1):
            cluster_id.loc[attempts] = number

        labeled = df.copy()
        labeled["cluster_id"] = cluster_id
        labeled["label"] = cluster_id.notna().astype(int)
        return labeled

    def describe_clusters(self, labeled_df: pd.DataFrame) -> pd.DataFrame:
        """Describe each cluster with one row.

        Args:
            labeled_df: The output of label().

        Returns:
            One row per cluster, with its attempts, first and last day, the
            number of distinct cards, emails, IPs and properties, the
            attempts per auth_result, and the EUR value of its authorised
            attempts.
        """
        in_cluster = labeled_df[labeled_df["label"] == 1]
        per_cluster = in_cluster.groupby("cluster_id")
        table = per_cluster.agg(
            attempts=("transaction_id", "count"),
            first_day=("day", "min"),
            last_day=("day", "max"),
        ).rename(columns={"first_day": "first day", "last_day": "last day"})
        for id_col, name in ID_NAMES.items():
            table[f"distinct {name}"] = per_cluster[id_col].nunique()

        result_counts = pd.crosstab(in_cluster["cluster_id"], in_cluster["auth_result"])
        table = table.join(result_counts.reindex(columns=self.auth_results, fill_value=0))
        is_authorised = in_cluster["auth_result"] == AUTHORISED
        authorised_value = in_cluster[is_authorised].groupby("cluster_id")["price_euro"].sum()
        table["authorised value (EUR)"] = authorised_value.reindex(table.index, fill_value=0).round(0)
        return table.reset_index()

    def _find_clusters(self, df: pd.DataFrame) -> list[list]:
        """Give the attempts of each cluster, largest cluster first.

        We build a graph: each attempt is a point, and it is tied to its
        card, its email and its IP. Attempts that share any of these end up
        in the same connected part of the graph. A part with at least
        min_cluster_size attempts (2 in the config) is a cluster.
        """
        graph = nx.Graph()
        for col in LINK_COLUMNS:
            graph.add_edges_from(zip(df.index, col + ":" + df[col]))

        attempt_ids = set(df.index)
        clusters = []
        for linked_points in nx.connected_components(graph):
            attempts = sorted(point for point in linked_points if point in attempt_ids)
            if len(attempts) >= self.min_cluster_size:
                clusters.append(attempts)
        # Largest first, so that cluster 1 is the same cluster in every run.
        return sorted(clusters, key=lambda attempts: (-len(attempts), attempts[0]))
