"""Check that the training days and the test days look alike.

We split the data by time: we learn on the first days and test on the
later days. This only gives a fair test if the two parts hold the same
kind of traffic. If, for example, the prices on the test days are much
higher than on the training days, a price limit learned on the training
days no longer fits.

The check uses the Population Stability Index (PSI). The PSI is one
number per column that says how much the mix of values changed between
two tables. 0 means no change at all, and a higher number means a bigger
change.
"""

import numpy as np
import pandas as pd


class SplitChecker:
    """Compare the mix of values on the training days and on the test days.

    The checker only reads the data. It works on any column, so the
    notebook uses it twice: on the raw columns right after the split, and
    on the model features later.

    Args:
        config: The settings from config.yaml, as a dictionary.
    """

    def __init__(self, config: dict):
        """Keep the settings that this class needs."""
        settings = config["split_check"]
        self.psi_bins = settings["psi_bins"]
        self.psi_min_share = settings["psi_min_share"]
        self.small_shift_limit, self.large_shift_limit = settings["psi_shift_limits"]

    def psi_table(self, train_df: pd.DataFrame, test_df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """Give the PSI of each column, with a plain status.

        The status follows the usual rule of thumb: below the first limit
        in the config (0.1) the column is "stable", below the second limit
        (0.25) it has a "small shift", and from there on a "large shift".

        Known limit: the PSI says that a column changed, not why. Use
        group_shares() to see which values changed.

        Args:
            train_df: The attempts of the training days.
            test_df: The attempts of the test days.
            columns: The columns to compare, numbers or text.

        Returns:
            One row per column, with its kind (number or category), the
            number of groups, the PSI and the status, largest PSI first.
        """
        rows = []
        for column in columns:
            parts = self._psi_parts(train_df, test_df, column)
            psi = parts["psi part"].sum()
            rows.append({
                "column": column,
                "kind": "number" if pd.api.types.is_numeric_dtype(train_df[column]) else "category",
                "groups": len(parts),
                "psi": psi,
                "status": self._status(psi),
            })
        table = pd.DataFrame(rows).sort_values("psi", ascending=False, ignore_index=True)
        table["psi"] = table["psi"].round(3)
        return table

    def group_shares(self, train_df: pd.DataFrame, test_df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Show the share of each group of values in both parts of the split.

        A text column has one group per value. A number column is cut into
        value ranges (see _to_groups). For each group, the "psi part" is
        (test share - train share) * ln(test share / train share). The PSI
        of the column is the sum of these parts, so the groups with the
        largest part explain the change.

        A group that is missing in one part would give a division by zero.
        So every share is at least psi_min_share (config) in the formula.

        Args:
            train_df: The attempts of the training days.
            test_df: The attempts of the test days.
            column: The column to compare.

        Returns:
            One row per group, with the train share and the test share in
            percent and the psi part, largest psi part first.
        """
        parts = self._psi_parts(train_df, test_df, column)
        table = pd.DataFrame({
            "group": parts.index.astype(str),
            "train share (%)": (100 * parts["train"]).round(2).values,
            "test share (%)": (100 * parts["test"]).round(2).values,
            "psi part": parts["psi part"].round(4).values,
        })
        return table.sort_values("psi part", ascending=False, ignore_index=True)

    def _psi_parts(self, train_df: pd.DataFrame, test_df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Give the share of each group in both parts and its psi part, not rounded.

        psi_table() sums these exact parts. A sum of rounded parts would
        be a little off for a column with many groups.
        """
        train_groups, test_groups = self._to_groups(train_df[column], test_df[column])
        parts = pd.DataFrame({
            "train": train_groups.value_counts(normalize=True),
            "test": test_groups.value_counts(normalize=True),
        }).fillna(0.0)

        train_share = parts["train"].clip(lower=self.psi_min_share)
        test_share = parts["test"].clip(lower=self.psi_min_share)
        parts["psi part"] = (test_share - train_share) * np.log(test_share / train_share)
        return parts

    def _to_groups(self, train_values: pd.Series, test_values: pd.Series) -> tuple[pd.Series, pd.Series]:
        """Turn the values of one column into groups that we can count.

        Text values stay as they are. Numbers are cut into psi_bins (10)
        ranges that each hold about the same number of training attempts.
        The ranges come from the training days only, and the test days use
        the same ranges. A column with few distinct numbers gets fewer ranges.
        """
        if not pd.api.types.is_numeric_dtype(train_values):
            return train_values, test_values

        inner_points = np.linspace(0, 1, self.psi_bins + 1)[1:-1]
        edges = np.unique(np.quantile(train_values, inner_points))
        edges = [-np.inf, *edges, np.inf]
        return pd.cut(train_values, edges), pd.cut(test_values, edges)

    def _status(self, psi: float) -> str:
        """Turn a PSI into the words stable, small shift or large shift."""
        if psi < self.small_shift_limit:
            return "stable"
        if psi < self.large_shift_limit:
            return "small shift"
        return "large shift"
