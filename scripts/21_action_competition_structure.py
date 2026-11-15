"""Which actions compete, and what the 2-d belief map adds beyond scalar summaries.

Response to the reviewer remark on the behavioural-manifold figure ("what does this 2-d
projection establish beyond argmax + max prob + entropy?"). Two checks on the faithful
extracted labels (`outputs/behavior_manifold_extracted.parquet`, which carries the
CLR-PCA coordinates):

  1. Redundancy: held-out R^2 of the PCA coordinates predicted from
     (argmax one-hot, max prob, normalised entropy) with gradient boosting, 5-fold
     grouped by trajectory.
  2. What is left: the identity of the runner-up action, predicted from the scalars
     versus from the PCA coordinates (same CV), plus the direct competition table
     P(runner-up | top action), overall and on low-confidence prefixes.

Usage (repo root):
    interp/.venv/bin/python scripts/21_action_competition_structure.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # noqa: E402  (repo convention)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import accuracy_score, r2_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="outputs/behavior_manifold_extracted.parquet")
    parser.add_argument("--report", default="outputs/action_competition_structure_report.md")
    parser.add_argument("--table_dir", default="outputs/action_competition_structure")
    parser.add_argument("--low_confidence", type=float, default=0.6,
                        help="top-probability threshold defining the low-confidence subset")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = pd.read_parquet(REPO_ROOT / args.input)
    q = table[[f"q_{a}" for a in ACTIONS]].to_numpy(dtype=float)
    order = np.argsort(-q, axis=1)
    table["runner_up"] = [ACTIONS[i] for i in order[:, 1]]

    top_onehot = pd.get_dummies(table["top_action"])[ACTIONS].to_numpy(dtype=float)
    scalars = np.column_stack([top_onehot, table["top_prob"], table["normalized_entropy"]])
    pcs = table[["beh_pca1", "beh_pca2"]].to_numpy(dtype=float)
    groups = table["source_file"].to_numpy()
    splitter = GroupKFold(n_splits=args.folds)

    r2 = {}
    for j, name in enumerate(["PC1", "PC2"]):
        y = pcs[:, j]
        pred = np.zeros_like(y)
        for train, test in splitter.split(scalars, y, groups):
            model = HistGradientBoostingRegressor(max_iter=300, random_state=args.seed)
            pred[test] = model.fit(scalars[train], y[train]).predict(scalars[test])
        r2[name] = float(r2_score(y, pred))

    y = table["runner_up"].to_numpy()

    def cv_accuracy(features: np.ndarray) -> float:
        pred = np.empty(len(y), dtype=object)
        for train, test in splitter.split(features, y, groups):
            model = HistGradientBoostingClassifier(max_iter=200, random_state=args.seed)
            pred[test] = model.fit(features[train], y[train]).predict(features[test])
        return float(accuracy_score(y, pred))

    majority = table.groupby("top_action")["runner_up"].transform(
        lambda s: s.value_counts().index[0])
    accuracy = {
        "majority runner-up per argmax": float((majority == table["runner_up"]).mean()),
        "argmax + max prob + entropy": cv_accuracy(scalars),
        "argmax + PC1 + PC2": cv_accuracy(np.column_stack([top_onehot, pcs])),
        "argmax + max prob + entropy + PC1 + PC2": cv_accuracy(np.column_stack([scalars, pcs])),
    }

    def competition(frame: pd.DataFrame) -> pd.DataFrame:
        return (pd.crosstab(frame["top_action"], frame["runner_up"], normalize="index")
                .reindex(index=ACTIONS, columns=ACTIONS).fillna(0.0))

    low = table[table["top_prob"] < args.low_confidence]
    all_table = competition(table)
    low_table = competition(low)

    table_dir = REPO_ROOT / args.table_dir
    table_dir.mkdir(parents=True, exist_ok=True)
    all_table.round(4).to_csv(table_dir / "runner_up_given_top_all.csv")
    low_table.round(4).to_csv(table_dir / "runner_up_given_top_low_confidence.csv")

    lines = [
        "# Action competition structure and 2-d map redundancy (script 21)",
        "",
        f"- Input: `{args.input}`; {len(table)} prefixes, {table['source_file'].nunique()} "
        f"trajectories; {args.folds}-fold CV grouped by trajectory; seed {args.seed}.",
        "",
        "## 1. Redundancy of the CLR-PCA coordinates with (argmax, max prob, entropy)",
        "",
        "Held-out R^2 of each PCA coordinate predicted from the three scalar summaries:",
        "",
        *[f"- {k}: {v:.3f}" for k, v in r2.items()],
        "",
        "## 2. Runner-up identity (held-out accuracy)",
        "",
        *[f"- {k}: {v:.3f}" for k, v in accuracy.items()],
        "",
        "## 3. P(runner-up | top action), all prefixes",
        "",
        all_table.round(3).to_markdown(),
        "",
        f"## 4. P(runner-up | top action), prefixes with top probability < "
        f"{args.low_confidence} (n={len(low)})",
        "",
        low_table.round(3).to_markdown(),
        "",
        f"- Tables written to `{args.table_dir}/`.",
    ]
    (REPO_ROOT / args.report).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
