#!/usr/bin/env python

import argparse
import os
from collections import defaultdict, deque

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize


# =========================================================
# 1) Ontology tree
# =========================================================

EDGES = [
    ("Blood Cell", "Platelet"),
    ("Blood Cell", "RBC"),
    ("Blood Cell", "HSC"),
    ("Blood Cell", "Doublet"),
    ("Blood Cell", "Leukocyte"),

    ("Leukocyte", "Lymphoid Cell"),
    ("Leukocyte", "Myeloid Cell"),

    ("Lymphoid Cell", "NK Cell"),
    ("Lymphoid Cell", "T Cell"),
    ("Lymphoid Cell", "B Cell"),

    ("T Cell", "CD4 T Cell (ab)"),
    ("T Cell", "CD8 T Cell (ab)"),
    ("T Cell", "ydT Cell"),
    ("T Cell", "MAIT Cell"),
    ("T Cell", "NKT Cell"),

    ("CD4 T Cell (ab)", "CD4 Naive / T Central Memory"),
    ("CD4 T Cell (ab)", "CD4 T Effector Memory"),
    ("CD4 T Cell (ab)", "Treg"),

    ("CD8 T Cell (ab)", "CD8 Naive / T Central Memory"),
    ("CD8 T Cell (ab)", "CD8 Cytotoxic / T Effector Memory"),

    ("B Cell", "Effector B"),
    ("B Cell", "Naive B Cell"),
    ("B Cell", "Memory B Cell"),

    ("Effector B", "Plasma Cell"),
    ("Effector B", "Plasmablast"),

    ("Myeloid Cell", "Monocyte"),
    ("Myeloid Cell", "Granulocyte"),
    ("Myeloid Cell", "DC"),

    ("Monocyte", "Classical Monocyte"),
    ("Monocyte", "Non-Classical Monocyte"),
    ("Monocyte", "Intermediate Monocyte"),

    ("Granulocyte", "Neutrophil"),
    ("Granulocyte", "Eosinophil"),
    ("Granulocyte", "Basophil"),
    ("Granulocyte", "Mast Cell"),

    ("DC", "Plasmacytoid DC"),
    ("DC", "Conventional DC 1"),
    ("DC", "Conventional DC 2"),
]

ROOT = "Blood Cell"


def build_ontology_helpers(edges=EDGES, root=ROOT):
    children = defaultdict(list)
    parent = {}
    nodes = set([root])

    for p, c in edges:
        children[p].append(c)
        parent[c] = p
        nodes.add(p)
        nodes.add(c)

    children = dict(children)
    nodes = sorted(nodes)

    level = {root: 0}
    q = deque([root])
    while q:
        u = q.popleft()
        for v in children.get(u, []):
            level[v] = level[u] + 1
            q.append(v)

    max_level = max(level.values())

    descendants = {n: set() for n in nodes}
    for n in nodes:
        dset = {n}
        dq = deque([n])
        while dq:
            u = dq.popleft()
            for v in children.get(u, []):
                if v not in dset:
                    dset.add(v)
                    dq.append(v)
        descendants[n] = dset

    def path_to_root(n):
        path = []
        cur = n
        while cur in parent:
            path.append(cur)
            cur = parent[cur]
        path.append(cur)
        path.reverse()
        return path

    return {
        "children": children,
        "parent": parent,
        "nodes": nodes,
        "level": level,
        "max_level": max_level,
        "descendants": descendants,
        "path_to_root": path_to_root,
    }


ONTO = build_ontology_helpers()
children = ONTO["children"]
nodes = ONTO["nodes"]
level = ONTO["level"]
MAX_LEVEL = ONTO["max_level"]
descendants = ONTO["descendants"]
path_to_root = ONTO["path_to_root"]


# =========================================================
# 2) I/O helpers
# =========================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_name(x):
    return str(x).replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")


def parse_input_tables(items):
    """
    Parse:
        Eui=/path/to/eui.csv BALI=/path/to/bali.csv
    """
    out = {}
    for item in items:
        if "=" not in item:
            raise ValueError(
                f"Invalid --input_tables item: {item}. "
                "Expected format DATASET=/path/to/table.csv"
            )

        name, path = item.split("=", 1)
        name = name.strip()
        path = path.strip()

        if not name or not path:
            raise ValueError(f"Invalid --input_tables item: {item}")

        out[name] = path

    return out


def infer_sep(path, user_sep=None):
    if user_sep is not None and user_sep != "auto":
        return user_sep
    if path.endswith(".tsv") or path.endswith(".txt"):
        return "\t"
    return ","


def read_annotation_table(path, sep="auto", index_col="0"):
    sep_use = infer_sep(path, sep)

    if index_col == "None":
        index_col = None
    else:
        index_col = int(index_col)

    df = pd.read_csv(path, sep=sep_use, index_col=index_col)
    df = df.dropna(axis=1, how="all")

    for col in df.columns:
        df[col] = df[col].where(df[col].isna(), df[col].astype(str))

    return df


def parse_comma_list(x):
    if x is None:
        return None
    x = str(x).strip()
    if x == "":
        return None
    return [v.strip() for v in x.split(",") if v.strip()]


def choose_targets(table, reference_cols, target_cols=None):
    if target_cols is not None and len(target_cols) > 0:
        missing = [c for c in target_cols if c not in table.columns]
        if missing:
            raise ValueError(f"Target columns not found in table: {missing}")
        return target_cols

    return [c for c in table.columns if c not in reference_cols]


# =========================================================
# 3) Cohen kappa
# =========================================================

def cohens_kappa_binary(y1, y2):
    y1 = np.asarray(y1, dtype=bool)
    y2 = np.asarray(y2, dtype=bool)

    if len(y1) != len(y2):
        raise ValueError("Binary vectors must have same length.")

    n11 = np.sum(y1 & y2)
    n10 = np.sum(y1 & ~y2)
    n01 = np.sum(~y1 & y2)
    n00 = np.sum(~y1 & ~y2)

    n = n11 + n10 + n01 + n00
    if n == 0:
        return np.nan

    po = (n11 + n00) / n

    p1_pos = (n11 + n10) / n
    p1_neg = 1 - p1_pos

    p2_pos = (n11 + n01) / n
    p2_neg = 1 - p2_pos

    pe = p1_pos * p2_pos + p1_neg * p2_neg

    if pe == 1:
        return np.nan

    return float((po - pe) / (1 - pe))


# =========================================================
# 4) Pairwise target-vs-reference table
# =========================================================

def build_pairwise_target_reference_table(
    annotator_table,
    target_col,
    reference_col,
    include_root=False,
    min_pos_per_side=1,
):
    """
    Compute node-level Cohen kappa for target A vs reference B.

    For each ontology node:
        binary target = cell belongs to node/subtree according to A
        binary ref    = cell belongs to node/subtree according to B
    """
    if target_col not in annotator_table.columns:
        raise ValueError(f"{target_col} not found in annotator_table.")

    if reference_col not in annotator_table.columns:
        raise ValueError(f"{reference_col} not found in annotator_table.")

    rows = []

    for node in nodes:
        if node == ROOT and not include_root:
            continue

        sub_nodes = descendants[node]

        target_binary = annotator_table[target_col].isin(sub_nodes).to_numpy()
        ref_binary = annotator_table[reference_col].isin(sub_nodes).to_numpy()

        target_pos = int(target_binary.sum())
        ref_pos = int(ref_binary.sum())

        if target_pos < min_pos_per_side or ref_pos < min_pos_per_side:
            kappa = np.nan
        else:
            kappa = cohens_kappa_binary(target_binary, ref_binary)

        both_pos = int((target_binary & ref_binary).sum())
        target_only = int((target_binary & ~ref_binary).sum())
        ref_only = int((~target_binary & ref_binary).sum())
        either_pos = int((target_binary | ref_binary).sum())

        if either_pos > 0:
            jaccard = both_pos / either_pos
        else:
            jaccard = np.nan

        if ref_pos > 0:
            target_ref_ratio = target_pos / ref_pos
        else:
            target_ref_ratio = np.nan

        path = path_to_root(node)

        row = {
            "Node": node,
            "Level": level[node],
            "target_col": target_col,
            "reference_col": reference_col,
            "Kappa": kappa,
            "target_positive_cells": target_pos,
            "reference_positive_cells": ref_pos,
            "both_positive_cells": both_pos,
            "target_only_cells": target_only,
            "reference_only_cells": ref_only,
            "either_positive_cells": either_pos,
            "jaccard_positive_set": jaccard,
            "target_to_reference_positive_ratio": target_ref_ratio,
        }

        for L in range(MAX_LEVEL + 1):
            row[f"Level{L}"] = path[L] if L < len(path) else np.nan

        rows.append(row)

    out = pd.DataFrame(rows)

    out = out.sort_values(
        ["Level"] + [f"Level{L}" for L in range(MAX_LEVEL + 1)]
    ).reset_index(drop=True)

    return out


def make_pairwise_tree_table(pairwise_table):
    out = pairwise_table.copy()
    out["use_node"] = out["Kappa"].notna()
    return out


def summarize_pairwise_target_reference(pairwise_table):
    valid = pairwise_table[pairwise_table["Kappa"].notna()].copy()

    target_col = pairwise_table["target_col"].iloc[0]
    reference_col = pairwise_table["reference_col"].iloc[0]

    if valid.shape[0] == 0:
        return pd.DataFrame([{
            "target_col": target_col,
            "reference_col": reference_col,
            "n_valid_nodes": 0,
            "mean_kappa": np.nan,
            "median_kappa": np.nan,
            "min_kappa": np.nan,
            "max_kappa": np.nan,
            "n_low_kappa_nodes_lt_0.2": 0,
            "n_negative_kappa_nodes": 0,
            "major_message": "No valid nodes for comparison.",
        }])

    low_nodes = valid[valid["Kappa"] < 0.2]
    neg_nodes = valid[valid["Kappa"] < 0]

    mean_k = valid["Kappa"].mean()
    median_k = valid["Kappa"].median()

    if mean_k >= 0.6:
        msg = "Overall agreement is strong across covered ontology nodes."
    elif mean_k >= 0.3:
        msg = "Overall agreement is moderate; several nodes may need review."
    else:
        msg = "Overall agreement is weak; annotation definitions or mapping likely need review."

    return pd.DataFrame([{
        "target_col": target_col,
        "reference_col": reference_col,
        "n_valid_nodes": int(valid.shape[0]),
        "mean_kappa": float(mean_k),
        "median_kappa": float(median_k),
        "min_kappa": float(valid["Kappa"].min()),
        "max_kappa": float(valid["Kappa"].max()),
        "n_low_kappa_nodes_lt_0.2": int(low_nodes.shape[0]),
        "n_negative_kappa_nodes": int(neg_nodes.shape[0]),
        "major_message": msg,
    }])


# =========================================================
# 5) Actionable feedback
# =========================================================

def make_actionable_node_feedback(
    pairwise_table,
    top_n=10,
    min_either_positive_cells=10,
    min_level=3,
):
    """
    Rank nodes for manual review.

    Score is heuristic:
        low kappa
        + large positive-count difference
        + many cells involved

    min_level can be used to avoid overly broad parent nodes.
    """
    df = pairwise_table.copy()

    df = df[
        (df["either_positive_cells"] >= min_either_positive_cells)
        & (df["Kappa"].notna())
        & (df["Level"] >= min_level)
    ].copy()

    if df.shape[0] == 0:
        return pd.DataFrame()

    df["abs_positive_difference"] = (
        df["target_positive_cells"] - df["reference_positive_cells"]
    ).abs()

    df["direction"] = np.where(
        df["target_positive_cells"] > df["reference_positive_cells"],
        "target_overcalls_vs_reference",
        np.where(
            df["target_positive_cells"] < df["reference_positive_cells"],
            "target_undercalls_vs_reference",
            "similar_positive_count",
        ),
    )

    df["review_priority_score"] = (
        (1 - df["Kappa"].clip(lower=-1, upper=1)) * 1.0
        + np.log1p(df["abs_positive_difference"]) * 0.15
        + np.log1p(df["either_positive_cells"]) * 0.05
    )

    out = df.sort_values(
        ["review_priority_score", "either_positive_cells"],
        ascending=[False, False],
    ).head(top_n)

    keep_cols = [
        "Node",
        "Level",
        "Kappa",
        "target_positive_cells",
        "reference_positive_cells",
        "both_positive_cells",
        "target_only_cells",
        "reference_only_cells",
        "either_positive_cells",
        "jaccard_positive_set",
        "target_to_reference_positive_ratio",
        "direction",
        "review_priority_score",
    ]

    return out[keep_cols].reset_index(drop=True)


# =========================================================
# 6) Plot ontology tree
# =========================================================

def shorten_label(name):
    replace_map = {
        "Blood Cell": "Blood\nCell",
        "Lymphoid Cell": "Lymphoid\nCell",
        "Myeloid Cell": "Myeloid\nCell",
        "Classical Monocyte": "Classical\nMono",
        "Intermediate Monocyte": "Intermediate\nMono",
        "Non-Classical Monocyte": "Non-Classical\nMono",
        "Plasmacytoid DC": "Plasmacytoid\nDC",
        "Conventional DC 1": "Conventional\nDC 1",
        "Conventional DC 2": "Conventional\nDC 2",
        "CD4 T Cell (ab)": "CD4 T Cell\n(ab)",
        "CD8 T Cell (ab)": "CD8 T Cell\n(ab)",
        "CD4 Naive / T Central Memory": "CD4 Naive /\nT Central Memory",
        "CD4 T Effector Memory": "CD4 T\nEffector Memory",
        "CD8 Naive / T Central Memory": "CD8 Naive /\nT Central Memory",
        "CD8 Cytotoxic / T Effector Memory": "CD8 Cytotoxic /\nT Effector Memory",
    }
    return replace_map.get(name, name)


def compute_tree_positions(
    root=ROOT,
    children_map=children,
    level_map=level,
    leaf_gap=1.45,
    level_gap=2.0,
):
    ordered_children = {
        p: sorted(children_map.get(p, []))
        for p in set(children_map) | {root}
    }

    pos_x = {}
    next_x = [0.0]

    def assign_x(u):
        kids = ordered_children.get(u, [])
        if not kids:
            pos_x[u] = next_x[0]
            next_x[0] += leaf_gap
        else:
            for v in kids:
                assign_x(v)
            xs = [pos_x[v] for v in kids]
            pos_x[u] = (min(xs) + max(xs)) / 2.0

    assign_x(root)

    return {
        n: (pos_x[n], -level_map[n] * level_gap)
        for n in pos_x
    }


def plot_ontology_kappa_tree(
    table,
    output_path,
    score_col="Kappa",
    title="Ontology-level kappa",
    cmap_name="viridis",
    vmin=-1,
    vmax=1,
    figsize=(22, 10),
    text_size=9,
    metric_label="Cohen kappa",
    dpi=300,
):
    score_by_node = dict(
        zip(
            table["Node"].astype(str),
            pd.to_numeric(table[score_col], errors="coerce"),
        )
    )

    if "use_node" in table.columns:
        use_by_node = dict(
            zip(table["Node"].astype(str), table["use_node"].astype(bool))
        )
    else:
        use_by_node = {n: True for n in nodes}

    pos = compute_tree_positions()
    cmap = get_cmap(cmap_name)
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=figsize)

    for p, c in EDGES:
        if p in pos and c in pos:
            x1, y1 = pos[p]
            x2, y2 = pos[c]
            ax.plot(
                [x1, x2],
                [y1, y2],
                color="#9a9a9a",
                lw=1.3,
                alpha=0.8,
                zorder=1,
            )

    for n, (x, y) in pos.items():
        s = score_by_node.get(n, np.nan)
        use_node = use_by_node.get(n, True)

        if (not use_node) or pd.isna(s):
            color = (0.83, 0.83, 0.83, 1.0)
        else:
            color = cmap(norm(s))

        ax.scatter(
            [x],
            [y],
            s=1800,
            c=[color],
            edgecolors="#555555",
            linewidths=1.2,
            zorder=2,
        )

    for n, (x, y) in pos.items():
        s = score_by_node.get(n, np.nan)
        use_node = use_by_node.get(n, True)

        label = shorten_label(n)

        if use_node and np.isfinite(s):
            label = f"{label}\n{s:.3f}"

        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=text_size,
            zorder=3,
            bbox=dict(
                boxstyle="round,pad=0.10",
                fc="white",
                ec="none",
                alpha=0.45,
            ),
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(metric_label)

    ax.set_title(title, fontsize=15)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# 7) Run one target annotator
# =========================================================

def run_one_target(
    annotator_table,
    dataset_name,
    target_col,
    reference_cols,
    dataset_outdir,
    args,
):
    target_dir = os.path.join(dataset_outdir, safe_name(target_col))
    ensure_dir(target_dir)

    print(f"\n--- Target annotator: {target_col} ---")
    print(f"Output dir: {target_dir}")

    pairwise_tables = {}
    summary_tables = {}
    feedback_tables = {}

    for ref_col in reference_cols:
        print(f"Comparing {target_col} vs {ref_col}")

        pair_tbl = build_pairwise_target_reference_table(
            annotator_table,
            target_col=target_col,
            reference_col=ref_col,
            include_root=args.include_root,
            min_pos_per_side=args.min_pos_per_side,
        )

        tree_tbl = make_pairwise_tree_table(pair_tbl)

        summary_tbl = summarize_pairwise_target_reference(pair_tbl)

        feedback_tbl = make_actionable_node_feedback(
            pair_tbl,
            top_n=args.top_n_feedback,
            min_either_positive_cells=args.min_either_positive_cells_for_feedback,
            min_level=args.min_level_feedback,
        )

        pairwise_tables[ref_col] = pair_tbl
        summary_tables[ref_col] = summary_tbl
        feedback_tables[ref_col] = feedback_tbl

        ref_safe = safe_name(ref_col)

        pair_path = os.path.join(
            target_dir,
            f"{safe_name(target_col)}_vs_{ref_safe}_node_table.csv",
        )
        summary_path = os.path.join(
            target_dir,
            f"{safe_name(target_col)}_vs_{ref_safe}_summary.csv",
        )
        feedback_path = os.path.join(
            target_dir,
            f"{safe_name(target_col)}_vs_{ref_safe}_feedback.csv",
        )
        fig_path = os.path.join(
            target_dir,
            f"{safe_name(target_col)}_vs_{ref_safe}_ontology_tree.{args.fig_format}",
        )

        pair_tbl.to_csv(pair_path, index=False)
        summary_tbl.to_csv(summary_path, index=False)
        feedback_tbl.to_csv(feedback_path, index=False)

        plot_ontology_kappa_tree(
            tree_tbl,
            output_path=fig_path,
            score_col="Kappa",
            title=f"{dataset_name}: {target_col} vs {ref_col}",
            metric_label="Cohen kappa",
            dpi=args.dpi,
        )

        print(f"Saved node table: {pair_path}")
        print(f"Saved summary: {summary_path}")
        print(f"Saved feedback: {feedback_path}")
        print(f"Saved tree figure: {fig_path}")

    combined_summary = pd.concat(
        list(summary_tables.values()),
        axis=0,
        ignore_index=True,
    )

    combined_feedback_parts = []
    for ref_col, tbl in feedback_tables.items():
        if tbl is not None and tbl.shape[0] > 0:
            tmp = tbl.copy()
            tmp.insert(0, "reference_col", ref_col)
            tmp.insert(0, "target_col", target_col)
            combined_feedback_parts.append(tmp)

    if len(combined_feedback_parts) > 0:
        combined_feedback = pd.concat(
            combined_feedback_parts,
            axis=0,
            ignore_index=True,
        )
    else:
        combined_feedback = pd.DataFrame()

    combined_summary_path = os.path.join(
        target_dir,
        f"{safe_name(target_col)}_combined_summary.csv",
    )
    combined_feedback_path = os.path.join(
        target_dir,
        f"{safe_name(target_col)}_combined_feedback.csv",
    )

    combined_summary.to_csv(combined_summary_path, index=False)
    combined_feedback.to_csv(combined_feedback_path, index=False)

    print(f"Saved combined summary: {combined_summary_path}")
    print(f"Saved combined feedback: {combined_feedback_path}")


# =========================================================
# 8) Run one dataset
# =========================================================

def run_one_dataset(dataset_name, input_path, args):
    dataset_outdir = os.path.join(args.outdir, dataset_name)
    ensure_dir(dataset_outdir)

    print(f"\n{'=' * 80}")
    print(f"Running individual reports for dataset: {dataset_name}")
    print(f"Input table: {input_path}")
    print(f"Output dir: {dataset_outdir}")
    print(f"{'=' * 80}")

    table = read_annotation_table(
        input_path,
        sep=args.sep,
        index_col=args.index_col,
    )

    reference_cols = parse_comma_list(args.reference_cols)
    if reference_cols is None or len(reference_cols) == 0:
        raise ValueError("--reference_cols must contain at least one column.")

    missing_refs = [c for c in reference_cols if c not in table.columns]
    if missing_refs:
        raise ValueError(f"Reference columns not found in table: {missing_refs}")

    target_cols = parse_comma_list(args.target_cols)
    target_cols = choose_targets(table, reference_cols, target_cols)

    if len(target_cols) == 0:
        raise ValueError("No target columns selected.")

    print(f"Annotation table shape: {table.shape}")
    print(f"Reference columns: {reference_cols}")
    print(f"Target columns: {target_cols}")

    for target_col in target_cols:
        run_one_target(
            annotator_table=table,
            dataset_name=dataset_name,
            target_col=target_col,
            reference_cols=reference_cols,
            dataset_outdir=dataset_outdir,
            args=args,
        )

    print(f"Finished dataset: {dataset_name}")


# =========================================================
# 9) CLI
# =========================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description="Run individual annotator ontology-level reports."
    )

    parser.add_argument(
        "--input_tables",
        nargs="+",
        required=True,
        help=(
            "Input tables as DATASET=/path/to/table.csv. "
            "Example: Eui=data/Eui.csv BALI=data/BALI.csv"
        ),
    )

    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory.",
    )

    parser.add_argument(
        "--reference_cols",
        required=True,
        help=(
            "Comma-separated reference annotation columns. "
            "Example: original,pred"
        ),
    )

    parser.add_argument(
        "--target_cols",
        default=None,
        help=(
            "Optional comma-separated target annotation columns. "
            "Default: all columns except reference columns."
        ),
    )

    parser.add_argument(
        "--sep",
        default="auto",
        help="Input separator. Use ',' or '\\t'. Default: auto.",
    )

    parser.add_argument(
        "--index_col",
        default="0",
        help="Index column for cell barcodes. Use 0 by default. Use None if no index column.",
    )

    parser.add_argument(
        "--include_root",
        action="store_true",
        help="Include ontology root node.",
    )

    parser.add_argument(
        "--min_pos_per_side",
        type=int,
        default=1,
        help=(
            "Minimum positive cells required on both target and reference sides "
            "to compute node-level Cohen kappa."
        ),
    )

    parser.add_argument(
        "--min_either_positive_cells_for_feedback",
        type=int,
        default=10,
        help="Minimum number of target-or-reference positive cells for feedback table.",
    )

    parser.add_argument(
        "--min_level_feedback",
        type=int,
        default=3,
        help=(
            "Minimum ontology level included in actionable feedback. "
            "Use 3 or 4 to avoid very broad parent nodes."
        ),
    )

    parser.add_argument(
        "--top_n_feedback",
        type=int,
        default=10,
        help="Number of feedback nodes to report per target-reference pair.",
    )

    parser.add_argument(
        "--fig_format",
        default="png",
        choices=["png", "pdf", "svg"],
        help="Figure format.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI.",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    ensure_dir(args.outdir)

    input_tables = parse_input_tables(args.input_tables)

    for dataset_name, input_path in input_tables.items():
        run_one_dataset(dataset_name, input_path, args)


if __name__ == "__main__":
    main()