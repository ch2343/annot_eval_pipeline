#!/usr/bin/env python

import argparse
import math
import os
from collections import defaultdict, deque
from itertools import combinations

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


def parse_input_tables(items):
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


def select_columns(df, columns_arg=None):
    if columns_arg is None or columns_arg.strip() == "":
        return list(df.columns)

    cols = [x.strip() for x in columns_arg.split(",") if x.strip()]
    missing = [c for c in cols if c not in df.columns]

    if missing:
        raise ValueError(f"Requested columns not found: {missing}")

    return cols


# =========================================================
# 3) Kappa functions
# =========================================================

def coverage_required(n_annotators, coverage_fraction=0.5, min_required=2):
    """
    At-least coverage rule.

    With 4 annotators and coverage_fraction=0.5:
        required = 2.

    min_required=2 prevents nodes covered by only one annotator.
    """
    return max(min_required, int(np.ceil(n_annotators * coverage_fraction)))


def fleiss_kappa_from_counts(counts):
    counts = np.asarray(counts, dtype=float)

    if counts.ndim != 2:
        return np.nan
    if counts.shape[0] == 0:
        return np.nan
    if counts.shape[1] < 2:
        return np.nan

    row_sums = counts.sum(axis=1)

    if not np.allclose(row_sums, row_sums[0]):
        return np.nan

    n_raters = row_sums[0]
    if n_raters <= 1:
        return np.nan

    n_items = counts.shape[0]

    p_i = np.sum(counts * (counts - 1.0), axis=1) / (
        n_raters * (n_raters - 1.0)
    )
    p_bar = np.mean(p_i)

    p_j = np.sum(counts, axis=0) / (n_items * n_raters)
    p_e = np.sum(p_j ** 2)

    denom = 1.0 - p_e

    if denom == 0:
        return np.nan

    return float((p_bar - p_e) / denom)


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


def normal_sf(z):
    if not np.isfinite(z):
        return np.nan
    return 0.5 * math.erfc(z / math.sqrt(2))


def get_node_binary_matrix(annotator_table, node, columns):
    sub_nodes = descendants[node]

    bool_mat = []
    for col in columns:
        bool_mat.append(annotator_table[col].isin(sub_nodes).to_numpy())

    return np.vstack(bool_mat)


def compute_node_kappa(
    annotator_table,
    node,
    columns,
    method="fleiss",
):
    columns = list(columns)

    if len(columns) < 2:
        return np.nan

    bool_mat = get_node_binary_matrix(
        annotator_table,
        node=node,
        columns=columns,
    )

    if bool_mat.sum() == 0:
        return np.nan

    if method == "fleiss":
        n_raters = bool_mat.shape[0]
        n_pos = bool_mat.sum(axis=0)
        n_neg = n_raters - n_pos
        counts = np.column_stack([n_pos, n_neg])
        return fleiss_kappa_from_counts(counts)

    kappas = []
    for i, j in combinations(range(bool_mat.shape[0]), 2):
        k = cohens_kappa_binary(bool_mat[i], bool_mat[j])
        if np.isfinite(k):
            kappas.append(k)

    if len(kappas) == 0:
        return np.nan

    if method == "pairwise_mean":
        return float(np.mean(kappas))
    if method == "pairwise_median":
        return float(np.median(kappas))

    raise ValueError("method must be fleiss, pairwise_mean, or pairwise_median.")


# =========================================================
# 4) Full node-level consistency table
# =========================================================

def build_overall_consistency_table(
    annotator_table,
    columns,
    method="fleiss",
    include_root=False,
    coverage_fraction=0.5,
    min_required_annotators=2,
    min_pos_per_annotator=1,
):
    n_annotators = len(columns)

    required_pos_annotators = coverage_required(
        n_annotators,
        coverage_fraction=coverage_fraction,
        min_required=min_required_annotators,
    )

    rows = []

    for node in nodes:
        if node == ROOT and not include_root:
            continue

        bool_mat = get_node_binary_matrix(
            annotator_table,
            node=node,
            columns=columns,
        )

        pos_per_annotator = bool_mat.sum(axis=1)

        n_pos_annotators = int(
            np.sum(pos_per_annotator >= min_pos_per_annotator)
        )

        use_node = n_pos_annotators >= required_pos_annotators

        if use_node:
            kappa = compute_node_kappa(
                annotator_table,
                node=node,
                columns=columns,
                method=method,
            )
        else:
            kappa = np.nan

        path = path_to_root(node)

        row = {
            "Node": node,
            "Level": level[node],
            "Kappa": kappa,
            "Method": method,
            "use_node": use_node,
            "n_annotators": n_annotators,
            "coverage_fraction": coverage_fraction,
            "required_pos_annotators": required_pos_annotators,
            "n_pos_annotators": n_pos_annotators,
            "pos_annotators": ",".join(
                [
                    col
                    for col, npos in zip(columns, pos_per_annotator)
                    if npos >= min_pos_per_annotator
                ]
            ),
            "total_positive_calls": int(bool_mat.sum()),
            "cells_positive_by_any": int(bool_mat.any(axis=0).sum()),
            "cells_positive_by_all": int(bool_mat.all(axis=0).sum()),
        }

        for L in range(MAX_LEVEL + 1):
            row[f"Level{L}"] = path[L] if L < len(path) else np.nan

        rows.append(row)

    out = pd.DataFrame(rows)

    out = out.sort_values(
        ["Level"] + [f"Level{L}" for L in range(MAX_LEVEL + 1)]
    ).reset_index(drop=True)

    return out


# =========================================================
# 5) Leave-one-out outlier analysis
# =========================================================

def one_sided_high_zscores(row_values):
    x = pd.Series(row_values, dtype=float)

    out_z = pd.Series(np.nan, index=x.index, dtype=float)
    out_p = pd.Series(np.nan, index=x.index, dtype=float)

    for col in x.index:
        xi = x.loc[col]
        others = x.drop(col).dropna()

        if not np.isfinite(xi):
            continue
        if len(others) < 2:
            continue

        mu = others.mean()
        sd = others.std(ddof=1)

        if not np.isfinite(sd) or sd <= 0:
            continue

        z = (xi - mu) / sd
        p = normal_sf(z)

        out_z.loc[col] = z
        out_p.loc[col] = p

    return out_z, out_p


def benjamini_hochberg_matrix(p_mat):
    pvals = p_mat.to_numpy().ravel()
    valid = np.isfinite(pvals)
    p_valid = pvals[valid]

    flat_adj = np.full_like(pvals, np.nan, dtype=float)

    if len(p_valid) == 0:
        return pd.DataFrame(
            flat_adj.reshape(p_mat.shape),
            index=p_mat.index,
            columns=p_mat.columns,
        )

    order = np.argsort(p_valid)
    ranked = p_valid[order]
    m = len(ranked)

    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.minimum(q, 1.0)

    q_back = np.empty(m, dtype=float)
    q_back[order] = q

    flat_adj[valid] = q_back

    return pd.DataFrame(
        flat_adj.reshape(p_mat.shape),
        index=p_mat.index,
        columns=p_mat.columns,
    )


def run_leave_one_out_outlier_analysis(
    annotator_table,
    full_consistency_table,
    columns,
    method="fleiss",
    p_threshold=0.05,
    min_delta_kappa=0.03,
    z_threshold=None,
    use_fdr=False,
):
    full_tbl = full_consistency_table.copy().set_index("Node")

    used_nodes = full_tbl.index[full_tbl["use_node"] == True].tolist()
    full_kappa = full_tbl.loc[used_nodes, "Kappa"]

    loo_kappa = pd.DataFrame(index=used_nodes, columns=columns, dtype=float)

    for left_out in columns:
        kept_cols = [c for c in columns if c != left_out]

        for node in used_nodes:
            loo_kappa.loc[node, left_out] = compute_node_kappa(
                annotator_table,
                node=node,
                columns=kept_cols,
                method=method,
            )

    loo_delta = loo_kappa.sub(full_kappa, axis=0)

    z_mat = pd.DataFrame(index=used_nodes, columns=columns, dtype=float)
    p_mat = pd.DataFrame(index=used_nodes, columns=columns, dtype=float)

    for node in used_nodes:
        z, p = one_sided_high_zscores(loo_kappa.loc[node, :])
        z_mat.loc[node, :] = z
        p_mat.loc[node, :] = p

    p_adj_mat = benjamini_hochberg_matrix(p_mat) if use_fdr else p_mat.copy()

    flag_mat = pd.DataFrame(False, index=used_nodes, columns=columns)

    for node in used_nodes:
        for col in columns:
            p_value_to_use = p_adj_mat.loc[node, col]
            delta = loo_delta.loc[node, col]
            z = z_mat.loc[node, col]

            flag = True

            if not (np.isfinite(p_value_to_use) and p_value_to_use < p_threshold):
                flag = False

            if not (np.isfinite(delta) and delta >= min_delta_kappa):
                flag = False

            if z_threshold is not None:
                if not (np.isfinite(z) and z >= z_threshold):
                    flag = False

            flag_mat.loc[node, col] = flag

    node_rows = []

    for node in used_nodes:
        for col in columns:
            node_rows.append({
                "Node": node,
                "annotator_left_out": col,
                "full_kappa": full_kappa.loc[node],
                "loo_kappa": loo_kappa.loc[node, col],
                "delta_kappa_LOO_minus_full": loo_delta.loc[node, col],
                "z_high": z_mat.loc[node, col],
                "one_sided_p_high": p_mat.loc[node, col],
                "one_sided_p_adj_high": p_adj_mat.loc[node, col] if use_fdr else np.nan,
                "is_outlier": bool(flag_mat.loc[node, col]),
            })

    node_outlier_table = pd.DataFrame(node_rows)

    summary_rows = []

    for col in columns:
        d = loo_delta[col]
        z = z_mat[col]
        flags = flag_mat[col]
        flagged_nodes = flag_mat.index[flags].tolist()

        summary_rows.append({
            "annotator": col,
            "n_valid_nodes": int(d.notna().sum()),
            "mean_delta_kappa_LOO_minus_full": float(d.mean(skipna=True)),
            "median_delta_kappa_LOO_minus_full": float(d.median(skipna=True)),
            "pct_nodes_improved_after_leaving_out": float((d > 0).mean(skipna=True)),
            "n_outlier_nodes": int(flags.sum()),
            "pct_outlier_nodes": float(flags.mean()),
            "mean_z_high": float(z.mean(skipna=True)),
            "max_z_high": float(z.max(skipna=True)),
            "outlier_nodes": "; ".join(flagged_nodes[:50]),
        })

    outlier_summary = pd.DataFrame(summary_rows).sort_values(
        ["n_outlier_nodes", "mean_delta_kappa_LOO_minus_full"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return {
        "loo_kappa_matrix": loo_kappa,
        "loo_delta_matrix": loo_delta,
        "loo_z_matrix": z_mat,
        "loo_p_matrix": p_mat,
        "loo_p_adj_matrix": p_adj_mat,
        "outlier_flag_matrix": flag_mat,
        "node_outlier_table": node_outlier_table,
        "outlier_summary": outlier_summary,
    }


# =========================================================
# 6) Remove node-specific outliers and recompute kappa
# =========================================================

def recompute_kappa_after_node_specific_outlier_removal(
    annotator_table,
    full_consistency_table,
    outlier_flag_matrix,
    columns,
    method="fleiss",
    min_remaining_annotators=2,
):
    full_tbl = full_consistency_table.copy().set_index("Node")

    rows = []

    for node in nodes:
        if node not in full_tbl.index:
            continue

        original_kappa = full_tbl.loc[node, "Kappa"]
        use_node = bool(full_tbl.loc[node, "use_node"])

        if not use_node:
            removed_annotators = []
            kept_annotators = columns
            recomputed_kappa = np.nan
            n_removed = 0
            n_remaining = len(columns)
        else:
            if node in outlier_flag_matrix.index:
                flags = outlier_flag_matrix.loc[node]
            else:
                flags = pd.Series(False, index=columns)

            removed_annotators = [
                col for col in columns if bool(flags.get(col, False))
            ]

            kept_annotators = [
                col for col in columns if col not in removed_annotators
            ]

            n_removed = len(removed_annotators)
            n_remaining = len(kept_annotators)

            if n_remaining >= min_remaining_annotators:
                recomputed_kappa = compute_node_kappa(
                    annotator_table,
                    node=node,
                    columns=kept_annotators,
                    method=method,
                )
            else:
                recomputed_kappa = np.nan

        path = path_to_root(node)

        row = {
            "Node": node,
            "Level": level[node],
            "Kappa_original": original_kappa,
            "Kappa_outliers_removed": recomputed_kappa,
            "Delta_removed_minus_original": (
                recomputed_kappa - original_kappa
                if np.isfinite(recomputed_kappa) and np.isfinite(original_kappa)
                else np.nan
            ),
            "use_node": use_node,
            "n_removed_annotators": n_removed,
            "removed_annotators": ",".join(removed_annotators),
            "n_remaining_annotators": n_remaining,
            "remaining_annotators": ",".join(kept_annotators),
            "Method": method,
        }

        for L in range(MAX_LEVEL + 1):
            row[f"Level{L}"] = path[L] if L < len(path) else np.nan

        rows.append(row)

    out = pd.DataFrame(rows)

    out = out.sort_values(
        ["Level"] + [f"Level{L}" for L in range(MAX_LEVEL + 1)]
    ).reset_index(drop=True)

    return out


# =========================================================
# 7) Plotting functions
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
    metric_label="Kappa",
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


def plot_leave_one_out_dotplot_with_outliers(
    loo_kappa_matrix,
    outlier_flag_matrix,
    output_path,
    title="Leave-one-out kappa with outlier flags",
    max_nodes=60,
    sort_by="outlier_then_range",
    figsize=(10, 12),
    vmin=-1,
    vmax=1,
    cmap_name="viridis",
    size_scale=280,
    min_dot_size=20,
    x_spacing=0.68,
    dpi=300,
):
    mat = loo_kappa_matrix.copy()
    flags = outlier_flag_matrix.copy()

    mat = mat.dropna(how="all")
    flags = flags.loc[mat.index, mat.columns]

    if sort_by == "outlier_then_range":
        n_flags = flags.sum(axis=1)
        score_range = mat.max(axis=1, skipna=True) - mat.min(axis=1, skipna=True)
        sort_df = pd.DataFrame({
            "n_flags": n_flags,
            "range": score_range,
        })
        ordered_nodes = sort_df.sort_values(
            ["n_flags", "range"],
            ascending=[False, False],
        ).index
        mat = mat.loc[ordered_nodes]
        flags = flags.loc[ordered_nodes]

    elif sort_by == "range":
        score = mat.max(axis=1, skipna=True) - mat.min(axis=1, skipna=True)
        mat = mat.loc[score.sort_values(ascending=False).index]
        flags = flags.loc[mat.index]

    elif sort_by == "mean":
        score = mat.mean(axis=1, skipna=True)
        mat = mat.loc[score.sort_values(ascending=False).index]
        flags = flags.loc[mat.index]

    else:
        raise ValueError("Invalid sort_by.")

    if max_nodes is not None and mat.shape[0] > max_nodes:
        mat = mat.iloc[:max_nodes, :]
        flags = flags.loc[mat.index, mat.columns]

    mat.index = mat.index.astype(str)
    mat.index.name = "Node"

    long_df = mat.reset_index().melt(
        id_vars="Node",
        var_name="Annotator_left_out",
        value_name="LOO_Kappa",
    )

    flag_long = flags.reset_index().melt(
        id_vars="index",
        var_name="Annotator_left_out",
        value_name="is_outlier",
    ).rename(columns={"index": "Node"})

    long_df = long_df.merge(
        flag_long,
        on=["Node", "Annotator_left_out"],
        how="left",
    )

    long_df = long_df.dropna(subset=["LOO_Kappa"]).copy()

    node_order = mat.index.tolist()
    annotator_order = mat.columns.tolist()

    long_df["Node"] = pd.Categorical(
        long_df["Node"],
        categories=node_order,
        ordered=True,
    )

    long_df["Annotator_left_out"] = pd.Categorical(
        long_df["Annotator_left_out"],
        categories=annotator_order,
        ordered=True,
    )

    x = long_df["Annotator_left_out"].cat.codes.to_numpy() * x_spacing
    y = long_df["Node"].cat.codes.to_numpy()
    sizes = np.abs(long_df["LOO_Kappa"].to_numpy()) * size_scale + min_dot_size

    fig, ax = plt.subplots(figsize=figsize)

    scatter = ax.scatter(
        x,
        y,
        c=long_df["LOO_Kappa"].to_numpy(),
        s=sizes,
        cmap=cmap_name,
        vmin=vmin,
        vmax=vmax,
        edgecolor="#555555",
        linewidth=0.4,
        alpha=0.95,
    )

    out_df = long_df[long_df["is_outlier"] == True].copy()

    if out_df.shape[0] > 0:
        xo = out_df["Annotator_left_out"].cat.codes.to_numpy() * x_spacing
        yo = out_df["Node"].cat.codes.to_numpy()
        so = np.abs(out_df["LOO_Kappa"].to_numpy()) * size_scale + min_dot_size + 120

        ax.scatter(
            xo,
            yo,
            s=so,
            facecolors="none",
            edgecolors="red",
            linewidths=1.8,
            alpha=0.95,
            label="Outlier",
        )
        ax.legend(loc="best", frameon=True)

    ax.set_xticks(np.arange(len(annotator_order)) * x_spacing)
    ax.set_xticklabels(annotator_order, rotation=30, ha="right")

    ax.set_yticks(np.arange(len(node_order)))
    ax.set_yticklabels(node_order)

    ax.set_ylim(len(node_order) - 0.5, -0.5)

    if len(annotator_order) > 1:
        ax.set_xlim(-0.25, (len(annotator_order) - 1) * x_spacing + 0.25)

    ax.set_xlabel("Annotator left out")
    ax.set_ylabel("Ontology node")
    ax.set_title(title)

    ax.grid(axis="y", linestyle=":", alpha=0.25)
    ax.set_axisbelow(True)

    cbar = plt.colorbar(scatter, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("LOO kappa")

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# 8) Main dataset run
# =========================================================

def run_one_dataset(dataset_name, input_path, args):
    dataset_outdir = os.path.join(args.outdir, dataset_name)
    ensure_dir(dataset_outdir)

    print(f"\n{'=' * 80}")
    print(f"Running overall consistency analysis for dataset: {dataset_name}")
    print(f"Input table: {input_path}")
    print(f"Output dir: {dataset_outdir}")
    print(f"{'=' * 80}")

    table = read_annotation_table(
        input_path,
        sep=args.sep,
        index_col=args.index_col,
    )

    columns = select_columns(table, args.columns)

    print(f"Annotation table shape: {table.shape}")
    print(f"Annotator columns used: {columns}")

    # -----------------------------
    # Full node kappa
    # -----------------------------
    full_table = build_overall_consistency_table(
        table,
        columns=columns,
        method=args.method,
        include_root=args.include_root,
        coverage_fraction=args.coverage_fraction,
        min_required_annotators=args.min_required_annotators,
        min_pos_per_annotator=args.min_pos_per_annotator,
    )

    full_table_path = os.path.join(
        dataset_outdir,
        f"{dataset_name}_full_node_kappa.csv",
    )

    full_table.to_csv(full_table_path, index=False)
    print(f"Saved full node kappa table: {full_table_path}")

    full_fig_path = os.path.join(
        dataset_outdir,
        f"{dataset_name}_ontology_tree_full.{args.fig_format}",
    )

    plot_ontology_kappa_tree(
        full_table,
        output_path=full_fig_path,
        score_col="Kappa",
        title=(
            f"{dataset_name}: full node-level {args.method} kappa\n"
            f"Node included if >= {args.coverage_fraction:.2f} annotators cover it"
        ),
        metric_label=f"{args.method} kappa",
        dpi=args.dpi,
    )

    print(f"Saved full ontology tree: {full_fig_path}")

    # -----------------------------
    # LOO outlier analysis
    # -----------------------------
    loo_results = run_leave_one_out_outlier_analysis(
        table,
        full_consistency_table=full_table,
        columns=columns,
        method=args.method,
        p_threshold=args.p_threshold,
        min_delta_kappa=args.min_delta_kappa,
        z_threshold=args.z_threshold,
        use_fdr=args.use_fdr,
    )

    outlier_table_path = os.path.join(
        dataset_outdir,
        f"{dataset_name}_loo_outlier_table.csv",
    )

    loo_results["node_outlier_table"].to_csv(outlier_table_path, index=False)
    print(f"Saved LOO outlier table: {outlier_table_path}")

    outlier_summary_path = os.path.join(
        dataset_outdir,
        f"{dataset_name}_loo_outlier_summary.csv",
    )

    loo_results["outlier_summary"].to_csv(outlier_summary_path, index=False)
    print(f"Saved LOO outlier summary: {outlier_summary_path}")

    dotplot_path = os.path.join(
        dataset_outdir,
        f"{dataset_name}_loo_dotplot_outliers.{args.fig_format}",
    )

    plot_leave_one_out_dotplot_with_outliers(
        loo_results["loo_kappa_matrix"],
        loo_results["outlier_flag_matrix"],
        output_path=dotplot_path,
        title=(
            f"{dataset_name}: leave-one-out {args.method} kappa\n"
            f"Red circles: p<{args.p_threshold}, delta>={args.min_delta_kappa}"
        ),
        max_nodes=args.max_nodes_dotplot,
        dpi=args.dpi,
    )

    print(f"Saved LOO dot plot: {dotplot_path}")

    # -----------------------------
    # Remove outliers and recompute
    # -----------------------------
    removed_table = recompute_kappa_after_node_specific_outlier_removal(
        table,
        full_consistency_table=full_table,
        outlier_flag_matrix=loo_results["outlier_flag_matrix"],
        columns=columns,
        method=args.method,
        min_remaining_annotators=args.min_remaining_annotators,
    )

    removed_table_path = os.path.join(
        dataset_outdir,
        f"{dataset_name}_removed_recomputed_kappa.csv",
    )

    removed_table.to_csv(removed_table_path, index=False)
    print(f"Saved removed/recomputed kappa table: {removed_table_path}")

    removed_fig_path = os.path.join(
        dataset_outdir,
        f"{dataset_name}_ontology_tree_outliers_removed.{args.fig_format}",
    )

    plot_ontology_kappa_tree(
        removed_table.rename(columns={"Kappa_outliers_removed": "Kappa"}),
        output_path=removed_fig_path,
        score_col="Kappa",
        title=(
            f"{dataset_name}: node-level {args.method} kappa "
            f"after removing flagged outliers"
        ),
        metric_label=f"{args.method} kappa after removal",
        dpi=args.dpi,
    )

    print(f"Saved removed/recomputed ontology tree: {removed_fig_path}")
    print(f"Finished dataset: {dataset_name}")


# =========================================================
# 9) CLI
# =========================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description="Run ontology-based overall annotation consistency analysis."
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
        "--columns",
        default=None,
        help="Comma-separated annotator columns to use. Default: all columns.",
    )

    parser.add_argument(
        "--method",
        default="fleiss",
        choices=["fleiss", "pairwise_mean", "pairwise_median"],
        help="Kappa method for multi-annotator consistency.",
    )

    parser.add_argument(
        "--include_root",
        action="store_true",
        help="Include ontology root node.",
    )

    parser.add_argument(
        "--coverage_fraction",
        type=float,
        default=0.5,
        help="Minimum fraction of annotators that must cover a node.",
    )

    parser.add_argument(
        "--min_required_annotators",
        type=int,
        default=2,
        help="Minimum number of annotators required to cover a node.",
    )

    parser.add_argument(
        "--min_pos_per_annotator",
        type=int,
        default=1,
        help="An annotator covers a node if it assigns at least this many cells to the node/subtree.",
    )

    parser.add_argument(
        "--p_threshold",
        type=float,
        default=0.05,
        help="One-sided p-value threshold for LOO outlier detection.",
    )

    parser.add_argument(
        "--min_delta_kappa",
        type=float,
        default=0.03,
        help="Minimum LOO kappa increase required for outlier flagging.",
    )

    parser.add_argument(
        "--z_threshold",
        type=float,
        default=None,
        help="Optional z-score threshold for LOO outlier detection.",
    )

    parser.add_argument(
        "--use_fdr",
        action="store_true",
        help="Use Benjamini-Hochberg FDR-adjusted p-values for outlier flags.",
    )

    parser.add_argument(
        "--min_remaining_annotators",
        type=int,
        default=2,
        help="Minimum annotators remaining after removing node-specific outliers.",
    )

    parser.add_argument(
        "--max_nodes_dotplot",
        type=int,
        default=60,
        help="Maximum number of nodes shown in LOO dot plot.",
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