#!/usr/bin/env python

import argparse
import os
from collections import defaultdict, deque

import numpy as np
import pandas as pd


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

    return {
        "children": children,
        "parent": parent,
        "nodes": nodes,
        "level": level,
    }


ONTO = build_ontology_helpers()
nodes = ONTO["nodes"]
level = ONTO["level"]


# =========================================================
# 2) I/O helpers
# =========================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


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


def select_columns(df, columns_arg=None):
    if columns_arg is None or columns_arg.strip() == "":
        return list(df.columns)

    cols = [x.strip() for x in columns_arg.split(",") if x.strip()]
    missing = [c for c in cols if c not in df.columns]

    if missing:
        raise ValueError(f"Requested columns not found: {missing}")

    return cols


# =========================================================
# 3) Ontology label check
# =========================================================

def check_annotation_label_ontology_match(annotator_table, columns):
    """
    For each annotator column, calculate how many labels exactly match
    one ontology node at any level.

    A label is valid if:
        label in ontology nodes

    This checks parent-level and leaf-level nodes equally.
    """
    valid_nodes = set(nodes)

    summary_rows = []
    invalid_detail_rows = []

    for col in columns:
        s = annotator_table[col]

        non_na = s.dropna().astype(str)
        n_total = len(s)
        n_non_na = len(non_na)
        n_na = n_total - n_non_na

        valid_mask = non_na.isin(valid_nodes)

        n_valid = int(valid_mask.sum())
        n_invalid = int(n_non_na - n_valid)

        prop_valid_among_non_na = n_valid / n_non_na if n_non_na > 0 else np.nan
        prop_valid_among_all_cells = n_valid / n_total if n_total > 0 else np.nan

        invalid_counts = (
            non_na.loc[~valid_mask]
            .value_counts()
            .rename_axis("invalid_label")
            .reset_index(name="n_cells")
        )

        invalid_labels = invalid_counts["invalid_label"].tolist()

        summary_rows.append({
            "annotator": col,
            "n_cells_total": int(n_total),
            "n_na_labels": int(n_na),
            "n_non_na_labels": int(n_non_na),
            "n_valid_ontology_labels": int(n_valid),
            "n_invalid_labels": int(n_invalid),
            "prop_valid_among_non_na": prop_valid_among_non_na,
            "prop_valid_among_all_cells": prop_valid_among_all_cells,
            "n_unique_invalid_labels": int(len(invalid_labels)),
            "invalid_label_examples": "; ".join(invalid_labels[:30]),
        })

        for _, row in invalid_counts.iterrows():
            invalid_detail_rows.append({
                "annotator": col,
                "invalid_label": row["invalid_label"],
                "n_cells": int(row["n_cells"]),
                "prop_among_non_na_labels": (
                    row["n_cells"] / n_non_na if n_non_na > 0 else np.nan
                ),
                "prop_among_all_cells": (
                    row["n_cells"] / n_total if n_total > 0 else np.nan
                ),
            })

    summary_df = pd.DataFrame(summary_rows)
    invalid_detail_df = pd.DataFrame(invalid_detail_rows)

    return summary_df, invalid_detail_df


def write_ontology_label_check_report(
    summary_df,
    invalid_detail_df,
    output_txt_path,
    dataset_name,
):
    with open(output_txt_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"Ontology label matching report: {dataset_name}\n")
        f.write("=" * 80 + "\n\n")

        f.write("Definition:\n")
        f.write(
            "A label is valid if it exactly matches any ontology node at any level.\n"
        )
        f.write(
            "Examples of valid labels include Blood Cell, Leukocyte, B Cell,\n"
        )
        f.write(
            "CD4 T Cell (ab), Classical Monocyte, Plasmacytoid DC, etc.\n\n"
        )

        f.write("-" * 80 + "\n")
        f.write("Summary by annotator\n")
        f.write("-" * 80 + "\n\n")

        for _, row in summary_df.iterrows():
            annotator = row["annotator"]
            prop_non_na = row["prop_valid_among_non_na"]
            prop_all = row["prop_valid_among_all_cells"]

            f.write(f"Annotator: {annotator}\n")
            f.write(f"  Total cells: {int(row['n_cells_total'])}\n")
            f.write(f"  NA labels: {int(row['n_na_labels'])}\n")
            f.write(f"  Non-NA labels: {int(row['n_non_na_labels'])}\n")
            f.write(f"  Valid ontology labels: {int(row['n_valid_ontology_labels'])}\n")
            f.write(f"  Invalid labels: {int(row['n_invalid_labels'])}\n")

            if np.isfinite(prop_non_na):
                f.write(f"  Proportion valid among non-NA labels: {prop_non_na:.6f}\n")
            else:
                f.write("  Proportion valid among non-NA labels: NA\n")

            if np.isfinite(prop_all):
                f.write(f"  Proportion valid among all cells: {prop_all:.6f}\n")
            else:
                f.write("  Proportion valid among all cells: NA\n")

            if int(row["n_invalid_labels"]) == 0:
                f.write("  Status: PASS, all non-NA labels match ontology nodes.\n")
            else:
                f.write("  Status: WARNING, some labels do not match ontology nodes.\n")
                f.write(f"  Invalid label examples: {row['invalid_label_examples']}\n")

            f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("Invalid label details\n")
        f.write("=" * 80 + "\n\n")

        if invalid_detail_df.shape[0] == 0:
            f.write("No invalid labels found. All non-NA labels match ontology nodes.\n")
        else:
            for annotator in summary_df["annotator"].tolist():
                sub = invalid_detail_df[
                    invalid_detail_df["annotator"] == annotator
                ].copy()

                if sub.shape[0] == 0:
                    continue

                f.write("-" * 80 + "\n")
                f.write(f"Annotator: {annotator}\n")
                f.write("-" * 80 + "\n")

                sub = sub.sort_values("n_cells", ascending=False)

                for _, row in sub.iterrows():
                    f.write(
                        f"  {row['invalid_label']}: "
                        f"{int(row['n_cells'])} cells, "
                        f"prop among non-NA = {row['prop_among_non_na_labels']:.6f}, "
                        f"prop among all cells = {row['prop_among_all_cells']:.6f}\n"
                    )

                f.write("\n")


def run_one_dataset(dataset_name, input_path, args):
    dataset_outdir = os.path.join(args.outdir, dataset_name)
    ensure_dir(dataset_outdir)

    print(f"\n{'=' * 80}")
    print(f"Checking ontology labels for dataset: {dataset_name}")
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

    summary_df, invalid_detail_df = check_annotation_label_ontology_match(
        table,
        columns,
    )

    summary_path = os.path.join(
        dataset_outdir,
        f"{dataset_name}_annotation_label_ontology_check_summary.csv",
    )
    detail_path = os.path.join(
        dataset_outdir,
        f"{dataset_name}_annotation_label_ontology_check_invalid_labels.csv",
    )
    txt_path = os.path.join(
        dataset_outdir,
        f"{dataset_name}_annotation_label_ontology_check_report.txt",
    )

    summary_df.to_csv(summary_path, index=False)
    invalid_detail_df.to_csv(detail_path, index=False)

    write_ontology_label_check_report(
        summary_df=summary_df,
        invalid_detail_df=invalid_detail_df,
        output_txt_path=txt_path,
        dataset_name=dataset_name,
    )

    print(f"Saved summary CSV: {summary_path}")
    print(f"Saved invalid-label CSV: {detail_path}")
    print(f"Saved text report: {txt_path}")

    print("\nSummary:")
    print(summary_df.to_string(index=False))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Check whether annotation labels match the ontology tree."
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
        help="Comma-separated annotation columns to check. Default: all columns.",
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