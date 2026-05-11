# Annotation Evaluation Pipeline

This repository provides an ontology-based pipeline for evaluating cell-type annotations from multiple annotation sources.

The pipeline includes three scripts:

```text
scripts/check_ontology_labels.py
scripts/run_overall_consistency.py
scripts/run_individual_reports.py
```

Example annotation tables are included:

```text
Eui_ontology_annotator_table.csv
BALI_ontology_annotator_table.csv
Eui_raw_annotator_table.csv
BALI_raw_annotator_table.csv
```

Use the `*_ontology_annotator_table.csv` files for the analysis scripts.

---

## Installation

Clone this repository:

```bash
git clone https://github.com/ch2343/annot_eval_pipeline.git
cd annot_eval_pipeline
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Required packages:

```text
numpy
pandas
matplotlib
```

---

## Input Format

The input table should be an ontology-mapped annotation table.

Rows are cells, and columns are annotators or annotation methods.

Example:

```csv
cell_barcode,original,pred,anno,celltypist
cell_1,B Cell,Naive B Cell,B Cell,B Cell
cell_2,Classical Monocyte,Classical Monocyte,Monocyte,Classical Monocyte
cell_3,CD8 T Cell (ab),CD8 Cytotoxic / T Effector Memory,CD8 T Cell (ab),CD8 T Cell (ab)
```

The first column should be the cell barcode or cell ID. The remaining columns should be annotation labels.

The labels should already match the ontology tree. Examples of valid labels include:

```text
Blood Cell
Leukocyte
Lymphoid Cell
Myeloid Cell
B Cell
Naive B Cell
Memory B Cell
T Cell
CD4 T Cell (ab)
CD4 T Effector Memory
CD8 T Cell (ab)
CD8 Cytotoxic / T Effector Memory
NK Cell
Monocyte
Classical Monocyte
Non-Classical Monocyte
DC
Plasmacytoid DC
Platelet
RBC
```

Raw labels such as `CD4.CM`, `B_naive`, `cMono`, or `CD8TEM` should be mapped to ontology labels before running the analysis.

---

## Optional: Create Ontology-Mapped Tables

If you start from raw annotation labels, map each label to the ontology tree first.

Example:

```python
import pandas as pd
import numpy as np

raw_table = pd.read_csv("Eui_raw_annotator_table.csv", index_col=0)

pred_to_onto = {
    "CD4.CM": "CD4 Naive / T Central Memory",
    "CD4.EM": "CD4 T Effector Memory",
    "CD4.Naive": "CD4 Naive / T Central Memory",
    "CD8.EM": "CD8 Cytotoxic / T Effector Memory",
    "CD8.TE": "CD8 Cytotoxic / T Effector Memory",
    "CD8.Naive": "CD8 Naive / T Central Memory",
    "B_naive": "Naive B Cell",
    "B_non-switched_memory": "Memory B Cell",
    "B_switched_memory": "Memory B Cell",
    "CD14_mono": "Classical Monocyte",
    "CD16_mono": "Non-Classical Monocyte",
    "pDC": "Plasmacytoid DC",
    "DC1": "Conventional DC 1",
    "DC2": "Conventional DC 2",
    "Platelets": "Platelet",
    "RBC": "RBC",
}

def map_with_dict_only(s, mapping):
    return s.map(lambda x: mapping.get(x, x) if pd.notna(x) else x)

mapped_table = raw_table.copy()
mapped_table["pred"] = map_with_dict_only(mapped_table["pred"], pred_to_onto)

mapped_table.to_csv("Eui_ontology_annotator_table.csv")
```

After mapping, run the ontology label check script to confirm that all labels match the ontology.

---

## 1. Check Ontology Label Matching

This step checks whether labels in each annotation column match any node in the ontology tree.

Run:

```bash
python scripts/check_ontology_labels.py \
  --input_tables \
    Eui=Eui_ontology_annotator_table.csv \
    BALI=BALI_ontology_annotator_table.csv \
  --outdir results/ontology_check \
  --index_col 0
```

Output example:

```text
results/ontology_check/Eui/
├── Eui_annotation_label_ontology_check_summary.csv
├── Eui_annotation_label_ontology_check_invalid_labels.csv
└── Eui_annotation_label_ontology_check_report.txt
```

The text report summarizes, for each annotation column:

```text
number of cells
number of non-missing labels
number of ontology-matched labels
proportion of valid ontology labels
invalid labels, if any
```

If the valid proportion is not 100%, check the invalid labels and update the mapping dictionary before running the κ analysis.

---

## 2. Overall Consistency Analysis

This script computes ontology-level multi-annotator consistency using Fleiss κ.

Run:

```bash
python scripts/run_overall_consistency.py \
  --input_tables \
    Eui=Eui_ontology_annotator_table.csv \
    BALI=BALI_ontology_annotator_table.csv \
  --outdir results/overall \
  --index_col 0 \
  --coverage_fraction 0.5 \
  --min_required_annotators 2 \
  --p_threshold 0.05 \
  --min_delta_kappa 0.03 \
  --fig_format png
```

For a stricter final run:

```bash
python scripts/run_overall_consistency.py \
  --input_tables \
    Eui=Eui_ontology_annotator_table.csv \
    BALI=BALI_ontology_annotator_table.csv \
  --outdir results/overall_strict \
  --index_col 0 \
  --coverage_fraction 0.5 \
  --min_required_annotators 2 \
  --p_threshold 0.05 \
  --min_delta_kappa 0.05 \
  --use_fdr \
  --fig_format png
```

### Important Parameters

```text
--coverage_fraction 0.5
```

A node is evaluated if at least 50% of annotators have at least one cell assigned to this node or its descendants.

```text
--min_required_annotators 2
```

At least two annotators must cover the node.

```text
--p_threshold 0.05
```

One-sided p-value threshold for leave-one-out outlier detection.

```text
--min_delta_kappa 0.03
```

Minimum increase in κ required to flag an annotator as a node-level outlier.

```text
--use_fdr
```

Use FDR-adjusted p-values for outlier detection.

---

## Overall Consistency Outputs

For each dataset, the script creates:

```text
results/overall/Eui/
├── Eui_full_node_kappa.csv
├── Eui_loo_outlier_table.csv
├── Eui_loo_outlier_summary.csv
├── Eui_removed_recomputed_kappa.csv
├── Eui_ontology_tree_full.png
├── Eui_loo_dotplot_outliers.png
└── Eui_ontology_tree_outliers_removed.png
```

### `*_full_node_kappa.csv`

This table contains the full ontology-node consistency results.

Important columns:

```text
Node
Level
Kappa
use_node
n_annotators
required_pos_annotators
n_pos_annotators
pos_annotators
cells_positive_by_any
cells_positive_by_all
```

Interpretation:

- `Kappa`: Fleiss κ across all annotators.
- `use_node`: whether the node passed the coverage rule.
- `n_pos_annotators`: how many annotators assigned at least one cell to this node or its descendants.
- `cells_positive_by_any`: number of cells annotated as this node by at least one annotator.
- `cells_positive_by_all`: number of cells annotated as this node by all annotators.

### `*_ontology_tree_full.png`

Ontology tree colored by full multi-annotator κ.

- High κ means strong consistency.
- Low κ means weak consistency.
- Grey nodes are not evaluated because they did not pass the coverage rule.

### `*_loo_outlier_table.csv`

Detailed leave-one-out outlier table.

Each row is one:

```text
ontology node × annotator left out
```

Important columns:

```text
Node
annotator_left_out
full_kappa
loo_kappa
delta_kappa_LOO_minus_full
z_high
one_sided_p_high
one_sided_p_adj_high
is_outlier
```

Interpretation:

If removing one annotator increases κ for a node, that annotator may be inconsistent with the others for that cell type.

### `*_loo_outlier_summary.csv`

Annotator-level summary of leave-one-out results.

Important columns:

```text
annotator
mean_delta_kappa_LOO_minus_full
median_delta_kappa_LOO_minus_full
pct_nodes_improved_after_leaving_out
n_outlier_nodes
pct_outlier_nodes
outlier_nodes
```

Interpretation:

- Positive `mean_delta_kappa_LOO_minus_full`: removing this annotator improves agreement on average.
- Negative `mean_delta_kappa_LOO_minus_full`: removing this annotator decreases agreement on average.
- Higher `n_outlier_nodes`: this annotator is flagged in more cell-type nodes.

### `*_loo_dotplot_outliers.png`

Dot plot of leave-one-out κ.

- Rows: ontology nodes.
- Columns: annotator left out.
- Dot color: κ after removing that annotator.
- Red circle: flagged node-level outlier.

### `*_removed_recomputed_kappa.csv`

κ recomputed after removing flagged node-specific outlier annotators.

Important columns:

```text
Node
Kappa_original
Kappa_outliers_removed
Delta_removed_minus_original
removed_annotators
remaining_annotators
```

### `*_ontology_tree_outliers_removed.png`

Ontology tree colored by recomputed κ after removing flagged outliers.

---

## 3. Individual Annotator Reports

This script compares each non-reference annotator against one or more reference annotations.

For example, if the table has:

```text
original
pred
anno
celltypist
```

and the references are:

```text
original,pred
```

then the script generates reports for:

```text
anno
celltypist
```

Run:

```bash
python scripts/run_individual_reports.py \
  --input_tables \
    Eui=Eui_ontology_annotator_table.csv \
    BALI=BALI_ontology_annotator_table.csv \
  --outdir results/individual \
  --reference_cols original,pred \
  --index_col 0 \
  --min_pos_per_side 1 \
  --min_either_positive_cells_for_feedback 10 \
  --min_level_feedback 3 \
  --top_n_feedback 10 \
  --fig_format png
```

Run only one target annotator:

```bash
python scripts/run_individual_reports.py \
  --input_tables \
    Eui=Eui_ontology_annotator_table.csv \
  --outdir results/individual_celltypist_only \
  --reference_cols original,pred \
  --target_cols celltypist \
  --index_col 0 \
  --fig_format png
```

---

## Individual Report Outputs

Example:

```text
results/individual/Eui/celltypist/
├── celltypist_vs_original_node_table.csv
├── celltypist_vs_original_summary.csv
├── celltypist_vs_original_feedback.csv
├── celltypist_vs_original_ontology_tree.png
├── celltypist_vs_pred_node_table.csv
├── celltypist_vs_pred_summary.csv
├── celltypist_vs_pred_feedback.csv
├── celltypist_vs_pred_ontology_tree.png
├── celltypist_combined_summary.csv
└── celltypist_combined_feedback.csv
```

### `target_vs_reference_node_table.csv`

Node-level Cohen κ between the target annotator and one reference annotator.

Important columns:

```text
Node
Kappa
target_positive_cells
reference_positive_cells
both_positive_cells
target_only_cells
reference_only_cells
either_positive_cells
jaccard_positive_set
target_to_reference_positive_ratio
```

### `target_vs_reference_summary.csv`

Summary of target-reference agreement across ontology nodes.

Important columns:

```text
mean_kappa
median_kappa
n_low_kappa_nodes_lt_0.2
n_negative_kappa_nodes
major_message
```

### `target_vs_reference_feedback.csv`

Actionable cell-type nodes for manual review.

Important columns:

```text
Node
Kappa
target_positive_cells
reference_positive_cells
target_only_cells
reference_only_cells
jaccard_positive_set
direction
review_priority_score
```

`direction` tells whether the target annotator overcalls or undercalls a node relative to the reference.

`review_priority_score` is a heuristic ranking score:

```text
review_priority_score =
    (1 - Kappa)
  + 0.15 * log(1 + absolute positive-count difference)
  + 0.05 * log(1 + either-positive cell count)
```

This is not a p-value. It is only used to prioritize nodes for manual review.

### `target_vs_reference_ontology_tree.png`

Ontology tree colored by Cohen κ between the target annotator and reference annotator.

---

## Example Full Workflow

```bash
# Step 1: check ontology label matching
python scripts/check_ontology_labels.py \
  --input_tables \
    Eui=Eui_ontology_annotator_table.csv \
    BALI=BALI_ontology_annotator_table.csv \
  --outdir results/ontology_check \
  --index_col 0

# Step 2: run overall multi-annotator consistency
python scripts/run_overall_consistency.py \
  --input_tables \
    Eui=Eui_ontology_annotator_table.csv \
    BALI=BALI_ontology_annotator_table.csv \
  --outdir results/overall \
  --index_col 0 \
  --coverage_fraction 0.5 \
  --min_required_annotators 2 \
  --p_threshold 0.05 \
  --min_delta_kappa 0.03 \
  --fig_format png

# Step 3: run individual annotator reports
python scripts/run_individual_reports.py \
  --input_tables \
    Eui=Eui_ontology_annotator_table.csv \
    BALI=BALI_ontology_annotator_table.csv \
  --outdir results/individual \
  --reference_cols original,pred \
  --index_col 0 \
  --min_pos_per_side 1 \
  --min_either_positive_cells_for_feedback 10 \
  --min_level_feedback 3 \
  --top_n_feedback 10 \
  --fig_format png
```

---

## Notes

1. Always run `check_ontology_labels.py` before the κ analysis.
2. The input to the analysis scripts should be ontology-mapped annotation tables.
3. Broad parent nodes such as `Leukocyte` or `Lymphoid Cell` may be less actionable than specific cell-type nodes.
4. For final reporting, consider using stricter outlier settings:

```bash
--min_delta_kappa 0.05 --use_fdr
```

5. For individual feedback, use `--min_level_feedback 3` or `--min_level_feedback 4` to focus on more specific cell-type nodes.
