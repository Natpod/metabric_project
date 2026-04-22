# Exploratory Data Analysis (EDA)

The EDA pipeline is implemented in `src/data/EDA.py` → `run_eda()`. It is driven by the same QC outputs that feed training (see [preprocessing.md](preprocessing.md)), applies a lightweight preprocessing pass to make the data suitable for visualization, and then runs a set of analyses for each of the three use cases. Results are collected into an HTML report (`eda_report.html`).

---

## 1. Entry Point

```
run_eda(
    id_column,
    non_gene_expression_columns,
    therapeutic_targets,
    diagnostic_targets,
    pronostic_targets,
    main_file_path,
    output_report_path,   # optional; if provided, HTML report is saved here
)
```

**Steps executed at startup:**

1. Load the raw CSV with `pd.read_csv`.
2. Call `run_qc(...)` (see [preprocessing.md § 0](preprocessing.md)) with `do_report=False` and `show_plots=False` to obtain `has_duplicates`, `boolean_cast_columns`, `outlier_columns`, and `high_cardinality_columns`.
3. Run `visualize_distributions` on the raw DataFrame to capture the unprocessed distribution of every column.
4. Instantiate `EDAHTMLReport` if an output path was given.

---

## 2. EDA Preprocessing (`run_preprocessing`)

Before any analysis, the dataset is passed through `run_preprocessing`, which applies the same transformations described in [preprocessing.md](preprocessing.md) sections 1–9, but **without a train/test split** — the whole analysis DataFrame is transformed in-place. This is used only for visualization purposes and does not affect training.

Key differences from training preprocessing:
- No outer/inner split — the whole filtered subset is transformed.
- One-hot encoding uses `pd.get_dummies` (drop_first=True) instead of `sklearn.OneHotEncoder`.
- Gene-expression columns are **not** separately coerced or batch-corrected.
- Scaling decisions follow the same IQR / KS / kurtosis rules as training.

---

## 3. Use Case Analyses

The EDA runs three analysis blocks, one per use case. Each block filters the raw data to a relevant subset, preprocesses it, and runs a common set of visualizations.

### Use Case 1 — Plausible Therapy

**Filter:** patients where `death_from_cancer ≠ 'Died of Other Causes'` OR `overall_survival = 1`. If the `death_from_cancer` column is absent, only `overall_survival = 1` patients are kept.

**Target construction:**
- Each therapy boolean column is normalized to `True`/`False`.
- A `breast_surgery` boolean flag is derived from the presence of `type_of_breast_surgery`.
- A readable **`multicategory_therapeutic_target`** string column is built by joining all active therapy and surgery labels with `|` (e.g. `"MASTECTOMY|CHEMOTHERAPY"`). Rows with no active labels are assigned `"none"`.

**Analyses run:**

| Analysis | Function | Target |
|----------|----------|--------|
| Bias analysis | `run_bias_analysis` | `multicategory_therapeutic_target` |
| Correlation heatmap — non-gene features | `run_correlation_analysis` | `multicategory_therapeutic_target` |
| Correlation heatmap — gene features | `run_correlation_analysis` | `multicategory_therapeutic_target` |
| Gene expression heatmap & clustermap | `run_gene_expression_heatmap_and_clustermap` | `multicategory_therapeutic_target` |
| Mutual information | `run_mutual_information_analysis` | each individual therapy target + multicategory target |
| PCA scatter | `run_pca_visualization` | `multicategory_therapeutic_target` |
| t-SNE scatter | `tsne_visualization` | `multicategory_therapeutic_target` |

---

### Use Case 2 — Plausible Survival Time (Prognostic)

**Filter:** `overall_survival = 0` (patients who died), ensuring the survival time target is not censored.

**Target:** first element of `pronostic_targets` (e.g. `overall_survival_months`).

**Analyses run:**

| Analysis | Function |
|----------|----------|
| Correlation heatmap — non-gene features | `run_correlation_analysis` |
| Correlation heatmap — gene features | `run_correlation_analysis` |
| Gene expression heatmap & clustermap | `run_gene_expression_heatmap_and_clustermap` |
| PCA scatter | `run_pca_visualization` |
| t-SNE scatter | `tsne_visualization` |

---

### Use Case 3 — Plausible Diagnosis

**Filter:** none (all rows used).

**Target:** first element of `diagnostic_targets` (e.g. `oncotree_code`).

**Analyses run:** same set as Use Case 2.

---

## 4. Analysis Functions

### 4.1 Distribution Visualization (`visualize_distributions`)

- **Numeric columns:** histogram with KDE overlay (`sns.histplot`) per column, excluding the ID column.
- **Categorical columns:** bar chart of value frequencies (`sns.countplot`) per column, excluding the ID column.

Runs on the **raw** DataFrame before any preprocessing.

---

### 4.2 Correlation Analysis (`run_correlation_analysis`)

1. Computes the Pearson correlation matrix for all numeric columns.
2. Removes columns with non-finite correlations.
3. Prints the correlation of each feature with the target column (sorted descending).
4. Renders a **hierarchically clustered heatmap** (`sns.clustermap`, `coolwarm` palette). Annotations are suppressed for matrices with more than 20 columns.

Run separately on the **non-gene** feature frame and the **gene-expression** feature frame.

---

### 4.3 Gene Expression Heatmap and Clustermap (`run_gene_expression_heatmap_and_clustermap`)

For datasets with many gene columns, only the **top 50 highest-variance genes** are selected (configurable via `max_features`).

**Heatmap:** samples × genes matrix transposed to genes × samples, rendered with `sns.heatmap` (`coolwarm`).

**Clustermap:**
- Hierarchical clustering on both genes and samples.
- **Sample annotation strips** on top of the clustermap:
  - **Target** strip: color-coded by the (optionally quantized) target value using the `Set2` palette.
  - **Cohort** strip: color-coded by `cohort` using the `Set1` palette (when the column is present).
- Numeric targets are binned into 4 quantile groups before coloring.

---

### 4.4 Bias Analysis (`run_bias_analysis`)

Default bias columns: `age_at_diagnosis`, `cohort`, `ethnicity`.

- `age_at_diagnosis` is converted to an ordinal `age_group` variable with bins `[0–39, 40–49, 50–59, 60–69, 70–79, 80+]`.

For each combination of (target, bias column):

1. **Cross-tabulation** — frequency table of bias group × target class, printed and added to the report.
2. **Stacked bar chart** — visual representation of the cross-tab (`sns.countplot`-equivalent stacked bar).
3. **Chi-squared test** (`scipy.stats.chi2_contingency`) — reports the test statistic and p-value to indicate statistical dependence between the bias variable and the target.

---

### 4.5 Mutual Information Analysis (`run_mutual_information_analysis`)

Computes **mutual information** between all numeric/boolean features and each target column. Top `N` scores (default 20) are reported.

- **Regression targets** (numeric, non-boolean): `sklearn.feature_selection.mutual_info_regression`.
- **Classification targets** (boolean or categorical): target is label-encoded with `pd.factorize`, then `sklearn.feature_selection.mutual_info_classif`.
- Features are prepared with `prepare_numeric_feature_frame`: nullable booleans cast to `Int64`, all `NaN` filled with 0.
- Columns that are part of the target encoding (resolved via `resolve_processed_non_gene_columns`) are excluded from the feature matrix.
- A horizontal bar chart of the top scores is added to the report.

---

### 4.6 PCA Visualization (`run_pca_visualization`)

- `sklearn.decomposition.PCA(n_components=2, random_state=42)` is fit on all numeric columns (NaN → 0).
- Scatter plot of PC1 vs PC2, colored by the target column using the `viridis` palette.
- Numeric targets are quantized into 4 bins; boolean targets are cast to string for legend clarity.
- Legend is placed outside the axes to avoid overlap.

---

### 4.7 t-SNE Visualization (`tsne_visualization`)

- `sklearn.manifold.TSNE(n_components=2, random_state=42)` fit on all numeric columns (NaN → 0).
- Same coloring and legend logic as PCA.
- Note: t-SNE is computationally expensive on large datasets.

---

## 5. HTML Report (`EDAHTMLReport`)

When `output_report_path` is provided, every visualization and log block is accumulated in an `EDAHTMLReport` instance. Figures are serialized to **base64 PNG** and embedded directly in the HTML (no external file dependencies). The report is written at the end of `run_eda`.

The report is organized as a flat sequence of `<h2>` sections, each containing:
- A `<pre>` block with any associated log text.
- An `<img>` tag with the embedded figure.

Default output: `reports/eda_report.html`.

---

## 6. Running the EDA

```bash
python -m src.data.EDA \
  --id_column patient_id \
  --main_file_path data/FCS_ml_test_input_data_rna_mutation.csv \
  --non_gene_expression_columns patient_id cohort age_at_diagnosis ... \
  --therapeutic_target_columns chemotherapy hormone_therapy ... \
  --diagnostic_target_columns oncotree_code \
  --pronostic_target_columns overall_survival_months \
  --output_report_path reports/eda_report.html
```
