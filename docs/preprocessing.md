# Pre-Training Preprocessing

This document describes the preprocessing steps applied to the dataset before model training. All transformations are fit exclusively on the **outer-train split** of each Leave-One-Cohort-Out fold and then applied to the held-out test split, preventing data leakage.

---

## 0. Quality Control (QC)

Before any transformation takes place, a QC pass (`run_qc` in `src/data/quality/QC.py`) is executed on the raw dataset. Its purpose is to **profile the data and produce the configuration flags** that drive the preprocessing steps below. It can optionally generate an HTML report (`qc_report.html`).

### 0.1 Duplicate Check

`df.duplicated(subset=patient_id)` counts rows that share the same patient/sample identifier.

- **Output:** `has_duplicates` (bool) — passed to the training pipeline to decide whether to drop duplicates in step 1.
- A bar chart of duplicate vs. unique entries is included in the HTML report.

### 0.2 Missing Value Profiling

`df.isnull().sum()` counts missing values per column.

- Columns with at least one missing value are listed in the preprocessing recommendation block of the HTML report.
- The result informs the imputation strategy applied in step 6.

### 0.3 Outlier Detection

For every **numeric** column (`float` / `int` dtypes), the IQR rule is applied on the full raw dataset to count outliers:

$$\text{outlier if } x < Q_1 - 1.5 \cdot \text{IQR} \quad \text{or} \quad x > Q_3 + 1.5 \cdot \text{IQR}$$

- **Output:** `outlier_columns` — list of numeric columns that contain at least one outlier; passed to `clip_numeric_outliers` in step 4.
- A box plot per column is included in the HTML report.

### 0.4 Cardinality and Boolean-Cast Detection

For every **categorical** (`object` dtype) column, the number of distinct non-null values (after lowercasing and stripping whitespace) is computed.

- **Boolean casting:** if a column has exactly **2 distinct values** and one of them belongs to the set `{"positive", "+", "yes", "1"}`, it is flagged for boolean casting.
  - **Output:** `boolean_cast_columns` — list of such columns; passed to step 2.
- **High-cardinality detection:** columns with **more than 20 distinct values** are considered high-cardinality.
  - **Output:** `high_cardinality_columns` — list of such columns; passed to the rare-grouping step (step 5).
- Count plots per column are included in the HTML report.

### 0.5 Value Frequency Profiling

Value counts for each categorical column are printed and, when reporting is enabled, added to the HTML report. This is a diagnostic-only step that does not produce any output flag.

### 0.6 Preprocessing Recommendation

At the end of the QC run, a plain-text summary is printed (and added to the HTML report) with actionable recommendations:

- Whether duplicate rows should be removed.
- Which columns require missing-value imputation.
- Which numeric columns contain outliers and may need robust treatment.
- Which categorical columns have high cardinality and may need frequency/target encoding.
- Which columns were automatically identified as boolean.

### QC Outputs Summary

| Output | Type | Used in step |
|--------|------|-------------|
| `has_duplicates` | `bool` | 1 – Duplicate Removal |
| `boolean_cast_columns` | `list[str]` | 2 – Boolean Casting |
| `outlier_columns` | `list[str]` | 4 – Outlier Treatment |
| `high_cardinality_columns` | `list[str]` | 5 – High-Cardinality Grouping |

---

## 1. Duplicate Removal

When the dataset is flagged as containing duplicates, rows with the same patient/sample identifier (`par`) are dropped from both train and test splits before any further processing.

---

## 2. Boolean Column Casting

Columns listed in `boolean_cast_columns` are normalized to proper boolean values:

- String values are lowercased and stripped of whitespace.
- `"positive"` and `"+"` map to `True`; everything else maps to `False`.
- Missing values are preserved as `pd.NA` (nullable boolean dtype).

Additionally, any remaining columns containing only `"True"` / `"False"` string values (detected by `cast_true_false_categorical_columns`) are cast to boolean automatically.

---

## 3. Gene Expression Treatment

Gene-expression columns are parsed into plain `float64` values through the following steps:

1. Apply `pd.to_numeric(..., errors='coerce')`.
2. Fallback: extract the first numeric token with a regex that handles scientific notation.
3. Any remaining `NaN` values are filled with `0.0`.

After coercion, gene-expression values with absolute magnitude ≤ 1.5 are zeroed out (treating them as noise below the expression threshold).

---

## 4. Outlier Treatment

Numeric outliers are clipped using the **IQR method**, fitted on the training split only:

| Bound | Formula |
|-------|---------|
| Lower | Q1 − 1.5 × IQR |
| Upper | Q3 + 1.5 × IQR |

- Values below the lower bound are replaced with **Q1**.
- Values above the upper bound are replaced with **Q3**.
- The same bounds derived from training data are applied to the test split.
- Boolean and non-numeric columns are skipped.

---

## 5. High-Cardinality Categorical Variables

Columns listed in `high_cardinality_columns` (e.g. `geolocation_id`) are handled by **rare-category grouping**, fitted on the training split:

1. Compute the frequency of each category in the training split.
2. Calculate a rarity threshold: `median(frequencies) − 2 × variance(frequencies)`.
3. Categories with frequency below the threshold are grouped into a single `"rare"` label.
4. A new grouped column is created (`geo_grouped` for `geolocation_id`, or `{col}_grouped` for others).
5. The mapping is applied to the test split: any test category not seen in the frequent set is also mapped to `"rare"`.

---

## 6. Null (Missing Value) Imputation

Imputation strategies are fitted on the training split and applied identically to the test split.

### Numeric columns
- Each numeric column is filled with the **median** of the training split.

### Categorical columns
- Missing values are filled with the string `"unknown"`.
- All categorical string values are additionally lowercased and stripped of whitespace (via `normalize_categorical_values`).

---

## 7. Categorical Encoding

After imputation, all categorical columns (object, string, and boolean dtypes) are encoded using **One-Hot Encoding** (`sklearn.preprocessing.OneHotEncoder`):

- `handle_unknown='ignore'` ensures unseen categories in the test split produce all-zero rows instead of raising errors.
- The encoder is fitted on training categorical columns only.
- The fitted encoder is serialized as an MLflow artifact (`preprocessing_one_hot_encoder`) for use during inference.

---

## 8. Batch Correction (Gene Expression)

To remove cohort-level technical variation from gene-expression features, **mean-centering per cohort** is applied to the training split:

1. Compute the global mean of each gene-expression column across all training samples.
2. For each cohort present in the training split, compute the cohort mean and subtract the difference (cohort mean − global mean) from that cohort's samples.
3. The held-out test split is **not** corrected — it remains as received to simulate a realistic deployment scenario.
4. Cohort offsets are stored in the MLflow preprocessing artifacts for reproducibility.

---

## 9. Numeric Scaling

Each numeric feature (after encoding) is scaled using a **data-driven strategy** selected per column, based on statistics derived from the training split:

| Decision | Condition | Scaler Applied |
|----------|-----------|----------------|
| `GeneExpression` | Column is a gene-expression feature | No scaling (already coerced and batch-corrected) |
| `NoScaling` | Zero variance or empty column | Identity (no transformation) |
| `Log1pTransform` | ≥ 50 % zero values and non-negative | `log1p(x)` applied element-wise |
| `MinMaxScaler` | All values in [0, 1] | Min-max normalization to [0, 1] |
| `StandardScaler` | KS p-value ≥ 0.05, KS statistic ≤ 0.08, and \|kurtosis\| ≤ 1.0 | Z-score standardization |
| `RobustScaler` | KS p-value < 0.05, or KS statistic > 0.08, or \|kurtosis\| > 3.0 | Median/IQR normalization |
| Default | None of the above | `StandardScaler` |

All scaler parameters (mean, std, min, max, median, IQR) are computed from the training split and recorded in the `numeric_transformers` MLflow artifact.

---

## 10. Target Encoding

Classification targets are label-encoded using `sklearn.preprocessing.LabelEncoder`:

- For multi-target problems (`DataFrame` targets), each column is encoded independently.
- The encoder is fitted on training labels only; unseen labels in the held-out cohort raise a `ValueError` to prevent silent mislabeling.
- Encoded class mappings are stored in the `target_encoders` MLflow artifact.
- For regression tasks, targets are passed through without modification.

---

## 11. Feature Selection (Variance Threshold)

Inside each inner cross-validation fold during model selection, a `VarianceThreshold` step (default threshold `0.0`) is prepended to the estimator as a `Pipeline`. This removes zero-variance features derived purely from the fold's training data, ensuring the selector never sees validation fold statistics.

---

## Artifact Summary

All preprocessing metadata is logged to MLflow under the `preprocessing/` artifact path:

| Artifact | Contents |
|----------|----------|
| `preprocessing/decision_scalers.json` | Per-column scaling decision |
| `preprocessing/numeric_transformers.json` | Scaler parameters (mean, std, IQR, etc.) |
| `preprocessing/feature_metadata.json` | Column lists, imputer values, encoder name, encoded feature names, target columns |
| `preprocessing/target_encoders.json` | Label-encoder class mappings per target column |
| `preprocessing_one_hot_encoder/` | Serialized `OneHotEncoder` model |

---

# Training

## Overview

Training is orchestrated by `src/ml/training.py` → `main()`. It supports three use cases, each mapped to an MLflow experiment:

| Use Case | MLflow Experiment | Task Type | Training Function |
|----------|------------------|-----------|------------------|
| Plausible Therapy | `METABRIC_UC_Plausible_Therapy` | `multilabel_classification` | `run_training_multilabel_classifier` |
| Plausible Survival Time | `METABRIC_UC_Plausible_Survival_Time` | `regression` | `run_training_regressor` |
| Plausible Diagnosis | `METABRIC_UC_Plausible_Diagnosis` | `multiclass_classification` | `run_training_multiclass_classifier` |

---

## Leave-One-Cohort-Out (LOCO) Cross-Validation

All use cases use **Leave-One-Cohort-Out** (LOCO) as the outer validation strategy, implemented in `run_leave_one_cohort_out_experiment`.

### Outer loop — LOCO

The dataset contains a `cohort` column. For each unique cohort value $c_i$:

1. **Outer-train split:** all rows where `cohort ≠ c_i`.
2. **Held-out test split:** all rows where `cohort = c_i`.
3. All preprocessing (steps 1–11 above) is fit on the outer-train split and applied to both splits.
4. A separate MLflow run is opened for each held-out cohort.

This simulates deploying to a completely unseen patient cohort, providing an unbiased estimate of generalization across institutional or batch boundaries.

### Inner loop — Stratified K-Fold model selection

Within each outer fold, `build_inner_cv_splits` constructs explicit inner CV splits **from the outer-train data only**:

- A 1-D surrogate stratification label is built per task type:
  - **Regression:** quantile bins of the target.
  - **Multiclass:** one-hot argmax label.
  - **Multilabel:** sum of active labels.
- `StratifiedKFold(n_splits=min(5, min_class_count), shuffle=True, random_state=42)` is used when every stratum has ≥ 2 samples.
- Falls back to plain `KFold` when a stratum has fewer than 2 samples; this is logged as a warning.

`GridSearchCV` is then called with these pre-built index splits, so the variance-threshold + estimator pipeline is tuned entirely within the outer-train fold.

---

## Model Candidates and Hyperparameter Search

Each training function runs `GridSearchCV` over the following candidates:

### Multilabel classifier (`OneVsRestClassifier`)
| Model | Key hyperparameters searched |
|-------|------------------------------|
| Random Forest | `n_estimators` ∈ {100, 200}, `max_depth` ∈ {None, 10, 20}, `min_samples_split` ∈ {2, 5}, `min_samples_leaf` ∈ {1, 2}, `class_weight` ∈ {None, "balanced"} |
| XGBoost (optional) | `n_estimators` ∈ {100, 200}, `max_depth` ∈ {3, 6}, `learning_rate` ∈ {0.05, 0.1}, `subsample` ∈ {0.8, 1.0}, `colsample_bytree` ∈ {0.8, 1.0} |

Scoring: `f1_samples`.

### Multiclass / multi-target classifier (`MultiOutputClassifier`)
Same candidate grid as multilabel. Scoring: `f1_weighted`.

### Regressor
| Model | Key hyperparameters searched |
|-------|------------------------------|
| Linear Regression | — |
| Random Forest Regressor | `n_estimators` ∈ {100, 200}, `max_depth` ∈ {None, 10, 20}, `min_samples_split` ∈ {2, 5}, `min_samples_leaf` ∈ {1, 2} |

Scoring: `r2`.

The candidate with the highest mean inner-CV score is selected. Each candidate's best parameters and score are stored in `model_selection/{model_name}_summary.json`.

---

## MLflow Run Structure

One MLflow **run** is created per held-out cohort per use case. Runs are grouped under the corresponding experiment.

### Run tags (set at run start)

| Tag | Value |
|-----|-------|
| `use_case` | run name prefix (e.g. `diagnosis_multiclass`) |
| `task_type` | `multiclass_classification` / `multilabel_classification` / `regression` |
| `target_columns` | comma-separated target column names |
| `dataset_path` | path to the input CSV |
| `outer_strategy` | `leave_one_cohort_out` |
| `inner_strategy` | `stratified_cv` or `kfold` |
| `held_out_cohort` | cohort label held out in this run |

### Logged parameters

| Parameter | Description |
|-----------|-------------|
| `held_out_cohort` | cohort held out as test set |
| `inner_cv_splits` | number of inner folds used |
| `inner_cv_strategy` | `stratified_cv` or `kfold` |
| `selected_model` | name of the winning candidate |
| `xgboost_available` | `False` when XGBoost is not installed |

---

## Metrics

### Classification (multiclass / multi-target)

| Metric key | Description |
|------------|-------------|
| `{model}_best_cv_f1_weighted` | Mean inner-CV F1 (weighted) per candidate |
| `selected_model_best_cv_f1_weighted` | Inner-CV score of the chosen model |
| `val_{target}_accuracy` | Per-target accuracy on validation split (non-LOCO path) |
| `val_{target}_f1_weighted` | Per-target weighted F1 on validation split |
| `val_exact_match_accuracy` | Fraction of rows where all targets match (val) |
| `val_mean_target_f1_weighted` | Mean per-target weighted F1 (val) |
| `{target}_accuracy` | Per-target accuracy on held-out cohort |
| `{target}_f1_weighted` | Per-target weighted F1 on held-out cohort |
| `test_exact_match_accuracy` | Exact-match accuracy on held-out cohort |
| `test_mean_target_f1_weighted` | Mean per-target weighted F1 on held-out cohort |

### Multilabel classification

| Metric key | Description |
|------------|-------------|
| `{model}_best_cv_f1_samples` | Mean inner-CV F1 (samples) per candidate |
| `val_subset_accuracy` / `test_subset_accuracy` | Subset accuracy |
| `val_f1_samples` / `test_f1_samples` | Sample-averaged F1 |
| `val_f1_macro` / `test_f1_macro` | Macro F1 |
| `{target}_f1_binary` | Per-label binary F1 on test set |

### Regression

| Metric key | Description |
|------------|-------------|
| `{model}_best_cv_r2` | Mean inner-CV R² per candidate |
| `val_r2` / `test_r2` | R² on validation / test |
| `val_mae` / `test_mae` | Mean absolute error |
| `val_rmse` / `test_rmse` | Root mean squared error |

### Timing metrics (all task types)

Each timed stage (`timed_stage`) logs `duration_seconds_{stage}` as an MLflow metric.

---

## Artifact Locations in MLflow

All artifacts are stored under the run's artifact URI (local path: `mlruns/{experiment_id}/{run_id}/artifacts/`).

### Preprocessing artifacts (`preprocessing/`)

See the **Preprocessing Artifact Summary** table above.

### Model selection artifacts (`model_selection/`)

| Artifact | Contents |
|----------|----------|
| `model_selection/{model}_summary.json` | Best params, best CV score, scoring metric, CV splits for each candidate |
| `model_selection/all_model_summaries.json` | Combined summary for all candidates |
| `model_selection/inner_strata_counts.json` | Class counts used to build inner CV splits |
| `model_selection/per_target_metrics.json` | Per-target accuracy and F1 on the test set |
| `model_selection/per_target_classification_reports.json` | Full `sklearn` classification report per target |

### SHAP artifacts (`model_selection/shap_{target}/`)

After the best model is selected, SHAP values are computed with `log_pipeline_shap_artifacts`:

1. The `VarianceThreshold` step is applied to extract the post-selection feature matrix.
2. For `MultiOutputClassifier` / `OneVsRestClassifier`, one SHAP explainer is built **per sub-estimator** (i.e. per target column).
3. A `TreeExplainer` is used for Random Forest and XGBoost; `shap.Explainer` is used as fallback.

| Artifact | Contents |
|----------|----------|
| `model_selection/shap_{target}/shap_values.json` | Feature names + raw SHAP value matrix |
| `model_selection/shap_{target}/shap_summary_plot.png` | SHAP beeswarm summary plot |
| `model_selection/shap_{target}/metadata.json` | Wrapper type and sub-estimator index |

If SHAP computation fails (e.g. incompatible dtypes), the failure is logged as `{artifact_key}_shap_skipped = True` and training continues.

### Trained model artifact (`trained_model/`)

The best pipeline (VarianceThreshold + estimator) is serialized with `mlflow.sklearn.log_model` under the artifact name `trained_model`. This produces the standard MLflow model directory:

```
trained_model/
├── MLmodel          ← model metadata (flavors, run ID, artifact path)
├── model.pkl        ← serialized sklearn pipeline
├── conda.yaml       ← conda environment specification
└── requirements.txt ← pip requirements
```

### Run summary (`run_summary.json`)

A compact JSON artifact logged at the root of each run containing:
- run name, task type, dataset path, row count
- preprocessing column counts and split sizes
- training summary (best model name, CV score, test metrics)

---

## Model Registration in MLflow

Models are **not automatically registered** to the MLflow Model Registry during training. The `log_sklearn_model_artifact` call stores the model as a run artifact only.

To promote a model to the registry, use the MLflow UI or the tracking client:

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
# Register from a specific run artifact
model_version = client.create_registered_model("METABRIC_Diagnosis")
client.create_model_version(
    name="METABRIC_Diagnosis",
    source="mlruns/<experiment_id>/<run_id>/artifacts/trained_model",
    run_id="<run_id>",
)
```

Existing runs and their artifacts can be queried with `src/ml/get_registered_models.py`, which wraps `MlflowClient.search_runs` across all experiments and returns a flattened DataFrame with metrics, params, tags, artifact URIs, and parent/child run relationships.
