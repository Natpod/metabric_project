# METABRIC Breast Cancer Clinical Data Analysis Project

## Background
Breast cancer is one of the highest-impact oncology problems in public health because treatment decisions depend on a mix of histopathology, molecular subtype, tumor burden, and patient context. The METABRIC dataset is well suited for this project because it combines clinical variables with molecular measurements from breast tumor samples, which makes it possible to study clinically relevant prediction tasks instead of only descriptive biomarker analysis.

This repository uses METABRIC data to build three practical machine learning use cases around diagnosis, treatment support, and prognosis. The goal is not to replace clinical judgment, but to structure signals from heterogeneous patient cohorts into reproducible analytical workflows that can later be evaluated for translational use.

## Project Objectives
The repository has three explicit objectives:

1. Document data quality, cohort heterogeneity, and preprocessing decisions before modeling.
2. Train clinically interpretable predictive workflows for diagnosis, therapy support, and survival modeling.
3. Evaluate whether models generalize across cohorts instead of only fitting a random split of the same dataset.

The current training pipeline is built around cohort-aware validation:

1. Outer validation: Leave-One-Cohort-Out using the `cohort` field.
2. Inner validation: stratified cross-validation on the outer-train partition.
3. Feature selection: variance-based filtering inside each inner fold.
4. Batch correction: applied only on outer-train gene-expression features.
5. Held-out cohort: left untouched for final outer-fold evaluation.

This design is justified because the main deployment risk in this dataset is not only label noise, but domain shift between acquisition cohorts. A standard random split would overestimate performance by mixing patients from the same cohort into both train and test.

## Use Cases

### 1. Therapy Support
Target variables:

- `chemotherapy`
- `hormone_therapy`
- `radio_therapy`
- `type_of_breast_surgery`

Operational objective:
Estimate plausible treatment patterns from molecular and clinical context.

Why this use case matters:
Treatment planning is intrinsically multi-label. A patient may receive systemic therapy, local therapy, and a surgery strategy in combination. Modeling these decisions jointly is more realistic than training one isolated model per treatment variable because co-administration patterns are clinically coupled.

Why it is justified in this repository:
The code transforms surgery into dummy targets and trains a multilabel classifier for therapy recommendation patterns. This is appropriate for retrospective pattern learning because the target is not "best treatment in an interventional sense", but "treatment profile historically associated with similar patients". That makes it useful as a decision-support or case-retrieval tool, not as an autonomous prescribing system.

### 2. Survival Time Prognosis
Target variable:

- `overall_survival_months`

Operational objective:
Estimate survival duration for patients in the subset used for prognosis modeling.

Why this use case matters:
Prognosis supports follow-up planning, intensity of monitoring, and risk communication. Time-to-event problems are central in oncology because even when they are not used directly for treatment selection, they affect how clinicians interpret disease burden and likely disease course.

Why it is justified in this repository:
The current code filters this use case to patients with `overall_survival == 0`, so the model is explicitly trained on patients with observed death events rather than mixing event times with censored observations. That restriction simplifies the task into a regression problem over observed survival time. It is not a full survival-analysis pipeline, but it is a defensible first prognostic benchmark given the current implementation.

### 3. Diagnosis Support
Target variable:

- `oncotree_code`

Operational objective:
Predict a diagnosis-oriented tumor coding label from the available clinical and molecular features.

Why this use case matters:
Diagnosis is upstream of treatment and prognosis. If a model can recover tumor coding or related disease subtype from the integrated feature space, it suggests that the data contain structured diagnostic signal rather than only downstream treatment signal.

Why it is justified in this repository:
The diagnosis workflow one-hot encodes `oncotree_code` and trains a multiclass-style classifier through the existing multi-output path. Clinically, this use case is best understood as a consistency or decision-support task: it can help flag whether the combined molecular and clinical profile is coherent with the recorded tumor classification.

## Repository Structure

```text
README.md
requirements.txt
data/
	FCS_ml_test_input_data_rna_mutation.csv
	inference_test/
		input/
			inference_diagnosis.csv
			inference_survival.csv
			inference_therapy.csv
		output/
docs/
	metadata.csv
mlruns/
	... MLflow tracking data, preprocessing artifacts, model artifacts, and run summaries
reports/
	eda_report.html
	qc_report.html
src/
	main.py
	data/
		EDA.py
		quality/
			QC.py
	ml/
		predict.py
		training.py
```

### What each area is for
- `data/`: source dataset plus small inference input files used to test trained experiments.
- `docs/`: metadata and documentation support files for the dataset.
- `mlruns/`: MLflow outputs, including preprocessing metadata, selected models, and per-run summaries.
- `reports/`: generated QC and EDA HTML artifacts.
- `src/data/quality/QC.py`: data quality checks used to derive preprocessing decisions.
- `src/data/EDA.py`: exploratory analysis and report generation.
- `src/ml/training.py`: cohort-aware training, preprocessing, model selection, and MLflow logging.
- `src/ml/predict.py`: inference-time preprocessing reconstruction and model loading from MLflow.
- `src/main.py`: project entrypoint that orchestrates training and inference.

## Workflow

The intended workflow in this repository is:

1. QC: inspect duplicates, boolean-like variables, outlier candidates, and high-cardinality categorical fields.
2. EDA: characterize feature distributions, cohort heterogeneity, and task-specific structure.
3. Training: run the three use cases with cohort-aware evaluation.
4. Logging: store preprocessing artifacts, metrics, and models in MLflow.
5. Inference: load the best run for each experiment and apply the same preprocessing logic to new CSV inputs.

### Training workflow in more detail
For each use case, the current training code follows this sequence:

1. Load the raw METABRIC dataset.
2. Define gene-expression columns as columns not included in the configured non-gene list.
3. Apply use-case specific target filtering and target engineering.
4. Split outer folds by `cohort` with Leave-One-Cohort-Out.
5. Fit preprocessing only on the outer-train partition.
6. Apply train-only batch correction on gene-expression features.
7. Run inner stratified CV within outer-train.
8. Perform variance-based feature selection inside the model-selection pipeline.
9. Select and refit the best estimator on outer-train.
10. Evaluate on the untouched held-out cohort.
11. Log metrics, preprocessing metadata, and model artifacts to MLflow.

This workflow is the core methodological choice of the repository, because it explicitly tests whether learned patterns transfer across cohorts rather than only across random patients.

### Gene-expression column treatment
`gene_expr_cols` are handled differently from ordinary clinical numeric fields because they are treated as molecular signal rather than generic tabular measurements.

In the current preprocessing code, gene-expression columns are processed as follows:

1. They are defined as every column not listed in the configured non-gene feature list.
2. They are coerced to numeric values, including cleanup of bracketed string representations such as scientific-notation strings stored as text.
3. Low-magnitude expression values are thresholded elementwise so that values with `abs(value) <= 1.5` are replaced with `0`.
4. In the Leave-One-Cohort-Out workflow, this thresholding happens once per outer split before inner CV begins, so it is not a fold-local feature-selection step.
5. Variance-based feature selection is separate and occurs inside the model-selection pipeline through `VarianceThreshold` during inner CV.
6. In the cohort-aware path, batch correction is applied only to outer-train gene-expression columns by centering each training cohort to the global outer-train mean. The held-out cohort is not batch-corrected.
7. Gene-expression columns are excluded from the scaler-selection heuristics used for other numeric features. In practice they are carried forward without StandardScaler, MinMaxScaler, RobustScaler, or log-scaling decisions.

The important implication is that gene-expression preprocessing is partly split-aware but not entirely CV-local: expression thresholding is fixed at the outer-split level, while variance filtering is learned again inside each inner fold.

### Current entrypoint behavior
`src/main.py` currently calls training for all three use cases and then runs inference on the prepared CSV files under `data/inference_test/input/`. The QC and EDA calls are present in the entrypoint but commented out, so they are available in the codebase even if they are not executed by default in the current script.

## Installation and Execution

```bash
conda create -n metabric python=3.10 -y
conda activate metabric
pip install -r requirements.txt
python src/main.py --no-show-plots
```

## Why Cohort-Aware Validation Is Central Here
The repository assumes that the most important source of optimistic bias is cohort mixing. METABRIC aggregates patients from different cohorts, and cohort can encode acquisition effects, site effects, and population differences. For that reason, the workflow treats `cohort` as the outer generalization boundary.

In practical terms, this means the repository is designed to answer the following question for each use case:

"If one cohort were genuinely new at deployment time, would the model trained on the remaining cohorts still perform acceptably?"

That is a stronger and more clinically relevant question than asking whether the model performs well on a random holdout from the same pooled dataset.

# METHODOLOGY REFERENCES
More details about quality check, preprocessing, training can be found in `docs/eda.md`, and `docs.preprocessing`

# RESULTS AND EVALUATION

Final report of best run - models per use case can be seen in reports\EvaluationResults.ipynb and docs/reports\EvaluationResults.md 

# RESULTS
**4. Evaluation of Use Case 1: Therapy Support (Multilabel)**
Treatment planning is an inherently multilabel task, where decisions regarding chemotherapy, radiotherapy, surgery, and hormonal therapy are clinically coupled. The model was designed to identify these historical prescription patterns, recognizing that jointly modeling these labels captures the logic of co-administration better than isolated models.

### Model Architecture Justification

**Random Forest (RF)** was selected as the superior estimator over XGBoost for two fundamental technical reasons:

* **Handling Imbalance:** In the context of highly imbalanced therapeutic labels, RF demonstrated a superior ability to adapt to minority classes.
* **Optimization Metric:** The **f1_samples** metric was prioritized to reflect patient-level (row-wise) precision, since XGBoost tends to optimize global performance at the expense of less frequent therapeutic combinations.

Although the use of SHAP provides indispensable granular interpretability for decision support, the current overall accuracy indicates that the model should be used as a case-retrieval and support tool, rather than for autonomous prescription.

---

**5. Evaluation of Use Case 2: Survival Prognosis (Regression)**
Prognosis is the cornerstone of clinical follow-up. For this validation, the use case was strictly restricted to patients with observed death events (**overall_survival == 0**, according to the source context), transforming the task into a regression problem over survival duration in months.

### Critical Performance Analysis

The resulting model explains only **13.7% of the variance** (R²). From an expert clinical validation perspective, **this metric rules out the clinical deployment of the model**. Such limited predictive capacity suggests that the current feature space (clinical-molecular) is insufficient, or that the RF/Linear regression approach is fundamentally incapable of capturing the temporal dynamics of survival without considering censored data or deeper nonlinear interactions.

The results should be treated as purely experimental and highlight a critical gap in prognosis modeling.

---

**6. Evaluation of Use Case 3: Diagnostic Support (Multiclass)**
Accurate diagnosis, grounded in **Oncotree** coding, is the prerequisite for any intervention. This use case evaluated model consistency under the **f1_weighted** metric.

The results in this area are significantly more robust. The **LOCO** strategy confirmed that the model successfully generalizes across all cohort combinations, which **validates the integrity of the integrated feature space as a reliable biological signal**. The model’s ability to recover tumor classification from molecular and clinical profiles demonstrates that the data possess a coherent diagnostic structure that is resilient to the heterogeneity of data sources.

---

**7. Conclusions and Recommendations for Clinical Applicability**
This technical validation process provides an honest roadmap of the potential and limitations of AI applied to **METABRIC**.

### Summary of Conclusions

* **Rigor in Generalization:** The LOCO method is the only tool capable of exposing optimistic bias in heterogeneous oncology data, revealing the true capacity for institutional transferability.
* **Random Forest Superiority in Clinical Contexts:** In scenarios of marked imbalance and complex tabular data (such as therapy support), RF offers stability and minority-class fitting that surpasses global gradient optimizers.
* **Diagnosis-Prognosis Gap:** There is an alarming disparity between the high reliability of diagnostic models and the inability of survival models to explain temporal variance. This requires a reassessment of the molecular features used for long-term prognosis.

### Final Statement on Use

These models should be integrated **exclusively as decision-support tools**. The survival metric demands extreme caution; the system should act as a mechanism for structuring complex signals to assist professional judgment, and never as an autonomous decision-making system.
