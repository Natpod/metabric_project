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


# RESULTS

# EVALUATION

Therapeutic target
- Metrics
- Explainability

Prognosis target
- Metrics
- Explainability

Survival target
- Metrics
- Explainability
