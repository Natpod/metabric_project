import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
import mlflow
import mlflow.sklearn
import shap
from imblearn.over_sampling import SMOTE
import os
import json
from mlflow.tracking import MlflowClient
import argparse
import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier


PREPROCESSING_ENCODER_ARTIFACT = "preprocessing_one_hot_encoder"
TRAINED_MODEL_ARTIFACT = "trained_model"


def create_one_hot_encoder():
    """Create a dense OneHotEncoder across scikit-learn versions."""
    try:
        return preprocessing.OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    except TypeError:
        return preprocessing.OneHotEncoder(sparse=False, handle_unknown='ignore')


def normalize_categorical_values(X, categorical_cols, categorical_fill_value="unknown"):
    """Convert categorical inputs to a uniform string representation for encoding."""
    if len(categorical_cols) == 0:
        return X

    X = X.copy()
    X.loc[:, categorical_cols] = X[categorical_cols].astype('string').fillna(categorical_fill_value)
    return X

def load_preprocessing_artifacts(run_id):
    """Load preprocessing artifacts from MLflow given a run ID."""
    client = MlflowClient()
    feature_metadata_path = client.download_artifacts(run_id, "preprocessing/feature_metadata.json")
    numeric_transformers_path = client.download_artifacts(run_id, "preprocessing/numeric_transformers.json")
    with open(feature_metadata_path, "r", encoding="utf-8") as metadata_file:
        feature_metadata = json.load(metadata_file)
    with open(numeric_transformers_path, "r", encoding="utf-8") as transformers_file:
        feature_metadata["numeric_transformers"] = json.load(transformers_file)

    encoder = None
    try:
        encoder_artifact = feature_metadata.get("one_hot_encoder_artifact", PREPROCESSING_ENCODER_ARTIFACT)
        encoder = mlflow.sklearn.load_model(f"runs:/{run_id}/{encoder_artifact}")
    except Exception:
        encoder = None

    feature_metadata["one_hot_encoder"] = encoder
    return feature_metadata


def impute_missing_values(X, categorical_cols, numerical_cols, numeric_fill_values=None, categorical_fill_value="unknown"):
    """Impute numeric columns with medians and categorical columns with a sentinel value."""
    X = X.copy()

    if len(categorical_cols) > 0:
        X = normalize_categorical_values(X, categorical_cols, categorical_fill_value)

    if len(numerical_cols) > 0:
        if numeric_fill_values is None:
            numeric_fill_values = X[numerical_cols].median().to_dict()
        X.loc[:, numerical_cols] = X[numerical_cols].fillna(numeric_fill_values)

    return X


def apply_numeric_transformers(X, numeric_transformers):
    """Apply the numeric transforms fitted during training to matching inference columns."""
    X = X.copy()

    for col, transformer in (numeric_transformers or {}).items():
        if col not in X.columns:
            continue

        X[col] = pd.to_numeric(X[col], errors="coerce")

        transform_type = transformer.get("type")
        if transform_type == "Log1pTransform":
            X.loc[:, col] = X[col].apply(lambda value: np.log1p(value) if pd.notna(value) else value)
        elif transform_type == "MinMaxScaler":
            min_val = transformer.get("min")
            max_val = transformer.get("max")
            if min_val is not None and max_val is not None and max_val > min_val:
                X.loc[:, col] = (X[col] - min_val) / (max_val - min_val)
        elif transform_type == "StandardScaler":
            mean = transformer.get("mean")
            std = transformer.get("std")
            if mean is not None and std is not None and std > 0:
                X.loc[:, col] = (X[col] - mean) / std
        elif transform_type == "RobustScaler":
            median = transformer.get("median")
            iqr = transformer.get("iqr")
            if median is not None and iqr is not None and iqr > 0:
                X.loc[:, col] = (X[col] - median) / iqr

    return X

def _is_model_artifact_dir(client, run_id, artifact_path):
    """Return True when artifact_path contains an MLflow model (MLmodel file)."""
    try:
        artifacts = client.list_artifacts(run_id, artifact_path)
    except Exception:
        return False
    return any((not item.is_dir) and os.path.basename(item.path) == "MLmodel" for item in artifacts)


def find_model_artifact_path(run_id, preferred_artifact_name=TRAINED_MODEL_ARTIFACT, max_depth=4):
    """Find the first artifact directory in a run that contains an MLmodel file."""
    client = MlflowClient()

    preferred_candidates = [
        preferred_artifact_name,
        TRAINED_MODEL_ARTIFACT,
        "model",
        "models",
    ]
    seen = set()
    for candidate in preferred_candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if _is_model_artifact_dir(client, run_id, candidate):
            return candidate

    queue = [("", 0)]
    while queue:
        current_path, depth = queue.pop(0)
        try:
            children = client.list_artifacts(run_id, current_path)
        except Exception:
            continue

        for child in children:
            if child.is_dir:
                if _is_model_artifact_dir(client, run_id, child.path):
                    return child.path
                if depth + 1 < max_depth:
                    queue.append((child.path, depth + 1))

    raise ValueError(
        f"No MLflow model artifact directory (containing MLmodel) found for run '{run_id}'."
    )


def load_model(run_id, model_artifact_name=TRAINED_MODEL_ARTIFACT):
    """Load a trained model from MLflow given a run ID."""
    resolved_model_artifact_path = find_model_artifact_path(run_id, preferred_artifact_name=model_artifact_name)
    return mlflow.sklearn.load_model(f"runs:/{run_id}/{resolved_model_artifact_path}")

def prepare_data_for_inference(df, test_size=0.2, random_state=42, run_id=None):
    X = df

    if run_id:
        # Load preprocessing artifacts from MLflow
        preprocessing_artifacts = load_preprocessing_artifacts(run_id)
        categorical_cols = pd.Index(preprocessing_artifacts["categorical_columns"])
        numerical_cols = pd.Index(preprocessing_artifacts["numeric_columns"])
        numeric_transformers = preprocessing_artifacts.get("numeric_transformers", {})
        encoder = preprocessing_artifacts["one_hot_encoder"]
        X = impute_missing_values(
            X,
            categorical_cols.intersection(X.columns),
            numerical_cols.intersection(X.columns),
            numeric_fill_values=preprocessing_artifacts.get("numeric_imputers"),
            categorical_fill_value=preprocessing_artifacts.get("categorical_imputer_value", "unknown")
        )

        if encoder is None:
            raise ValueError(f"No preprocessing one-hot encoder artifact found for run '{run_id}'.")

        X_encoded = encoder.transform(X[categorical_cols])
        encoded_col_names = encoder.get_feature_names_out(categorical_cols)
        X_encoded_df = pd.DataFrame(X_encoded, columns=encoded_col_names, index=X.index)
        X = pd.concat([X.drop(columns=categorical_cols), X_encoded_df], axis=1)
        X = apply_numeric_transformers(X, numeric_transformers)

        encoded_feature_names = preprocessing_artifacts.get("encoded_feature_names")
        if encoded_feature_names:
            available = [c for c in encoded_feature_names if c in X.columns]
            X = X[available]
    else:
        # Identify categorical and numerical columns
        categorical_cols = X.select_dtypes(include=['object', 'boolean']).columns
        numerical_cols = X.select_dtypes(include=['number']).columns
        X = impute_missing_values(X, categorical_cols, numerical_cols)

        # Encode categorical variables
        if len(categorical_cols) > 0:
            encoder = create_one_hot_encoder()
            X_encoded = encoder.fit_transform(X[categorical_cols])
            encoded_col_names = encoder.get_feature_names_out(categorical_cols)
            X_encoded_df = pd.DataFrame(X_encoded, columns=encoded_col_names, index=X.index)
            X = pd.concat([X.drop(columns=categorical_cols), X_encoded_df], axis=1)
    return X


def get_best_run_id(experiment_name):
    """Get the best run ID from an MLflow experiment among runs that have a logged model artifact."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    runs = client.search_runs(
        [experiment.experiment_id],
        max_results=200,
        order_by=["attributes.start_time DESC"],
    )
    if not runs:
        raise ValueError(f"No runs found for experiment '{experiment_name}'.")

    metric_priority = [
        "selected_model_best_cv_f1_weighted",
        "selected_model_best_cv_f1_samples",
        "selected_model_best_cv_r2",
        "test_exact_match_accuracy",
        "test_f1_samples",
        "test_r2",
    ]

    ranked_runs = []
    for run in runs:
        run_id = run.info.run_id
        try:
            find_model_artifact_path(run_id)
        except Exception:
            continue

        score = None
        for metric_key in metric_priority:
            metric_value = run.data.metrics.get(metric_key)
            if metric_value is not None:
                score = float(metric_value)
                break

        if score is None:
            score = float("-inf")
        ranked_runs.append((score, run.info.start_time or 0, run_id))

    if not ranked_runs:
        raise ValueError(
            f"No runs with a loadable model artifact were found for experiment '{experiment_name}'."
        )

    ranked_runs.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best_run_id = ranked_runs[0][2]
    return best_run_id

def get_shap_values(model, X):
    """Calculate SHAP values for the pipeline model and input data.

    Unwraps the VarianceThreshold step, applies it to X, then builds a
    SHAP explainer on the inner estimator (TreeExplainer for tree models,
    generic Explainer otherwise).
    """
    # Unwrap the variance selector from the Pipeline
    if hasattr(model, "named_steps") and "variance" in model.named_steps:
        variance_step = model.named_steps["variance"]
        X_selected = pd.DataFrame(
            variance_step.transform(X),
            columns=X.columns[variance_step.get_support()],
            index=X.index,
        )
        inner_model = model.named_steps["model"]
    else:
        X_selected = X
        inner_model = model

    def safe_model_output(estimator, feature_frame):
        """Return estimator output as a numpy array with stable shape for SHAP fallbacks."""
        if hasattr(estimator, "predict_proba"):
            output = estimator.predict_proba(feature_frame)
        else:
            output = estimator.predict(feature_frame)

        if isinstance(output, list):
            arrays = [np.asarray(item) for item in output]
            if arrays and all(arr.ndim == 2 for arr in arrays):
                return np.concatenate(arrays, axis=1)
            return np.asarray(output)

        return np.asarray(output)

    def explain_single_estimator(estimator, feature_frame):
        model_type = type(estimator).__name__
        explained = None
        if "XGB" in model_type or "RandomForest" in model_type:
            try:
                explainer = shap.TreeExplainer(estimator)
                explained = explainer(feature_frame)
            except Exception:
                explained = None

        if explained is None:
            background = shap.sample(feature_frame, min(100, len(feature_frame)))
            explainer = shap.KernelExplainer(lambda data: safe_model_output(estimator, data), background)
            explained = explainer(feature_frame)

        values = np.array(getattr(explained, "values", explained))
        if values.ndim == 3:
            values = np.abs(values).mean(axis=2)
        return values

    if isinstance(inner_model, (MultiOutputClassifier, OneVsRestClassifier)):
        if not hasattr(inner_model, "estimators_") or len(inner_model.estimators_) == 0:
            raise ValueError("Wrapped classifier has no fitted sub-estimators for SHAP inference.")

        per_target_values = []
        for estimator in inner_model.estimators_:
            per_target_values.append(explain_single_estimator(estimator, X_selected))

        # Aggregate target-specific SHAP values into a single (samples, features) matrix.
        stacked_values = np.stack(per_target_values, axis=2)
        aggregated_values = np.abs(stacked_values).mean(axis=2)
        return shap.Explanation(
            values=aggregated_values,
            data=X_selected.to_numpy(),
            feature_names=X_selected.columns.tolist(),
        )

    single_values = explain_single_estimator(inner_model, X_selected)
    return shap.Explanation(
        values=single_values,
        data=X_selected.to_numpy(),
        feature_names=X_selected.columns.tolist(),
    )

def main(DATASET_PATH, EXPERIMENT_NAME):
    # Example usage
    df = pd.read_csv(DATASET_PATH)
    best_run_id = get_best_run_id(EXPERIMENT_NAME)

    # Load your trained model here (e.g., using joblib or pickle)
    X = prepare_data_for_inference(df, run_id=best_run_id)
    model = load_model(best_run_id)
    predictions = model.predict(X)

    # Calculate SHAP values
    shap_values = get_shap_values(model, X)

    return predictions, shap_values

if __name__ == "__main__":
    argv = argparse.ArgumentParser()
    argv.add_argument("--dataset_path", type=str, required=True, help="Path to the inference dataset CSV file")
    argv.add_argument("--experiment_name", type=str, required=True, help="Name of the MLflow experiment")
    main_args = argv.parse_args()
    main(main_args.dataset_path, main_args.experiment_name)