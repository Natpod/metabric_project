import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
import mlflow
import shap
from imblearn.over_sampling import SMOTE
import os
from mlflow.tracking import MlflowClient
import argparse

def load_preprocessing_artifacts(run_id):
    """Load preprocessing artifacts from MLflow given a run ID."""
    client = MlflowClient()
    artifacts_dir = client.download_artifacts(run_id, "preprocessing")

    # Load the encoder artifact
    encoder_path = os.path.join(artifacts_dir, "encoder.pkl")
    encoder = pd.read_pickle(encoder_path)

    return encoder

def load_model(run_id, model_name="model.pkl"):
    """Load a trained model from MLflow given a run ID."""
    client = MlflowClient()
    artifacts_dir = client.download_artifacts(run_id, "model")
    model_path = os.path.join(artifacts_dir, model_name)
    model = pd.read_pickle(model_path)
    return model

def prepare_data_for_inference(df, test_size=0.2, random_state=42, run_id=None):
    X = df

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'boolean']).columns
    numerical_cols = X.select_dtypes(include=['number']).columns

    if run_id:
        # Load preprocessing artifacts from MLflow
        encoder = load_preprocessing_artifacts(run_id)
        X_encoded = encoder.transform(X[categorical_cols])
        encoded_col_names = encoder.get_feature_names_out(categorical_cols)
        X_encoded_df = pd.DataFrame(X_encoded, columns=encoded_col_names, index=X.index)
        X = pd.concat([X.drop(columns=categorical_cols), X_encoded_df], axis=1)
    else:
        # Encode categorical variables
        if len(categorical_cols) > 0:
            encoder = preprocessing.OneHotEncoder(sparse=False, handle_unknown='ignore')
            X_encoded = encoder.fit_transform(X[categorical_cols])
            encoded_col_names = encoder.get_feature_names_out(categorical_cols)
            X_encoded_df = pd.DataFrame(X_encoded, columns=encoded_col_names, index=X.index)
            X = pd.concat([X.drop(columns=categorical_cols), X_encoded_df], axis=1)
    return X


def get_best_run_id(experiment_name):
    """Get the best run ID from an MLflow experiment based on a specified metric."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")
    
    runs = client.search_runs(experiment.experiment_id, order_by=["metrics.val_auc DESC"])
    if not runs:
        raise ValueError(f"No runs found for experiment '{experiment_name}'.")
    
    best_run_id = runs[0].info.run_id
    return best_run_id

def get_shap_values(model, X):
    """Calculate SHAP values for the given model and input data."""
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    return shap_values

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