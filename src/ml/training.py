import sklearn.preprocessing as preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import shap
from imblearn.over_sampling import SMOTE
from data.quality.QC import run_qc
from scipy import stats
import argparse
import time
from contextlib import contextmanager

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


PREPROCESSING_ENCODER_ARTIFACT = "preprocessing_one_hot_encoder"
TRAINED_MODEL_ARTIFACT = "trained_model"
COHORT_COLUMN = "cohort"
DEFAULT_INNER_CV_SPLITS = 5
DEFAULT_VARIANCE_THRESHOLD = 0.0


def log_progress(message):
    """Emit a timestamped progress message for long-running training steps."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


@contextmanager
def timed_stage(stage_name, artifact_key=None):
    """Time a training stage, print progress, and log duration when an MLflow run is active."""
    log_progress(f"START {stage_name}")
    start_time = time.perf_counter()
    try:
        yield
    finally:
        elapsed_seconds = time.perf_counter() - start_time
        log_progress(f"END {stage_name} ({elapsed_seconds:.2f}s)")
        if mlflow.active_run() is not None and artifact_key is not None:
            mlflow.log_metric(f"duration_seconds_{artifact_key}", float(elapsed_seconds))


def log_sklearn_model_artifact(model, artifact_name):
    """Log a scikit-learn model using an MLflow-safe artifact name."""
    try:
        mlflow.sklearn.log_model(model, name=artifact_name)
    except TypeError:
        mlflow.sklearn.log_model(model, artifact_path=artifact_name)


def log_preprocessing_artifacts(preprocessing_artifacts, artifact_path="preprocessing"):
    """Log fitted preprocessing metadata to the active MLflow run."""
    if not preprocessing_artifacts:
        return

    mlflow.log_dict(
        preprocessing_artifacts["decision_scalers"],
        f"{artifact_path}/decision_scalers.json"
    )
    mlflow.log_dict(
        preprocessing_artifacts["numeric_transformers"],
        f"{artifact_path}/numeric_transformers.json"
    )
    mlflow.log_dict(
        {
            "categorical_columns": preprocessing_artifacts["categorical_columns"],
            "numeric_columns": preprocessing_artifacts["numeric_columns"],
            "numeric_imputers": preprocessing_artifacts["numeric_imputers"],
            "categorical_imputer_value": preprocessing_artifacts["categorical_imputer_value"],
            "one_hot_encoder_artifact": PREPROCESSING_ENCODER_ARTIFACT,
            "encoded_feature_names": preprocessing_artifacts["encoded_feature_names"],
            "target_columns": preprocessing_artifacts["target_columns"]
        },
        f"{artifact_path}/feature_metadata.json"
    )
    mlflow.log_dict(
        preprocessing_artifacts["target_encoders"],
        f"{artifact_path}/target_encoders.json"
    )

    if preprocessing_artifacts["one_hot_encoder"] is not None:
        log_sklearn_model_artifact(
            preprocessing_artifacts["one_hot_encoder"],
            PREPROCESSING_ENCODER_ARTIFACT
        )

def log_initial_columns(df):
    """Log the initial columns of the DataFrame to MLflow."""
    if mlflow.active_run() is None:
        return
    initial_columns = df.columns.tolist()
    mlflow.log_dict({"initial_columns": initial_columns}, "initial_columns.json")


def end_active_mlflow_run():
    """End the current MLflow run if one is active."""
    if mlflow.active_run() is not None:
        mlflow.end_run()


def set_run_tags(tags):
    """Set MLflow tags for the active run."""
    for key, value in tags.items():
        mlflow.set_tag(key, value)


def log_run_summary(summary, artifact_path="run_summary.json"):
    """Log a compact per-run summary artifact."""
    mlflow.log_dict(summary, artifact_path)


def build_run_summary(run_name, task_type, target_columns, preprocessing_artifacts, training_summary, dataset_path, row_count):
    """Build a compact summary of the training run for MLflow."""
    return {
        'run_name': run_name,
        'task_type': task_type,
        'dataset_path': dataset_path,
        'row_count': int(row_count),
        'target_columns': target_columns,
        'preprocessing': {
            'categorical_feature_count': len(preprocessing_artifacts.get('categorical_columns', [])),
            'numeric_feature_count': len(preprocessing_artifacts.get('numeric_columns', [])),
            'encoded_feature_count': len(preprocessing_artifacts.get('encoded_feature_names', [])),
            'split_sizes': preprocessing_artifacts.get('split_sizes', {}),
        },
        'training': training_summary,
    }


def log_shap_artifacts(shap_values, X, artifact_prefix="model_selection"):
    """Log SHAP values together with the feature names used to compute them."""
    mlflow.log_dict(
        {
            "feature_names": X.columns.tolist(),
            "shap_values": shap_values.values.tolist(),
        },
        f"{artifact_prefix}/shap_values.json"
    )
    shap.summary_plot(shap_values, X, show=False)
    mlflow.log_figure(plt.gcf(), f"{artifact_prefix}/shap_summary_plot.png")
    plt.close()


def get_categorical_feature_columns(df):
    """Return columns that should be treated as categorical, including booleans."""
    return pd.Index(
        [
            col for col in df.columns
            if pd.api.types.is_object_dtype(df[col])
            or pd.api.types.is_string_dtype(df[col])
            or pd.api.types.is_bool_dtype(df[col])
        ]
    )


def get_numeric_feature_columns(df):
    """Return numeric columns, excluding boolean dtypes."""
    return pd.Index(
        [
            col for col in df.columns
            if pd.api.types.is_numeric_dtype(df[col])
            and not pd.api.types.is_bool_dtype(df[col])
        ]
    )


def create_one_hot_encoder():
    """Create a dense OneHotEncoder across scikit-learn versions."""
    try:
        return preprocessing.OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    except TypeError:
        return preprocessing.OneHotEncoder(sparse=False, handle_unknown='ignore')


def sanitize_run_suffix(value):
    """Return a filesystem-safe suffix for MLflow run names and artifacts."""
    return str(value).strip().lower().replace(' ', '_').replace('/', '_').replace('\\', '_')


def get_outer_cohorts(df, cohort_column=COHORT_COLUMN):
    """Return outer cohorts to hold out for Leave-One-Cohort-Out evaluation."""
    if cohort_column not in df.columns:
        raise ValueError(f"Expected cohort column '{cohort_column}' in the dataset.")

    cohorts = df[cohort_column].dropna().astype(str).unique().tolist()
    if len(cohorts) < 2:
        raise ValueError("Leave-One-Cohort-Out requires at least two distinct cohorts.")
    return sorted(cohorts)


def build_stratification_labels(y, task_type):
    """Build a 1D surrogate target for stratified inner CV."""
    if task_type == 'regression':
        y_series = pd.Series(y).reset_index(drop=True)
        bin_count = min(5, max(2, y_series.nunique()))
        if bin_count < 2:
            raise ValueError("Regression target needs at least two unique values for stratified CV.")
        return pd.qcut(
            y_series.rank(method='first'),
            q=bin_count,
            labels=False,
            duplicates='drop'
        ).astype(str)

    if isinstance(y, pd.DataFrame):
        numeric_df = y.apply(pd.to_numeric, errors='ignore')
        if task_type == 'multiclass_classification' and set(numeric_df.dtypes.astype(str)).issubset({'int64', 'float64', 'uint8', 'bool', 'boolean'}):
            return numeric_df.astype(float).idxmax(axis=1).astype(str)
        if task_type == 'multilabel_classification':
            return numeric_df.fillna(0).astype(float).sum(axis=1).astype(int).astype(str)
        return y.astype(str).agg('||'.join, axis=1)

    return pd.Series(y).astype(str)


def build_inner_cv_splits(y, task_type, max_splits=DEFAULT_INNER_CV_SPLITS):
    """Build explicit stratified CV splits from the outer-train target only."""
    strata = pd.Series(build_stratification_labels(y, task_type)).reset_index(drop=True)
    class_counts = strata.value_counts()
    if class_counts.empty:
        raise ValueError("Unable to build inner CV splits from an empty target.")

    n_splits = min(max_splits, int(class_counts.min()))
    if n_splits < 2:
        raise ValueError(
            "Inner stratified CV requires at least two samples in every stratum. "
            f"Observed strata counts: {class_counts.to_dict()}"
        )

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    placeholder_X = np.zeros(len(strata))
    return list(splitter.split(placeholder_X, strata)), n_splits, class_counts.to_dict()


def encode_targets_for_outer_split(y_train, y_test, task_type):
    """Encode classification targets using outer-train labels only."""
    if task_type == 'regression':
        return y_train.copy(), y_test.copy(), {}

    if isinstance(y_train, pd.DataFrame):
        y_train_encoded = y_train.copy()
        y_test_encoded = y_test.copy()
        target_encoders = {}
        for col in y_train.columns:
            encoder = preprocessing.LabelEncoder()
            y_train_col = y_train[col].astype(str)
            y_train_encoded[col] = encoder.fit_transform(y_train_col)

            y_test_col = y_test[col].astype(str)
            unseen_labels = sorted(set(y_test_col.unique()) - set(encoder.classes_))
            if unseen_labels:
                raise ValueError(
                    f"Target column '{col}' contains unseen labels in the held-out cohort: {unseen_labels}"
                )
            y_test_encoded[col] = encoder.transform(y_test_col)
            target_encoders[col] = encoder.classes_.tolist()
        return y_train_encoded, y_test_encoded, target_encoders

    encoder = preprocessing.LabelEncoder()
    y_train_series = pd.Series(y_train).astype(str)
    y_test_series = pd.Series(y_test).astype(str)
    y_train_encoded = pd.Series(encoder.fit_transform(y_train_series), index=y_train.index, name=y_train.name)
    unseen_labels = sorted(set(y_test_series.unique()) - set(encoder.classes_))
    if unseen_labels:
        raise ValueError(f"Held-out cohort contains unseen target labels: {unseen_labels}")
    y_test_encoded = pd.Series(encoder.transform(y_test_series), index=y_test.index, name=y_test.name)
    return y_train_encoded, y_test_encoded, {y_train.name: encoder.classes_.tolist()}


def clip_numeric_outliers(train_df, test_df, outlier_columns):
    """Fit IQR clipping bounds on train and apply them to train and held-out test."""
    for col in outlier_columns:
        if col not in train_df.columns or not pd.api.types.is_numeric_dtype(train_df[col]) or pd.api.types.is_bool_dtype(train_df[col]):
            continue
        q1 = train_df[col].quantile(0.25)
        q3 = train_df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        train_df.loc[train_df[col] < lower_bound, col] = q1
        train_df.loc[train_df[col] > upper_bound, col] = q3
        if col in test_df.columns:
            test_df.loc[test_df[col] < lower_bound, col] = q1
            test_df.loc[test_df[col] > upper_bound, col] = q3


def group_high_cardinality_columns(train_df, test_df, high_cardinality_columns):
    """Fit rare-category grouping on train and apply the mapping to held-out test."""
    for col in high_cardinality_columns:
        if col not in train_df.columns:
            continue
        freq = train_df[col].value_counts(dropna=False)
        variance = freq.var()
        if pd.isna(variance):
            variance = 0
        threshold = freq.median() - 2 * variance
        rare = freq[freq < threshold].index
        grouped_col = 'geo_grouped' if col == 'geolocation_id' else f'{col}_grouped'
        train_df[grouped_col] = train_df[col].replace(rare, 'rare')
        if col in test_df.columns:
            test_df[grouped_col] = test_df[col].where(~test_df[col].isin(rare), 'rare')


def apply_train_only_batch_correction(X_train, train_cohorts, gene_expr_cols):
    """Center gene-expression features by train cohort means without touching held-out test."""
    corrected = X_train.copy()
    batch_metadata = {}
    candidate_columns = [col for col in gene_expr_cols if col in corrected.columns and pd.api.types.is_numeric_dtype(corrected[col])]
    if not candidate_columns:
        return corrected, batch_metadata

    train_cohorts = pd.Series(train_cohorts, index=corrected.index).astype(str)
    global_means = corrected[candidate_columns].mean()
    batch_metadata['columns'] = candidate_columns
    batch_metadata['global_mean'] = {col: float(global_means[col]) for col in candidate_columns}
    batch_metadata['cohort_offsets'] = {}

    for cohort_name, cohort_index in train_cohorts.groupby(train_cohorts).groups.items():
        cohort_means = corrected.loc[cohort_index, candidate_columns].mean()
        offsets = cohort_means - global_means
        corrected.loc[cohort_index, candidate_columns] = corrected.loc[cohort_index, candidate_columns] - offsets
        batch_metadata['cohort_offsets'][cohort_name] = {
            col: float(offsets[col]) for col in candidate_columns
        }

    return corrected, batch_metadata


def preprocess_outer_split(
    train_df,
    test_df,
    target_col,
    gene_expr_cols,
    has_duplicates,
    boolean_cast_columns,
    outlier_columns,
    high_cardinality_columns,
    id_column,
    cohort_column=COHORT_COLUMN,
    task_type=None,
):
    """Fit preprocessing on the outer-train split and keep the held-out cohort untouched by batch correction."""
    train_df = train_df.copy()
    test_df = test_df.copy()

    if has_duplicates:
        train_df = train_df.drop_duplicates(subset=id_column)
        test_df = test_df.drop_duplicates(subset=id_column)

    for dataset in (train_df, test_df):
        for col in boolean_cast_columns:
            if col not in dataset.columns:
                continue
            normalized = dataset[col].astype('string').str.strip().str.lower()
            dataset[col] = normalized.map(
                lambda value: pd.NA if pd.isna(value) else value in {'positive', '+'}
            ).astype('boolean')
        cast_true_false_categorical_columns(dataset)

    clip_numeric_outliers(train_df, test_df, outlier_columns)

    for dataset in (train_df, test_df):
        for col in gene_expr_cols:
            if (
                col not in dataset.columns
                or not pd.api.types.is_numeric_dtype(dataset[col])
                or pd.api.types.is_bool_dtype(dataset[col])
            ):
                continue
            dataset[col] = dataset[col].where(dataset[col].abs() > 1.5, 0)

    group_high_cardinality_columns(train_df, test_df, high_cardinality_columns)

    target_columns = [target_col] if isinstance(target_col, str) else list(target_col)
    y_train_raw = train_df[target_columns].copy() if len(target_columns) > 1 else train_df[target_columns[0]].copy()
    y_test_raw = test_df[target_columns].copy() if len(target_columns) > 1 else test_df[target_columns[0]].copy()

    train_cohorts = train_df[cohort_column].copy() if cohort_column in train_df.columns else pd.Series('unknown', index=train_df.index)
    X_train = train_df.drop(columns=target_columns).copy()
    X_test = test_df.drop(columns=target_columns).copy()
    for drop_col in (id_column, cohort_column):
        if drop_col in X_train.columns:
            X_train = X_train.drop(columns=[drop_col])
        if drop_col in X_test.columns:
            X_test = X_test.drop(columns=[drop_col])

    categorical_cols = get_categorical_feature_columns(X_train)
    numerical_cols = get_numeric_feature_columns(X_train)
    numeric_imputers = {col: X_train[col].median() for col in numerical_cols}
    categorical_imputer_value = 'unknown'

    if len(categorical_cols) > 0:
        X_train = normalize_categorical_values(X_train, categorical_cols, categorical_imputer_value)
        X_test = normalize_categorical_values(X_test, categorical_cols.intersection(X_test.columns), categorical_imputer_value)

    if len(numerical_cols) > 0:
        X_train.loc[:, numerical_cols] = X_train[numerical_cols].fillna(numeric_imputers)
        X_test.loc[:, numerical_cols] = X_test[numerical_cols].fillna(numeric_imputers)

    encoder = None
    encoded_col_names = []
    if len(categorical_cols) > 0:
        encoder = create_one_hot_encoder()
        X_train_encoded = encoder.fit_transform(X_train[categorical_cols])
        encoded_col_names = encoder.get_feature_names_out(categorical_cols).tolist()
        X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=encoded_col_names, index=X_train.index)
        X_test_encoded = encoder.transform(X_test[categorical_cols])
        X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=encoded_col_names, index=X_test.index)
        X_train = pd.concat([X_train.drop(columns=categorical_cols), X_train_encoded_df], axis=1)
        X_test = pd.concat([X_test.drop(columns=categorical_cols), X_test_encoded_df], axis=1)

    X_train, batch_correction_metadata = apply_train_only_batch_correction(X_train, train_cohorts, gene_expr_cols)

    decision_scalers = {}
    numeric_transformers = {}
    numeric_cols_after_encoding = get_numeric_feature_columns(X_train)
    for col in numeric_cols_after_encoding:
        series = X_train[col].dropna()
        if series.empty:
            decision_scalers[col] = 'NoScaling'
            continue

        std = series.std()
        if std == 0 or pd.isna(std):
            decision_scalers[col] = 'NoScaling'
            continue

        ks_stat, ks_p = stats.kstest(series, 'norm', args=(series.mean(), std))
        kurtosis = stats.kurtosis(series, fisher=True, bias=False)
        zero_ratio = (series == 0).mean()
        within_unit_interval = series.min() >= 0 and series.max() <= 1

        if zero_ratio >= 0.5 and series.min() >= 0:
            decision = 'Log1pTransform'
        elif within_unit_interval:
            decision = 'MinMaxScaler'
        elif ks_p >= 0.05 and ks_stat <= 0.08 and abs(kurtosis) <= 1.0:
            decision = 'StandardScaler'
        elif ks_p < 0.05 or ks_stat > 0.08 or abs(kurtosis) > 3.0:
            decision = 'RobustScaler'
        else:
            decision = 'StandardScaler'

        decision_scalers[col] = decision

    for col, decision in decision_scalers.items():
        if decision == 'Log1pTransform':
            numeric_transformers[col] = {'type': 'Log1pTransform'}
            X_train[col] = X_train[col].apply(lambda value: np.log1p(value) if pd.notna(value) else value)
            X_test[col] = X_test[col].apply(lambda value: np.log1p(value) if pd.notna(value) else value)
        elif decision == 'MinMaxScaler':
            min_val = X_train[col].min()
            max_val = X_train[col].max()
            numeric_transformers[col] = {'type': 'MinMaxScaler', 'min': float(min_val), 'max': float(max_val)}
            if max_val > min_val:
                X_train[col] = (X_train[col] - min_val) / (max_val - min_val)
                X_test[col] = (X_test[col] - min_val) / (max_val - min_val)
        elif decision == 'StandardScaler':
            mean = X_train[col].mean()
            std = X_train[col].std()
            numeric_transformers[col] = {'type': 'StandardScaler', 'mean': float(mean), 'std': float(std)}
            if std > 0:
                X_train[col] = (X_train[col] - mean) / std
                X_test[col] = (X_test[col] - mean) / std
        elif decision == 'RobustScaler':
            median = X_train[col].median()
            q1 = X_train[col].quantile(0.25)
            q3 = X_train[col].quantile(0.75)
            iqr = q3 - q1
            numeric_transformers[col] = {
                'type': 'RobustScaler',
                'median': float(median),
                'q1': float(q1),
                'q3': float(q3),
                'iqr': float(iqr)
            }
            if iqr > 0:
                X_train[col] = (X_train[col] - median) / iqr
                X_test[col] = (X_test[col] - median) / iqr
        else:
            numeric_transformers[col] = {'type': 'NoScaling'}

    y_train, y_test, target_encoders = encode_targets_for_outer_split(y_train_raw, y_test_raw, task_type)

    preprocessing_artifacts = {
        'decision_scalers': decision_scalers,
        'numeric_transformers': numeric_transformers,
        'categorical_columns': categorical_cols.tolist(),
        'numeric_columns': numeric_cols_after_encoding.tolist(),
        'numeric_imputers': {
            col: float(value) if pd.notna(value) else None
            for col, value in numeric_imputers.items()
        },
        'categorical_imputer_value': categorical_imputer_value,
        'encoded_feature_names': X_train.columns.tolist(),
        'one_hot_encoder': encoder,
        'target_columns': target_columns,
        'target_encoders': target_encoders,
        'split_sizes': {
            'train': int(len(X_train)),
            'validation': 0,
            'test': int(len(X_test)),
        },
        'batch_correction': batch_correction_metadata,
    }

    return X_train, y_train, X_test, y_test, preprocessing_artifacts


def build_model_selection_pipeline(estimator, param_grid, variance_threshold=DEFAULT_VARIANCE_THRESHOLD):
    """Wrap an estimator with fold-local variance selection for GridSearchCV."""
    pipeline = Pipeline([
        ('variance', VarianceThreshold(threshold=variance_threshold)),
        ('model', estimator),
    ])
    prefixed_param_grid = {f'model__{key}': value for key, value in param_grid.items()}
    return pipeline, prefixed_param_grid


def get_variance_selected_frame(best_estimator, X):
    """Return the feature matrix after the fitted variance selector step."""
    variance_step = best_estimator.named_steps['variance']
    selected_array = variance_step.transform(X)
    selected_feature_names = X.columns[variance_step.get_support()].tolist()
    return pd.DataFrame(selected_array, columns=selected_feature_names, index=X.index)


def log_pipeline_shap_artifacts(best_estimator, X, artifact_key):
    """Best-effort SHAP logging for pipelines with an explicit variance-selection step."""
    try:
        selected_frame = get_variance_selected_frame(best_estimator, X)
        shap_explainer = shap.Explainer(best_estimator.named_steps['model'])
        shap_values = shap_explainer(selected_frame)
        log_shap_artifacts(shap_values, selected_frame, artifact_prefix=artifact_key)
    except Exception as exc:
        log_progress(f"Skipping SHAP logging for {artifact_key}: {exc}")
        if mlflow.active_run() is not None:
            mlflow.log_param(f'{artifact_key}_shap_skipped', True)


def run_leave_one_cohort_out_experiment(
    df,
    target_col,
    task_type,
    experiment_name,
    run_name_prefix,
    training_fn,
    dataset_path,
    gene_expr_cols,
    has_duplicates,
    boolean_cast_columns,
    outlier_columns,
    high_cardinality_columns,
    id_column,
    cohort_column=COHORT_COLUMN,
):
    """Run one MLflow training run per held-out cohort using LOCO outer validation."""
    cohorts = get_outer_cohorts(df, cohort_column=cohort_column)
    mlflow.set_experiment(experiment_name)

    for held_out_cohort in cohorts:
        cohort_label = str(held_out_cohort)
        run_name = f"{run_name_prefix}_heldout_{sanitize_run_suffix(cohort_label)}"
        log_progress(f"Preparing {run_name} with held-out cohort '{cohort_label}'")
        outer_train_df = df[df[cohort_column].astype(str) != cohort_label].copy()
        outer_test_df = df[df[cohort_column].astype(str) == cohort_label].copy()
        if outer_train_df.empty or outer_test_df.empty:
            raise ValueError(f"Invalid LOCO split for cohort '{cohort_label}'.")

        X_train, y_train, X_test, y_test, preprocessing_artifacts = preprocess_outer_split(
            outer_train_df,
            outer_test_df,
            target_col,
            gene_expr_cols,
            has_duplicates,
            boolean_cast_columns,
            outlier_columns,
            high_cardinality_columns,
            id_column,
            cohort_column=cohort_column,
            task_type=task_type,
        )
        cv_splits, cv_count, stratification_counts = build_inner_cv_splits(y_train, task_type)

        end_active_mlflow_run()
        try:
            with mlflow.start_run(run_name=run_name):
                target_columns = [target_col] if isinstance(target_col, str) else list(target_col)
                set_run_tags({
                    'use_case': run_name_prefix,
                    'task_type': task_type,
                    'target_columns': ','.join(target_columns),
                    'dataset_path': dataset_path,
                    'outer_strategy': 'leave_one_cohort_out',
                    'inner_strategy': 'stratified_cv',
                    'held_out_cohort': cohort_label,
                })
                mlflow.log_param('held_out_cohort', cohort_label)
                mlflow.log_param('inner_cv_splits', cv_count)
                mlflow.log_dict(stratification_counts, 'model_selection/inner_strata_counts.json')
                log_preprocessing_artifacts(preprocessing_artifacts)
                _, training_summary = training_fn(
                    X_train,
                    y_train,
                    None,
                    None,
                    X_test,
                    y_test,
                    cv=cv_splits,
                )
                log_run_summary(
                    build_run_summary(
                        run_name,
                        task_type,
                        target_columns,
                        preprocessing_artifacts,
                        training_summary,
                        dataset_path,
                        len(df),
                    )
                )
        finally:
            end_active_mlflow_run()


def normalize_categorical_values(df, columns, fill_value="unknown"):
    """Convert categorical inputs to a uniform string representation for encoding."""
    if len(columns) == 0:
        return df

    normalized_df = df.copy()
    normalized_df.loc[:, columns] = normalized_df[columns].astype('string').fillna(fill_value)
    return normalized_df


def cast_true_false_categorical_columns(df):
    """Cast categorical columns with only true/false values to pandas boolean dtype."""
    for col in df.columns:
        if not (
            pd.api.types.is_object_dtype(df[col])
            or pd.api.types.is_string_dtype(df[col])
            or pd.api.types.is_categorical_dtype(df[col])
        ):
            continue

        normalized = df[col].astype('string').str.strip().str.lower()
        non_null_values = set(normalized.dropna().unique())
        if non_null_values and non_null_values.issubset({'true', 'false'}):
            df[col] = normalized.map(
                lambda value: pd.NA if pd.isna(value) else value == 'true'
            ).astype('boolean')

    return df

def run_preprocessing(df, target_col, gene_expr_cols, has_duplicates, boolean_cast_columns, outlier_columns, high_cardinality_columns, id_column):
    """Run preprocessing steps on the dataset."""

    with timed_stage("preprocessing", "preprocessing"):
        return _run_preprocessing_impl(
            df,
            target_col,
            gene_expr_cols,
            has_duplicates,
            boolean_cast_columns,
            outlier_columns,
            high_cardinality_columns,
            id_column,
        )


def _run_preprocessing_impl(df, target_col, gene_expr_cols, has_duplicates, boolean_cast_columns, outlier_columns, high_cardinality_columns, id_column):
    """Implementation for preprocessing steps on the dataset."""

    # Log initial columns
    log_initial_columns(df)

    ####### QC DERIVED QUALITY CHECKS AND PREPROCESSING STEPS #######
    # Example preprocessing steps:
    # - Handle duplicates
    if has_duplicates:
        df = df.drop_duplicates(subset=id_column)


    
    # - Handle boolean columns
    for col in boolean_cast_columns:
        normalized = df[col].astype('string').str.strip().str.lower()
        df[col] = normalized.map(
            lambda value: pd.NA if pd.isna(value) else value in {'positive', '+'}
        ).astype('boolean')

    df = cast_true_false_categorical_columns(df)
    
    # - Handle outliers (simple example using IQR)
    for col in outlier_columns:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df.iloc[(df[col] < lower_bound), df.columns.get_loc(col)] = Q1 # clipping to Q1-Q3
        df.iloc[(df[col] > upper_bound), df.columns.get_loc(col)] = Q3
    
    # filter low signal |z| > 1.8

    for col in gene_expr_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].where(df[col].abs() > 1.5, 0)
        
    
    # - Handle high cardinality categorical columns based on frequency threshold
    #   threshold = median(freq) - 2 * var(freq)
    for col in high_cardinality_columns:
        freq = df[col].value_counts(dropna=False)
        variance = freq.var()
        if pd.isna(variance):
            variance = 0
        threshold = freq.median() - 2 * variance
        rare = freq[freq < threshold].index

        grouped_col = 'geo_grouped' if col == 'geolocation_id' else f'{col}_grouped'
        df[grouped_col] = df[col].replace(rare, 'rare')



     
    ########### TRAIN-VAL-TEST SPLIT, ENCODING, AND SCALING ###########
    if isinstance(target_col, str): # only one target column is supported for now
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Identify categorical and numerical columns
        categorical_cols = get_categorical_feature_columns(X)
        numerical_cols = get_numeric_feature_columns(X)

        X_train, X_holdout, y_train, y_holdout = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )
    
    elif isinstance(target_col, list): # multi-target classification path
        X = df.drop(columns=target_col)
        y = df[target_col].copy()
        categorical_cols = get_categorical_feature_columns(X)
        numerical_cols = get_numeric_feature_columns(X)
        X_train, X_holdout, y_train, y_holdout = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )
    else:
        raise TypeError("target_col must be a string or a list of strings")

    X_val, X_test, y_val, y_test = train_test_split(
        X_holdout,
        y_holdout,
        test_size=0.5,
        random_state=42,
    )

    numeric_imputers = {
        col: X_train[col].median()
        for col in numerical_cols
    }
    categorical_imputer_value = "unknown"

    if len(categorical_cols) > 0:
        X_train = normalize_categorical_values(X_train, categorical_cols, categorical_imputer_value)
        if X_val is not None:
            X_val = normalize_categorical_values(X_val, categorical_cols, categorical_imputer_value)
        if X_test is not None:
            X_test = normalize_categorical_values(X_test, categorical_cols, categorical_imputer_value)

    if len(numerical_cols) > 0:
        X_train.loc[:, numerical_cols] = X_train[numerical_cols].fillna(numeric_imputers)
        if X_val is not None:
            X_val.loc[:, numerical_cols] = X_val[numerical_cols].fillna(numeric_imputers)
        if X_test is not None:
            X_test.loc[:, numerical_cols] = X_test[numerical_cols].fillna(numeric_imputers)


    # TO PREVENT DATA LEAKAGE, TRAIN OHE ON TRAINING DATA AND CHOOSE AND TRAIN SCALERS BASED ON TRAINING DATA ONLY, THEN APPLY TRANSFORMATIONS TO BOTH TRAIN AND TEST SETS
    # Encode categorical variables
    encoder = None
    encoded_col_names = []
    if len(categorical_cols) > 0:
        encoder = create_one_hot_encoder()
        X_encoded = encoder.fit_transform(X_train[categorical_cols])
        encoded_col_names = encoder.get_feature_names_out(categorical_cols).tolist()
        X_encoded_df = pd.DataFrame(X_encoded, columns=encoded_col_names, index=X_train.index)
        X_train = pd.concat([X_train.drop(columns=categorical_cols), X_encoded_df], axis=1)
        if X_val is not None:
            X_encoded_val = encoder.transform(X_val[categorical_cols])
            X_encoded_val_df = pd.DataFrame(X_encoded_val, columns=encoded_col_names, index=X_val.index)
            X_val = pd.concat([X_val.drop(columns=categorical_cols), X_encoded_val_df], axis=1)
        if X_test is not None:
            X_encoded_test = encoder.transform(X_test[categorical_cols])
            X_encoded_test_df = pd.DataFrame(X_encoded_test, columns=encoded_col_names, index=X_test.index)
            X_test = pd.concat([X_test.drop(columns=categorical_cols), X_encoded_test_df], axis=1)

    target_encoders = {}
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.copy()
        y_val = y_val.copy() if y_val is not None else None
        y_test = y_test.copy() if y_test is not None else None
        for col in y_train.columns:
            target_encoder = preprocessing.LabelEncoder()
            y_train_col = y_train[col].astype(str)
            y_train[col] = target_encoder.fit_transform(y_train_col)

            if y_val is not None:
                y_val_col = y_val[col].astype(str)
                unseen_labels = sorted(set(y_val_col.unique()) - set(target_encoder.classes_))
                if unseen_labels:
                    raise ValueError(
                        f"Target column '{col}' contains labels in the validation split that were not seen during training: {unseen_labels}"
                    )
                y_val[col] = target_encoder.transform(y_val_col)

            if y_test is not None:
                y_test_col = y_test[col].astype(str)
                unseen_labels = sorted(set(y_test_col.unique()) - set(target_encoder.classes_))
                if unseen_labels:
                    raise ValueError(
                        f"Target column '{col}' contains labels in the test split that were not seen during training: {unseen_labels}"
                    )
                y_test[col] = target_encoder.transform(y_test_col)

            target_encoders[col] = target_encoder.classes_.tolist()


    # Distribution diagnostics for numeric columns after outlier treatment
    # Rules:
    # - zero-inflated non-negative: Log1pTransform
    # - bounded [0,1]: MinMaxScaler
    # - approximately normal: StandardScaler
    # - heavy tails / non-normal: RobustScaler
    decision_scalers = {}
    numeric_transformers = {}
    numeric_cols = get_numeric_feature_columns(X_train)
    for col in numeric_cols:
        if col not in gene_expr_cols:
        
            series = X_train[col].dropna()

            if series.empty:
                decision_scalers[col] = 'NoScaling'
                print(f"Column: {col}, empty after NA removal -> NoScaling")
                continue

            std = series.std()
            if std == 0 or pd.isna(std):
                decision_scalers[col] = 'NoScaling'
                print(f"Column: {col}, constant feature -> NoScaling")
                continue

            ks_stat, ks_p = stats.kstest(series, 'norm', args=(series.mean(), std))
            kurtosis = stats.kurtosis(series, fisher=True, bias=False)

            zero_ratio = (series == 0).mean()
            within_unit_interval = series.min() >= 0 and series.max() <= 1

            if zero_ratio >= 0.5 and series.min() >= 0:
                decision = 'Log1pTransform'
            elif within_unit_interval:
                decision = 'MinMaxScaler'
            elif ks_p >= 0.05 and ks_stat <= 0.08 and abs(kurtosis) <= 1.0:
                decision = 'StandardScaler'
            elif ks_p < 0.05 or ks_stat > 0.08 or abs(kurtosis) > 3.0:
                decision = 'RobustScaler'
            else:
                decision = 'StandardScaler'

            decision_scalers[col] = decision
            print(
                f"Column: {col}, KS stat: {ks_stat:.4f}, KS p-value: {ks_p:.4f}, "
                f"Kurtosis: {kurtosis:.4f}, Decision: {decision}"
            )

        # implement the actual transformations based on the decisions
        for col, decision in decision_scalers.items():
            if decision == 'Log1pTransform':
                numeric_transformers[col] = {'type': 'Log1pTransform'}
                X_train[col] = X_train[col].apply(lambda x: np.log1p(x) if pd.notna(x) else x)
                if X_val is not None:
                    X_val[col] = X_val[col].apply(lambda x: np.log1p(x) if pd.notna(x) else x)
                if X_test is not None:
                    X_test[col] = X_test[col].apply(lambda x: np.log1p(x) if pd.notna(x) else x)
            elif decision == 'MinMaxScaler':
                min_val = X_train[col].min()
                max_val = X_train[col].max()
                numeric_transformers[col] = {
                    'type': 'MinMaxScaler',
                    'min': float(min_val),
                    'max': float(max_val)
                }
                if max_val > min_val:
                    X_train[col] = (X_train[col] - min_val) / (max_val - min_val)
                    if X_val is not None:
                        X_val[col] = (X_val[col] - min_val) / (max_val - min_val)
                    if X_test is not None:
                        X_test[col] = (X_test[col] - min_val) / (max_val - min_val)
            elif decision == 'StandardScaler':
                mean = X_train[col].mean()
                std = X_train[col].std()
                numeric_transformers[col] = {
                    'type': 'StandardScaler',
                    'mean': float(mean),
                    'std': float(std)
                }
                if std > 0:
                    X_train[col] = (X_train[col] - mean) / std
                    if X_val is not None:
                        X_val[col] = (X_val[col] - mean) / std
                    if X_test is not None:
                        X_test[col] = (X_test[col] - mean) / std
            elif decision == 'RobustScaler':
                median = X_train[col].median()
                q1 = X_train[col].quantile(0.25)
                q3 = X_train[col].quantile(0.75)
                iqr = q3 - q1
                numeric_transformers[col] = {
                    'type': 'RobustScaler',
                    'median': float(median),
                    'q1': float(q1),
                    'q3': float(q3),
                    'iqr': float(iqr)
                }
                if iqr > 0:
                    X_train[col] = (X_train[col] - median) / iqr
                    if X_val is not None:
                        X_val[col] = (X_val[col] - median) / iqr
                    if X_test is not None:
                        X_test[col] = (X_test[col] - median) / iqr
            else:
                numeric_transformers[col] = {'type': 'NoScaling'}


    preprocessing_artifacts = {
        'decision_scalers': decision_scalers,
        'numeric_transformers': numeric_transformers,
        'categorical_columns': categorical_cols.tolist(),
        'numeric_columns': numeric_cols.tolist(),
        'numeric_imputers': {
            col: float(value) if pd.notna(value) else None
            for col, value in numeric_imputers.items()
        },
        'categorical_imputer_value': categorical_imputer_value,
        'encoded_feature_names': encoded_col_names,
        'one_hot_encoder': encoder,
        'target_columns': y_train.columns.tolist() if isinstance(y_train, pd.DataFrame) else [target_col],
        'target_encoders': target_encoders,
        'split_sizes': {
            'train': int(len(X_train)),
            'validation': int(len(X_val)),
            'test': int(len(X_test)),
        }
    }

    return X_train, y_train, X_val, y_val, X_test, y_test, preprocessing_artifacts

def run_training_regressor(X_train, y_train, X_val, y_val, X_test, y_test, cv=None):
    """Run CV model selection for regression and log evaluation."""
    # This function can be implemented similarly to the multi-target classifier but using regression models and appropriate metrics like R^2, MAE, RMSE, etc.
    cv = cv if cv is not None else 5
    cv_count = len(cv) if isinstance(cv, list) else cv
    scoring = 'r2'
    model_candidates = {
        'linear_regression': {
            'estimator': LinearRegression(),
            'param_grid': {}
        },
        'logistic_regression': {
            'estimator': LogisticRegression(random_state=42, max_iter=1000),
            'param_grid': {
                'C': [0.1, 1, 10],
                'penalty': ['l2'],
                'solver': ['lbfgs']
            }
        },  
        'random_forest': {
            'estimator': RandomForestRegressor(random_state=42, n_jobs=-1),
            'param_grid': {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        },
        # 'svm': {
        #     'estimator': SVR(random_state=42),
        #     'param_grid': {
        #         'C': [0.1, 1, 10],
        #         'kernel': ['linear', 'rbf'],
        #         'gamma': ['scale', 'auto']
        #     }
        # }
    }
    best_model_name = None
    best_search = None
    best_cv_score = float('-inf')
    val_metrics = {}
    test_metrics = {}
    for model_name, config in model_candidates.items():
        log_progress(f"Model selection candidate '{model_name}' started for regressor")
        with timed_stage(f"regressor_grid_search_{model_name}", f"regressor_grid_search_{model_name}"):
            pipeline, param_grid = build_model_selection_pipeline(config['estimator'], config['param_grid'])
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring=scoring,
                cv=cv,
                n_jobs=-1,
                refit=True,
                verbose=2
            )
            search.fit(X_train, y_train)
        mlflow.log_metric(f'{model_name}_best_cv_{scoring}', float(search.best_score_))
        mlflow.log_dict(
            {
                'best_params': search.best_params_,
                'best_cv_score': float(search.best_score_),
                'scoring': scoring,
                'cv_splits': cv_count
            },
            f'model_selection/{model_name}_summary.json'
        )
        if search.best_score_ > best_cv_score:
            best_cv_score = float(search.best_score_)
            best_model_name = model_name
            best_search = search


    mlflow.log_param('selected_model', best_model_name)
    mlflow.log_metric(f'selected_model_best_cv_{scoring}', best_cv_score)
    if X_val is not None and y_val is not None:
        y_val_pred = best_search.predict(X_val)
        val_metrics = {
            'r2': float(r2_score(y_val, y_val_pred)),
            'mae': float(mean_absolute_error(y_val, y_val_pred)),
            'rmse': float(mean_squared_error(y_val, y_val_pred, squared=False))
        }
        mlflow.log_metric('val_r2', val_metrics['r2'])
        mlflow.log_metric('val_mae', val_metrics['mae'])
        mlflow.log_metric('val_rmse', val_metrics['rmse'])
    if X_test is not None and y_test is not None:
        y_pred = best_search.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        test_metrics = {
            'r2': float(r2),
            'mae': float(mae),
            'rmse': float(rmse)
        }
        mlflow.log_metric('test_r2', float(r2))
        mlflow.log_metric('test_mae', float(mae))
        mlflow.log_metric('test_rmse', float(rmse))
    log_sklearn_model_artifact(best_search.best_estimator_, TRAINED_MODEL_ARTIFACT)
    
    # SHAP
    with timed_stage("regressor_shap", "regressor_shap"):
        log_pipeline_shap_artifacts(best_search.best_estimator_, X_test, 'model_selection')

    return best_search.best_estimator_, {
        'best_model_name': best_model_name,
        'best_cv_score': best_cv_score,
        'scoring': scoring,
        'validation_metrics': val_metrics,
        'test_metrics': test_metrics,
    }




def run_training_multitarget_classifier(X_train, y_train, X_val, y_val, X_test, y_test, cv=None):
    """Run CV model selection for multi-target classification and log per-target evaluation."""
    if not isinstance(y_train, pd.DataFrame):
        raise TypeError("y_train must be a DataFrame for multi-target training.")

    cv = cv if cv is not None else 5
    cv_count = len(cv) if isinstance(cv, list) else cv
    scoring = 'f1_weighted'
    model_candidates = {
        'random_forest': {
            'estimator': MultiOutputClassifier(
                RandomForestClassifier(random_state=42, n_jobs=-1),
                n_jobs=-1
            ),
            'param_grid': {
                'estimator__n_estimators': [100, 200],
                'estimator__max_depth': [None, 10, 20],
                'estimator__min_samples_split': [2, 5],
                'estimator__min_samples_leaf': [1, 2],
                'estimator__class_weight': [None, 'balanced']
            }
        },
        # 'svm': {
        #     'estimator': MultiOutputClassifier(
        #         SVC(random_state=42),
        #         n_jobs=-1
        #     ),
        #     'param_grid': {
        #         'estimator__C': [0.1, 1, 10],
        #         'estimator__kernel': ['linear', 'rbf'],
        #         'estimator__gamma': ['scale', 'auto'],
        #         'estimator__class_weight': [None, 'balanced']
        #     }
        # }
    }

    if XGBClassifier is not None:
        model_candidates['xgboost'] = {
            'estimator': MultiOutputClassifier(
                XGBClassifier(
                    random_state=42,
                    eval_metric='mlogloss',
                    n_jobs=-1
                ),
                n_jobs=-1
            ),
            'param_grid': {
                'estimator__n_estimators': [100, 200],
                'estimator__max_depth': [3, 6],
                'estimator__learning_rate': [0.05, 0.1],
                'estimator__subsample': [0.8, 1.0],
                'estimator__colsample_bytree': [0.8, 1.0]
            }
        }
    else:
        mlflow.log_param('xgboost_available', False)

    best_model_name = None
    best_search = None
    best_cv_score = float('-inf')
    model_summaries = {}
    validation_summary = {}
    test_summary = {}

    for model_name, config in model_candidates.items():
        log_progress(f"Model selection candidate '{model_name}' started for multi-target classifier")
        with timed_stage(f"multitarget_grid_search_{model_name}", f"multitarget_grid_search_{model_name}"):
            pipeline, param_grid = build_model_selection_pipeline(config['estimator'], config['param_grid'])
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring=scoring,
                cv=cv,
                n_jobs=-1,
                refit=True,
                verbose=2
            )
            search.fit(X_train, y_train)

        model_summaries[model_name] = {
            'best_params': search.best_params_,
            'best_cv_score': float(search.best_score_)
        }
        mlflow.log_metric(f'{model_name}_best_cv_{scoring}', float(search.best_score_))
        mlflow.log_dict(
            {
                'best_params': search.best_params_,
                'best_cv_score': float(search.best_score_),
                'scoring': scoring,
                'cv_splits': cv_count
            },
            f'model_selection/{model_name}_summary.json'
        )

        if search.best_score_ > best_cv_score:
            best_cv_score = float(search.best_score_)
            best_model_name = model_name
            best_search = search

    if best_search is None:
        raise RuntimeError('No model candidates were available for cross-validation.')

    mlflow.log_param('selected_model', best_model_name)
    mlflow.log_metric(f'selected_model_best_cv_{scoring}', best_cv_score)
    mlflow.log_dict(model_summaries, 'model_selection/all_model_summaries.json')

    if X_val is not None and y_val is not None:
        y_val_pred = pd.DataFrame(best_search.predict(X_val), columns=y_train.columns, index=y_val.index)
        val_exact_match_accuracy = float((y_val_pred == y_val).all(axis=1).mean())
        mlflow.log_metric('val_exact_match_accuracy', val_exact_match_accuracy)
        val_target_f1_scores = []
        for target_name in y_train.columns:
            safe_target_name = target_name.replace(' ', '_')
            target_accuracy = accuracy_score(y_val[target_name], y_val_pred[target_name])
            target_f1 = f1_score(y_val[target_name], y_val_pred[target_name], average='weighted')
            val_target_f1_scores.append(float(target_f1))
            mlflow.log_metric(f'val_{safe_target_name}_accuracy', float(target_accuracy))
            mlflow.log_metric(f'val_{safe_target_name}_f1_weighted', float(target_f1))
        val_mean_target_f1_weighted = float(np.mean(val_target_f1_scores))
        mlflow.log_metric('val_mean_target_f1_weighted', val_mean_target_f1_weighted)
        validation_summary = {
            'exact_match_accuracy': val_exact_match_accuracy,
            'mean_target_f1_weighted': val_mean_target_f1_weighted,
        }

    if X_test is not None and y_test is not None:
        y_pred = pd.DataFrame(best_search.predict(X_test), columns=y_train.columns, index=y_test.index)
        exact_match_accuracy = float((y_pred == y_test).all(axis=1).mean())
        mlflow.log_metric('test_exact_match_accuracy', exact_match_accuracy)

        per_target_metrics = {}
        per_target_reports = {}
        per_target_f1_scores = []
        for target_name in y_train.columns:
            safe_target_name = target_name.replace(' ', '_')
            target_accuracy = accuracy_score(y_test[target_name], y_pred[target_name])
            target_f1 = f1_score(y_test[target_name], y_pred[target_name], average='weighted')
            per_target_metrics[target_name] = {
                'accuracy': float(target_accuracy),
                'f1_weighted': float(target_f1)
            }
            per_target_reports[target_name] = classification_report(
                y_test[target_name],
                y_pred[target_name],
                output_dict=True,
                zero_division=0
            )
            per_target_f1_scores.append(float(target_f1))
            mlflow.log_metric(f'{safe_target_name}_accuracy', float(target_accuracy))
            mlflow.log_metric(f'{safe_target_name}_f1_weighted', float(target_f1))

        test_mean_target_f1_weighted = float(np.mean(per_target_f1_scores))
        mlflow.log_metric('test_mean_target_f1_weighted', test_mean_target_f1_weighted)
        mlflow.log_dict(per_target_metrics, 'model_selection/per_target_metrics.json')
        mlflow.log_dict(per_target_reports, 'model_selection/per_target_classification_reports.json')
        test_summary = {
            'exact_match_accuracy': exact_match_accuracy,
            'mean_target_f1_weighted': test_mean_target_f1_weighted,
            'per_target_metrics': per_target_metrics,
        }

    # shap explainer can be added here for the best model if needed, but it may require additional handling based on the model type and data size
    with timed_stage("multitarget_shap", "multitarget_shap"):
        log_pipeline_shap_artifacts(best_search.best_estimator_, X_test, 'model_selection')

    log_sklearn_model_artifact(best_search.best_estimator_, TRAINED_MODEL_ARTIFACT)
    return best_search.best_estimator_, {
        'best_model_name': best_model_name,
        'best_cv_score': best_cv_score,
        'scoring': scoring,
        'candidate_models': model_summaries,
        'validation_metrics': validation_summary,
        'test_metrics': test_summary,
    }


def run_training_multilabel_classifier(X_train, y_train, X_val, y_val, X_test, y_test, cv=None):
    """Run CV model selection for multilabel classification and log multilabel metrics."""
    if not isinstance(y_train, pd.DataFrame):
        raise TypeError("y_train must be a DataFrame for multilabel training.")

    cv = cv if cv is not None else 5
    cv_count = len(cv) if isinstance(cv, list) else cv
    scoring = 'f1_samples'
    model_candidates = {
        'random_forest': {
            'estimator': OneVsRestClassifier(
                RandomForestClassifier(random_state=42, n_jobs=-1),
                n_jobs=-1
            ),
            'param_grid': {
                'estimator__n_estimators': [100, 200],
                'estimator__max_depth': [None, 10, 20],
                'estimator__min_samples_split': [2, 5],
                'estimator__min_samples_leaf': [1, 2],
                'estimator__class_weight': [None, 'balanced']
            }
        },
        # 'svm': {
        #     'estimator': OneVsRestClassifier(
        #         SVC(random_state=42),
        #         n_jobs=-1
        #     ),
        #     'param_grid': {
        #         'estimator__C': [0.1, 1, 10],
        #         'estimator__kernel': ['linear', 'rbf'],
        #         'estimator__gamma': ['scale', 'auto'],
        #         'estimator__class_weight': [None, 'balanced']
        #     }
        # }
    }

    if XGBClassifier is not None:
        model_candidates['xgboost'] = {
            'estimator': OneVsRestClassifier(
                XGBClassifier(
                    random_state=42,
                    eval_metric='logloss',
                    n_jobs=-1
                ),
                n_jobs=-1
            ),
            'param_grid': {
                'estimator__n_estimators': [100, 200],
                'estimator__max_depth': [3, 6],
                'estimator__learning_rate': [0.05, 0.1],
                'estimator__subsample': [0.8, 1.0],
                'estimator__colsample_bytree': [0.8, 1.0]
            }
        }
    else:
        mlflow.log_param('xgboost_available', False)

    best_model_name = None
    best_search = None
    best_cv_score = float('-inf')
    model_summaries = {}
    validation_summary = {}
    test_summary = {}

    for model_name, config in model_candidates.items():
        log_progress(f"Model selection candidate '{model_name}' started for multilabel classifier")
        with timed_stage(f"multilabel_grid_search_{model_name}", f"multilabel_grid_search_{model_name}"):
            pipeline, param_grid = build_model_selection_pipeline(config['estimator'], config['param_grid'])
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring=scoring,
                cv=cv,
                n_jobs=-1,
                refit=True,
                verbose=2
            )
            search.fit(X_train, y_train)

        model_summaries[model_name] = {
            'best_params': search.best_params_,
            'best_cv_score': float(search.best_score_)
        }
        mlflow.log_metric(f'{model_name}_best_cv_{scoring}', float(search.best_score_))
        mlflow.log_dict(
            {
                'best_params': search.best_params_,
                'best_cv_score': float(search.best_score_),
                'scoring': scoring,
                'cv_splits': cv_count
            },
            f'model_selection/{model_name}_summary.json'
        )

        if search.best_score_ > best_cv_score:
            best_cv_score = float(search.best_score_)
            best_model_name = model_name
            best_search = search

    if best_search is None:
        raise RuntimeError('No model candidates were available for cross-validation.')

    mlflow.log_param('selected_model', best_model_name)
    mlflow.log_metric(f'selected_model_best_cv_{scoring}', best_cv_score)
    mlflow.log_dict(model_summaries, 'model_selection/all_model_summaries.json')

    if X_val is not None and y_val is not None:
        y_val_pred = pd.DataFrame(best_search.predict(X_val), columns=y_train.columns, index=y_val.index)
        val_subset_accuracy = float(accuracy_score(y_val, y_val_pred))
        val_f1_samples = float(f1_score(y_val, y_val_pred, average='samples', zero_division=0))
        val_f1_macro = float(f1_score(y_val, y_val_pred, average='macro', zero_division=0))
        mlflow.log_metric('val_subset_accuracy', val_subset_accuracy)
        mlflow.log_metric('val_f1_samples', val_f1_samples)
        mlflow.log_metric('val_f1_macro', val_f1_macro)
        validation_summary = {
            'subset_accuracy': val_subset_accuracy,
            'f1_samples': val_f1_samples,
            'f1_macro': val_f1_macro,
        }

    if X_test is not None and y_test is not None:
        y_pred = pd.DataFrame(best_search.predict(X_test), columns=y_train.columns, index=y_test.index)
        test_subset_accuracy = float(accuracy_score(y_test, y_pred))
        test_f1_samples = float(f1_score(y_test, y_pred, average='samples', zero_division=0))
        test_f1_macro = float(f1_score(y_test, y_pred, average='macro', zero_division=0))
        mlflow.log_metric('test_subset_accuracy', test_subset_accuracy)
        mlflow.log_metric('test_f1_samples', test_f1_samples)
        mlflow.log_metric('test_f1_macro', test_f1_macro)

        per_target_metrics = {}
        per_target_reports = {}
        for target_name in y_train.columns:
            safe_target_name = target_name.replace(' ', '_')
            target_accuracy = accuracy_score(y_test[target_name], y_pred[target_name])
            target_f1 = f1_score(y_test[target_name], y_pred[target_name], average='binary', zero_division=0)
            per_target_metrics[target_name] = {
                'accuracy': float(target_accuracy),
                'f1_binary': float(target_f1)
            }
            per_target_reports[target_name] = classification_report(
                y_test[target_name],
                y_pred[target_name],
                output_dict=True,
                zero_division=0
            )
            mlflow.log_metric(f'{safe_target_name}_accuracy', float(target_accuracy))
            mlflow.log_metric(f'{safe_target_name}_f1_binary', float(target_f1))

        mlflow.log_dict(per_target_metrics, 'model_selection/per_target_metrics.json')
        mlflow.log_dict(per_target_reports, 'model_selection/per_target_classification_reports.json')
        test_summary = {
            'subset_accuracy': test_subset_accuracy,
            'f1_samples': test_f1_samples,
            'f1_macro': test_f1_macro,
            'per_target_metrics': per_target_metrics,
        }

    with timed_stage("multilabel_shap", "multilabel_shap"):
        log_pipeline_shap_artifacts(best_search.best_estimator_, X_test, 'model_selection')

    log_sklearn_model_artifact(best_search.best_estimator_, TRAINED_MODEL_ARTIFACT)
    return best_search.best_estimator_, {
        'best_model_name': best_model_name,
        'best_cv_score': best_cv_score,
        'scoring': scoring,
        'candidate_models': model_summaries,
        'validation_metrics': validation_summary,
        'test_metrics': test_summary,
    }


def run_training_multiclass_classifier(X_train, y_train, X_val, y_val, X_test, y_test, cv=None):
    """Train diagnosis targets using the existing one-vs-rest multi-output path."""
    return run_training_multitarget_classifier(X_train, y_train, X_val, y_val, X_test, y_test, cv=cv)

def main(DATASET_PATH, id_column, non_gene_cols, therapeutic_target_columns, orctree_target_column, pronostic_target_column, output_report_path=None):

    # if output_report_path is None:
    # Run QC and get derived quality checks
    with timed_stage("qc", "qc"):
        has_duplicates, boolean_cast_columns, outlier_columns, high_cardinality_columns = run_qc(
            id_column=id_column,
            main_file_path=DATASET_PATH,
            do_report=False,
            output_report_path="",
            show_plots=False,
        )

    ###################
    # USE CASE 1 - MULTICLASSIFIER WITH MULTIPLE TARGET COLUMNS
    ###################
    # Load dataset
    df_nofilter = pd.read_csv(DATASET_PATH)
    gene_cols = [col for col in df_nofilter.columns if col not in non_gene_cols]


    df = df_nofilter.copy()
    df = df[(df['death_from_cancer'] != 'Died of Other Causes')| (df["overall_survival"]==1)] # ensure we are modeling breast cancer specific survival and not overall survival, which would be a different use case with different target variable definition and modeling approach
    # Run preprocessing
    # normalize therapeutic targets for true multi-target classification
    for col in therapeutic_target_columns:
        if col in df.columns and col != "type_of_breast_surgery":
            df[col] = df[col].astype(str).str.strip().str.lower().map(
                lambda x: pd.NA if pd.isna(x) else x in {'yes', 'positive', '+', '1'}
            ).astype('boolean')
    df_surgery = pd.get_dummies(df["type_of_breast_surgery"], prefix="breast_surgery")
    df = pd.concat([df, df_surgery], axis=1).drop(columns=["type_of_breast_surgery"])
    therapeutic_final_targets = list(df_surgery.columns) + [col for col in therapeutic_target_columns if col != "type_of_breast_surgery"]
    run_leave_one_cohort_out_experiment(
        df=df,
        target_col=therapeutic_final_targets,
        task_type='multilabel_classification',
        experiment_name='METABRIC_UC_Plausible_Therapy',
        run_name_prefix='therapy_multilabel',
        training_fn=run_training_multilabel_classifier,
        dataset_path=DATASET_PATH,
        gene_expr_cols=gene_cols,
        has_duplicates=has_duplicates,
        boolean_cast_columns=boolean_cast_columns,
        outlier_columns=outlier_columns,
        high_cardinality_columns=high_cardinality_columns,
        id_column=id_column,
    )

    
    ###################
    # USE CASE 2 - REGRESSOR WITH SINGLE TARGET COLUMN OVERALL SURVIVAL TIME
    ################---
    # Load dataset
    df = pd.read_csv(DATASET_PATH)
    df = df[df['overall_survival'] == 0] # filter to only patients who died to ensure we are modeling survival time and not a mix of survival time and censoring, which would be a different use case with different target variable definition and modeling approach
    run_leave_one_cohort_out_experiment(
        df=df,
        target_col=pronostic_target_column,
        task_type='regression',
        experiment_name='METABRIC_UC_Plausible_Survival_Time',
        run_name_prefix='survival_regression',
        training_fn=run_training_regressor,
        dataset_path=DATASET_PATH,
        gene_expr_cols=gene_cols,
        has_duplicates=has_duplicates,
        boolean_cast_columns=boolean_cast_columns,
        outlier_columns=outlier_columns,
        high_cardinality_columns=high_cardinality_columns,
        id_column=id_column,
    )

        
    ###################
    # USE CASE 3 - CLASSIFIER ORCTREE CODES DIAGNOSIS
    ###################
    # Load dataset
    df = pd.read_csv(DATASET_PATH)
    # normalize orctree target for multi-class classification
    end_active_mlflow_run()
    mlflow.set_experiment("METABRIC_UC_Plausible_Diagnosis")
    log_progress("Preparing diagnosis multiclass experiment")
    df[orctree_target_column] = df[orctree_target_column].astype(str).str.strip().str.lower()
    df_orctree = pd.get_dummies(df[orctree_target_column], prefix="orctree")
    df = pd.concat([df, df_orctree], axis=1).drop(columns=[orctree_target_column])

    run_leave_one_cohort_out_experiment(
        df=df,
        target_col=df_orctree.columns.tolist(),
        task_type='multiclass_classification',
        experiment_name='METABRIC_UC_Plausible_Diagnosis',
        run_name_prefix='diagnosis_multiclass',
        training_fn=run_training_multiclass_classifier,
        dataset_path=DATASET_PATH,
        gene_expr_cols=gene_cols,
        has_duplicates=has_duplicates,
        boolean_cast_columns=boolean_cast_columns,
        outlier_columns=outlier_columns,
        high_cardinality_columns=high_cardinality_columns,
        id_column=id_column,
    )

    return


if "__init__" == __name__:
    argv = argparse.ArgumentParser()
    argv.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset CSV file")
    argv.add_argument("--id_column", type=str, required=True, help="Name of the ID column in the dataset")
    argv.add_argument("--therapeutic_target_columns", type=str, nargs='*', default=[], help="List of columns as therapeutic targets")
    argv.add_argument("--orctree_target_column", type=str, required=True, help="Diagnostic target column")
    argv.add_argument("--pronostic_target_column", type=str, required=True, help="Prognostic target column")
    argv.add_argument("--non_gene_expression_columns", type=str, nargs='*', default=[], help="List of non-gene expression columns to include in EDA")
    args = argv.parse_args()
    main(
        DATASET_PATH=args.dataset_path,
        id_column=args.id_column,
        non_gene_cols=args.non_gene_expression_columns,
        therapeutic_target_columns=args.therapeutic_target_columns,
        orctree_target_column=args.orctree_target_column,
        pronostic_target_column=args.pronostic_target_column,
    )
