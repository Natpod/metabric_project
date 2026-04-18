import sklearn.preprocessing as preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
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

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


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
        mlflow.sklearn.log_model(
            preprocessing_artifacts["one_hot_encoder"],
            artifact_path=f"{artifact_path}/one_hot_encoder"
        )

def log_initial_columns(df):
    """Log the initial columns of the DataFrame to MLflow."""
    initial_columns = df.columns.tolist()
    mlflow.log_dict({"initial_columns": initial_columns}, "initial_columns.json")

def run_preprocessing(df, target_col, has_duplicates, boolean_cast_columns, outlier_columns, high_cardinality_columns, id_column):
    """Run preprocessing steps on the dataset."""

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
    
    # - Handle outliers (simple example using IQR)
    for col in outlier_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df.iloc[(df[col] < lower_bound), df.columns.get_loc(col)] = Q1 # clipping to Q1-Q3
        df.iloc[(df[col] > upper_bound), df.columns.get_loc(col)] = Q3
    
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


     
    ########### TRAIN-TEST SPLIT, ENCODING, AND SCALING ###########
    if isinstance(target_col, str): # only one target column is supported for now
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Identify categorical and numerical columns
        categorical_cols = X.select_dtypes(include=['object', 'boolean']).columns
        numerical_cols = X.select_dtypes(include=['number']).columns

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    elif isinstance(target_col, list): # multi-target classification path
        X = df.drop(columns=target_col)
        y = df[target_col].copy()
        categorical_cols = X.select_dtypes(include=['object', 'boolean']).columns
        numerical_cols = X.select_dtypes(include=['number']).columns
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        raise TypeError("target_col must be a string or a list of strings")


    # TO PREVENT DATA LEAKAGE, TRAIN OHE ON TRAINING DATA AND CHOOSE AND TRAIN SCALERS BASED ON TRAINING DATA ONLY, THEN APPLY TRANSFORMATIONS TO BOTH TRAIN AND TEST SETS
    # Encode categorical variables
    encoder = None
    encoded_col_names = []
    if len(categorical_cols) > 0:
        encoder = preprocessing.OneHotEncoder(sparse=False, handle_unknown='ignore')
        X_encoded = encoder.fit_transform(X_train[categorical_cols])
        encoded_col_names = encoder.get_feature_names_out(categorical_cols).tolist()
        X_encoded_df = pd.DataFrame(X_encoded, columns=encoded_col_names, index=X_train.index)
        X_train = pd.concat([X_train.drop(columns=categorical_cols), X_encoded_df], axis=1)
        if X_test is not None:
            X_encoded_test = encoder.transform(X_test[categorical_cols])
            X_encoded_test_df = pd.DataFrame(X_encoded_test, columns=encoded_col_names, index=X_test.index)
            X_test = pd.concat([X_test.drop(columns=categorical_cols), X_encoded_test_df], axis=1)

    target_encoders = {}
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.copy()
        y_test = y_test.copy() if y_test is not None else None
        for col in y_train.columns:
            target_encoder = preprocessing.LabelEncoder()
            y_train_col = y_train[col].astype(str)
            y_train[col] = target_encoder.fit_transform(y_train_col)

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
    numeric_cols = X_train.select_dtypes(include='number').columns
    for col in numeric_cols:
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
                if X_test is not None:
                    X_test[col] = (X_test[col] - median) / iqr
        else:
            numeric_transformers[col] = {'type': 'NoScaling'}

    preprocessing_artifacts = {
        'decision_scalers': decision_scalers,
        'numeric_transformers': numeric_transformers,
        'categorical_columns': categorical_cols.tolist(),
        'numeric_columns': numeric_cols.tolist(),
        'encoded_feature_names': encoded_col_names,
        'one_hot_encoder': encoder,
        'target_columns': y_train.columns.tolist() if isinstance(y_train, pd.DataFrame) else [target_col],
        'target_encoders': target_encoders
    }

    return X_train, y_train, X_test, y_test, preprocessing_artifacts

def run_training_regressor(X_train, y_train, X_test, y_test):
    """Run CV model selection for regression and log evaluation."""
    # This function can be implemented similarly to the multi-target classifier but using regression models and appropriate metrics like R^2, MAE, RMSE, etc.
    cv=5
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
        'svm': {
            'estimator': SVR(random_state=42),
            'param_grid': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
        }
    }
    best_model_name = None
    best_search = None
    best_cv_score = float('-inf')
    for model_name, config in model_candidates.items():
        search = GridSearchCV(
            estimator=config['estimator'],
            param_grid=config['param_grid'],
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            refit=True
        )
        search.fit(X_train, y_train)
        mlflow.log_metric(f'{model_name}_best_cv_{scoring}', float(search.best_score_))
        mlflow.log_dict(
            {
                'best_params': search.best_params_,
                'best_cv_score': float(search.best_score_),
                'scoring': scoring,
                'cv': cv
            },
            f'model_selection/{model_name}_summary.json'
        )
        if search.best_score_ > best_cv_score:
            best_cv_score = float(search.best_score_)
            best_model_name = model_name
            best_search = search


    mlflow.log_param('selected_model', best_model_name)
    mlflow.log_metric(f'selected_model_best_cv_{scoring}', best_cv_score)
    if X_test is not None and y_test is not None:
        y_pred = best_search.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mlflow.log_metric('test_r2', float(r2))
        mlflow.log_metric('test_mae', float(mae))
        mlflow.log_metric('test_rmse', float(rmse))
    mlflow.sklearn.log_model(best_search.best_estimator_, artifact_path=f'models/{best_model_name}')
    
    # SHAP
    shap_explainer = shap.Explainer(best_search.best_estimator_)
    shap_values = shap_explainer(X_test)
    mlflow.log_dict(shap_values.values.tolist(), 'model_selection/shap_values.json')
    shap.summary_plot(shap_values, X_test, show=False)
    mlflow.log_figure(plt.gcf(), 'model_selection/shap_summary_plot.png')
    plt.close()

    return best_search.best_estimator_, {
        'best_model_name': best_model_name,
        'best_cv_score': best_cv_score
    }




def run_training_multitarget_classifier(X_train, y_train, X_test, y_test):
    """Run CV model selection for multi-target classification and log per-target evaluation."""
    if not isinstance(y_train, pd.DataFrame):
        raise TypeError("y_train must be a DataFrame for multi-target training.")

    cv = 5
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
        'svm': {
            'estimator': MultiOutputClassifier(
                SVC(random_state=42),
                n_jobs=-1
            ),
            'param_grid': {
                'estimator__C': [0.1, 1, 10],
                'estimator__kernel': ['linear', 'rbf'],
                'estimator__gamma': ['scale', 'auto'],
                'estimator__class_weight': [None, 'balanced']
            }
        }
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

    for model_name, config in model_candidates.items():
        search = GridSearchCV(
            estimator=config['estimator'],
            param_grid=config['param_grid'],
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            refit=True
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
                'cv': cv
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

        mlflow.log_metric('test_mean_target_f1_weighted', float(np.mean(per_target_f1_scores)))
        mlflow.log_dict(per_target_metrics, 'model_selection/per_target_metrics.json')
        mlflow.log_dict(per_target_reports, 'model_selection/per_target_classification_reports.json')

    # shap explainer can be added here for the best model if needed, but it may require additional handling based on the model type and data size
    shap_explainer = shap.Explainer(best_search.best_estimator_)
    shap_values = shap_explainer(X_test)
    mlflow.log_dict(shap_values.values.tolist(), 'model_selection/shap_values.json')
    shap.summary_plot(shap_values, X_test, show=False)
    mlflow.log_figure(plt.gcf(), 'model_selection/shap_summary_plot.png')
    plt.close()

    mlflow.sklearn.log_model(best_search.best_estimator_, artifact_path=f'models/{best_model_name}')
    return best_search.best_estimator_, model_summaries

def main(DATASET_PATH, id_column, therapeutic_target_columns, orctree_target_column, pronostic_target_column):


    # Run QC and get derived quality checks
    run_qc(
        id_column=id_column,
        main_file_path=DATASET_PATH,
        do_report=True,
        output_report_path=output_report_path,
        show_plots=False,
    )

    ###################
    # USE CASE 1 - MULTICLASSIFIER WITH MULTIPLE TARGET COLUMNS
    ###################
    # Load dataset
    df = pd.read_csv(DATASET_PATH)
    # Run preprocessing
    # normalize therapeutic targets for true multi-target classification
    for col in therapeutic_targets:
        if col in df.columns and col != "type_of_breast_surgery":
            df[col] = df[col].astype(str).str.strip().str.lower().map(
                lambda x: pd.NA if pd.isna(x) else x in {'yes', 'positive', '+', '1'}
            ).astype('boolean')
    df_surgery = pd.get_dummies(df["type_of_breast_surgery"], prefix="breast_surgery")
    df = pd.concat([df, df_surgery], axis=1).drop(columns=["type_of_breast_surgery"])
    therapeutic_final_targets = list(df_surgery.columns) + [col for col in therapeutic_targets if col != "type_of_breast_surgery"]
    
    mlflow.set_experiment("METABRIC_UC_Plausible_Therapy")
    # preprocess
    X_train, y_train, X_test, y_test, preprocessing_artifacts = run_preprocessing(
        df,
        therapeutic_final_targets,
        has_duplicates,
        boolean_cast_columns,
        outlier_columns,
        high_cardinality_columns,
        id_column
    )
    with mlflow.start_run():
        log_preprocessing_artifacts(preprocessing_artifacts)
        run_training_multitarget_classifier(X_train, y_train, X_test, y_test)

    
    ###################
    # USE CASE 2 - REGRESSOR WITH SINGLE TARGET COLUMN OVERALL SURVIVAL TIME
    ################---
    mlflow.set_experiment("METABRIC_UC_Plausible_Survival_Time")
    # Load dataset
    df = pd.read_csv(DATASET_PATH)

    # preprocess
    X_train, y_train, X_test, y_test, preprocessing_artifacts = run_preprocessing(
        df,
        pronostic_target_column,
        has_duplicates,
        boolean_cast_columns,
        outlier_columns,
        high_cardinality_columns,
        id_column
    )
    with mlflow.start_run():
        log_preprocessing_artifacts(preprocessing_artifacts)
        run_training_regressor(X_train, y_train, X_test, y_test)

        
    ###################
    # USE CASE 3 - CLASSIFIER ORCTREE CODES DIAGNOSIS
    ###################
    # Load dataset
    df = pd.read_csv(DATASET_PATH)
    # normalize orctree target for multi-class classification
    mlflow.set_experiment("METABRIC_UC_Plausible_Diagnosis")
    df[orctree_target_column] = df[orctree_target_column].astype(str).str.strip().str.lower()
    df_orctree = pd.get_dummies(df[orctree_target_column], prefix="orctree")
    df = pd.concat([df, df_orctree], axis=1).drop(columns=[orctree_target_column])

    # preprocess
    X_train, y_train, X_test, y_test, preprocessing_artifacts = run_preprocessing(
        df,
        df_orctree.columns.tolist(),
        has_duplicates,
        boolean_cast_columns,
        outlier_columns,
        high_cardinality_columns,
        id_column
    )
    with mlflow.start_run():
        log_preprocessing_artifacts(preprocessing_artifacts)
        run_training_multiclass_classifier(X_train, y_train, X_test, y_test)

    return


if "__init__" == __name__:
    argv = argparse.ArgumentParser()
    argv.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset CSV file")
    argv.add_argument("--id_column", type=str, required=True, help="Name of the ID column in the dataset")
    argv.add_argument("--therapeutic_target_columns", type=str, nargs='*', default=[], help="List of columns as therapeutic targets")
    argv.add_argument("--orctree_target_column", type=str, nargs='*', default=[], help="List of columns as diagnostic targets")
    argv.add_argument("--pronostic_target_column", type=str, nargs='*', default=[], help="List of columns as prognostic targets")
    argv.add_argument("--output_report_path", type=str, default=None, help="Path to save the EDA HTML report (optional)")
    args = argv.parse_args()
    main(
        DATASET_PATH=args.dataset_path,
        id_column=args.id_column,
        therapeutic_target_columns=args.therapeutic_target_columns,
        orctree_target_column=args.orctree_target_column,
        pronostic_target_column=args.pronostic_target_column
    )
