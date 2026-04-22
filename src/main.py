from data.quality.QC import run_qc
from data.EDA import run_eda
from ml.training import main as training_main
from ml.predict import main as predict_main
import argparse
import os
import numpy as np
import pandas as pd
from ml.get_registered_models import main as mlflow_export_main
from mlflow.tracking import MlflowClient

therapeutic_targets = ['chemotherapy','hormone_therapy','radio_therapy','type_of_breast_surgery']
prognosis_targets = ['overall_survival_months']
diagnosis_targets = ['oncotree_code']
non_gene_expression_columns = ['patient_id','age_at_diagnosis','cohort','er_status_measured_by_ihc','lymph_nodes_examined_positive','mutation_count','nottingham_prognostic_index','overall_survival_months','tumor_size','tumor_stage','chemotherapy','neoplasm_histologic_grade','hormone_therapy','radio_therapy','death_from_cancer','geo_location_id','ethnicity','type_of_breast_surgery','cancer_type','cancer_type_detailed','cellularity','pam50_+_claudin-low_subtype','er_status','her2_status_measured_by_snp6','her2_status','tumor_other_histologic_subtype','inferred_menopausal_state','integrative_cluster','primary_tumor_laterality','oncotree_code','overall_survival','pr_status','3-gene_classifier_subtype']
DATASET_PATH = "C:\\Users\\User\\metabric_project\\data\\FCS_ml_test_input_data_rna_mutation.csv"
DATASET_PATH_INFERENCE_THERAPY = "C:\\Users\\User\\metabric_project\\data\\inference_test\\input\\inference_therapy.csv"
DATASET_PATH_INFERENCE_SURVIVAL = "C:\\Users\\User\\metabric_project\\data\\inference_test\\input\\inference_survival.csv"
DATASET_PATH_INFERENCE_DIAGNOSIS = "C:\\Users\\User\\metabric_project\\data\\inference_test\\input\\inference_diagnosis.csv"




def ml_cicle(patient_id_column, dataset_path, therapeutic_targets, prognosis_targets, diagnosis_targets, dataset_path_inference_therapy, dataset_path_inference_survival, dataset_path_inference_diagnosis, show_plots=True):

    # Run QC
    run_qc(
        id_column=patient_id_column,
        main_file_path=dataset_path,
        do_report=True,
        output_report_path="./reports/qc_report.html",
        show_plots=show_plots,
    )
    
    # Run EDA
    run_eda(patient_id_column, non_gene_expression_columns, 
    therapeutic_targets, diagnosis_targets, prognosis_targets, dataset_path, "./reports/eda_report.html")

    # TRAINING
    # Train models for each target and save results
    training_main(
        dataset_path,
        patient_id_column,
        non_gene_expression_columns,
        therapeutic_targets,
        "oncotree_code",
        "overall_survival_months",
    )



    runs = mlflow_export_main(csv_path="./reports/training_runs.csv")
    print(f"Total runs retrieved: {len(runs)}")

    # INFERENCE
    # Prepare data for inference and make predictions
    possible_experiments_inference = {"METABRIC_UC_Plausible_Therapy": dataset_path_inference_therapy,
                                      "METABRIC_UC_Plausible_Survival_Time": dataset_path_inference_survival,
                                      "METABRIC_UC_Plausible_Diagnosis": dataset_path_inference_diagnosis}
    
    for experiment_name, dataset_path in possible_experiments_inference.items():
        predictions, shap_values = predict_main(dataset_path, experiment_name)
        inference_index = pd.read_csv(dataset_path).index
        predictions_array = np.array(predictions)
        if predictions_array.ndim == 1:
            pred_cols = [f"{experiment_name}_prediction"]
            predictions_array = predictions_array.reshape(-1, 1)
        else:
            pred_cols = [f"{experiment_name}_prediction_{i}" for i in range(predictions_array.shape[1])]
        predictions = pd.DataFrame(predictions_array, columns=pred_cols, index=inference_index)
        shap_array = np.array(getattr(shap_values, "values", shap_values))
        if shap_array.ndim == 3:
            # (samples, features, classes) → mean |SHAP| across classes per feature
            shap_array = np.abs(shap_array).mean(axis=2)
        shap_cols = [f"{experiment_name}_shap_{i}" for i in range(shap_array.shape[1])]
        shap_values_df = pd.DataFrame(shap_array, columns=shap_cols, index=inference_index)
        results_df = pd.concat([predictions, shap_values_df], axis=1)
        getpwd = os.getcwd()
        parent_dir = os.path.dirname(getpwd)
        output_dir = os.path.join(parent_dir, "reports/data_inference_output")
        os.makedirs(output_dir, exist_ok=True)
        results_df.to_csv(os.path.join(output_dir, f"{experiment_name}_predictions_shap.csv"), index=False)

if __name__ == "__main__":
    argv = argparse.ArgumentParser()
    argv.add_argument("--dataset_path", type=str, default=DATASET_PATH, help="Path to the dataset CSV file")
    argv.add_argument("--therapeutic_targets", nargs='+', default=therapeutic_targets, help="List of therapeutic target columns")
    argv.add_argument("--prognosis_targets", nargs='+', default=prognosis_targets, help="List of prognosis target columns")
    argv.add_argument("--diagnosis_targets", nargs='+', default=diagnosis_targets, help="List of diagnosis target columns")
    argv.add_argument("--dataset_inference_therapy", type=str, default=DATASET_PATH_INFERENCE_THERAPY, help="Path to the therapy inference dataset CSV file")
    argv.add_argument("--dataset_inference_survival", type=str, default=DATASET_PATH_INFERENCE_SURVIVAL, help="Path to the survival inference dataset CSV file")
    argv.add_argument("--dataset_inference_diagnosis", type=str, default=DATASET_PATH_INFERENCE_DIAGNOSIS, help="Path to the diagnosis inference dataset CSV file")
    argv.add_argument("--patient_id_column", type=str, default="patient_id", help="Name of the patient ID column for QC")
    argv.add_argument("--no-show-plots", action="store_true", help="Disable interactive matplotlib windows during QC")
    argv.add_argument("--non_gene_expression_columns", nargs='+', default=non_gene_expression_columns, help="List of non-gene expression columns")
    args = argv.parse_args()
    ml_cicle(
        args.patient_id_column,
        args.dataset_path,
        args.therapeutic_targets,
        args.prognosis_targets,
        args.diagnosis_targets,
        args.dataset_inference_therapy,
        args.dataset_inference_survival,
        args.dataset_inference_diagnosis,
        show_plots=not args.no_show_plots,
    )
