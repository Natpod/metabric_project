import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import io
import base64
import html
import os
import scipy.stats as stats
from sklearn.manifold import TSNE
import numpy as np
import sys
from data.quality.QC import run_qc
from sklearn.decomposition import PCA
from scipy import stats

# TODO: Define a class to generate HTML reports for EDA

class EDAHTMLReport:
    def __init__(self, output_path):
        self.output_path = output_path
        self.sections = []

    def add_section(self, title, content):
        self.sections.append((title, content))

    def generate_report(self):
        with open(self.output_path, 'w') as f:
            f.write("<html><head><title>EDA Report</title></head><body>")
            f.write("<h1>Exploratory Data Analysis Report</h1>")
            for title, content in self.sections:
                f.write(f"<h2>{title}</h2>")
                f.write(content)
            f.write("</body></html>")

    @staticmethod
    def figure_to_base64(fig):
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        encoded = base64.b64encode(buffer.read()).decode('utf-8')
        buffer.close()
        return encoded

    def add_stage(self, title, logs=None, fig=None):
        logs = logs or []
        content_parts = []

        if logs:
            escaped_logs = html.escape("\n".join(logs))
            content_parts.append(f"<pre>{escaped_logs}</pre>")

        if fig is not None:
            encoded_img = self.figure_to_base64(fig)
            content_parts.append(
                f'<img src="data:image/png;base64,{encoded_img}" alt="{html.escape(title)} visualization" style="max-width:100%;height:auto;"/>'
            )

        self.add_section(title, "".join(content_parts))

def run_preprocessing(df, id_column,has_duplicates,boolean_cast_columns,outlier_columns, high_cardinality_columns):
    """Run data preprocessing steps on the dataset."""
    
    # Example preprocessing steps:
    # - Handle duplicates
    if has_duplicates:
        df = df.drop_duplicates(subset=id_column)
    
    # - Handle boolean columns
    for col in boolean_cast_columns:
        normalized = df[col].astype('string').str.strip().str.lower()
        df[col] = normalized.map(
            lambda value: pd.NA if pd.isna(value) else value in {'positive', '+', '1', 'yes'}
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

    # one hot encoding for grouped categorical columns and the rest of categorical columns except id_column
    categorical_cols = df.select_dtypes(include='object').columns.difference(high_cardinality_columns + [id_column])
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Distribution diagnostics for numeric columns after outlier treatment
    # Rules:
    # - zero-inflated non-negative: Log1pTransform
    # - bounded [0,1]: MinMaxScaler
    # - approximately normal: StandardScaler
    # - heavy tails / non-normal: RobustScaler
    decision_scalers = {}
    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        series = df[col].dropna()

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
            df[col] = df[col].apply(lambda x: np.log1p(x) if pd.notna(x) else x)
        elif decision == 'MinMaxScaler':
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val > min_val:
                df[col] = (df[col] - min_val) / (max_val - min_val)
        elif decision == 'StandardScaler':
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                df[col] = (df[col] - mean) / std
        elif decision == 'RobustScaler':
            median = df[col].median()
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                df[col] = (df[col] - median) / iqr
        # NoScaling case does not require any transformation

    return df

def run_pca_visualization(df, target_col, report=None):
    """Generate PCA visualization for the dataset."""

    numeric_cols = df.select_dtypes(include='number').columns
    pca = PCA(n_components=2, random_state=42)
    pca_results = pca.fit_transform(df[numeric_cols].fillna(0))

    fig, ax = plt.subplots()
    if target_col:
        sns.scatterplot(x=pca_results[:, 0], y=pca_results[:, 1], hue=df[target_col], ax=ax, palette='viridis')
        ax.set_title(f"PCA Visualization colored by {target_col}")
    else:
        sns.scatterplot(x=pca_results[:, 0], y=pca_results[:, 1], ax=ax)
        ax.set_title("PCA Visualization")

    if report:
        report.add_stage("PCA Visualization", [], fig)
    plt.close(fig)
    
def tsne_visualization(df, target_col, report=None):
    """Generate t-SNE visualization for the dataset."""
    numeric_cols = df.select_dtypes(include='number').columns
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(df[numeric_cols].fillna(0))

    fig, ax = plt.subplots()
    if target_col:
        sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=df[target_col], ax=ax, palette='viridis')
        ax.set_title(f"t-SNE Visualization colored by {target_col}")
    else:
        sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], ax=ax)
        ax.set_title("t-SNE Visualization")

    if report:
        report.add_stage("t-SNE Visualization", [], fig)
    plt.close(fig)

def run_correlation_analysis(df, columns, target, report=None):
    """Generate correlation heatmap for numeric features."""
    numeric_cols = df.select_dtypes(include='number').columns
    if len(numeric_cols) < 2:
        print("Not enough numeric columns for correlation analysis.")
        return

    corr_matrix = df[numeric_cols].corr()

    if target in corr_matrix.columns:
        target_corr = corr_matrix[target].drop(target).sort_values(ascending=False)
        print(f"Correlation of features with target '{target}':\n{target_corr}")

    cluster_grid = sns.clustermap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', figsize=(10, 8))
    cluster_grid.ax_heatmap.set_title("Correlation Clustermap of Numeric Features")
    cluster_grid.fig.tight_layout()


    if report:
        report.add_stage("Correlation Analysis", [], cluster_grid.fig)
    plt.close(cluster_grid.fig)

def visualize_distributions(df, columns, report=None):
    """Visualize distributions of numeric features and categorical features frequencies"""

    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
        if report:
            report.add_stage(f"Distribution of {col}", [], fig)
        plt.close(fig)
    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols:
        fig, ax = plt.subplots()
        sns.countplot(x=df[col], ax=ax)
        ax.set_title(f"Frequency of {col}")
        ax.tick_params(axis='x', rotation=90)
        if report:
            report.add_stage(f"Frequency of {col}", [], fig)
        plt.close(fig)

def run_bias_analysis(df, target_columns, report, bias_columns = ['age_at_diagnosis','cohort','ethnicity']):   

    # age format into categories
    df['age_group'] = pd.cut(df['age_at_diagnosis'], bins=[0, 40, 50, 60, 70, 80, 100], labels=['0-39', '40-49', '50-59', '60-69', '70-79', '80+'], right=False)
    bias_columns.append('age_group')
    bias_columns.remove('age_at_diagnosis') # replace age_at_diagnosis with age_group in bias_columns

    # frequency of bias groups per target visualization
    bias_logs = []
    for target in target_columns:
        if target not in df.columns:
            bias_logs.append(f"Skipped bias-frequency visualization: target column '{target}' not found.")
            continue

        for bias_col in bias_columns:
            if bias_col not in df.columns:
                bias_logs.append(f"Skipped visualization for {target} vs {bias_col}: bias column not found.")
                continue

            freq_table = pd.crosstab(df[bias_col], df[target])
            bias_logs.append(f"Frequency table for {target} vs {bias_col}:\n{freq_table.to_string()}")

            fig, ax = plt.subplots()
            freq_table.plot(kind='bar', stacked=True, ax=ax)
            ax.set_title(f"Frequency of {bias_col} by {target}")
            ax.set_xlabel(bias_col)
            ax.set_ylabel("Count")
            ax.legend(title=target)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            if report:
                report.add_stage(
                    f"Bias Frequency - {target} vs {bias_col}",
                    [f"Frequency distribution for {bias_col} grouped by {target}."],
                    fig
                )
            plt.close(fig)


    # hypothesis testing for bias analysis
    for target in target_columns:
        for bias_col in bias_columns:
            if target not in df.columns or bias_col not in df.columns:
                continue
            contingency_table = pd.crosstab(df[target], df[bias_col])
            chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
            bias_logs.append(f"Chi-squared test for {target} vs {bias_col}: chi2={chi2:.4f}, p-value={p:.4f}")
    if report:
        report.add_stage("Bias Analysis", bias_logs)


def run_eda(id_column, non_gene_expression_columns, therapeutic_targets, diagnostic_targets, pronostic_targets, main_file_path, output_report_path=None):
    """Run exploratory data analysis on the dataset."""

    df = pd.read_csv(main_file_path)
    report = EDAHTMLReport(output_report_path) if output_report_path else None
    visualize_distributions(df, df.columns, report)
    
    df_p = run_preprocessing(
        df,
        id_column,
        *run_qc(id_column, main_file_path, do_report=False, show_plots=False),
    )
    # therapeutic columns

    for col in therapeutic_targets:
        if col != "type_of_breast_surgery":
            df_p[col] = df_p[col].astype(str).str.strip().str.lower().map(
                lambda x: pd.NA if pd.isna(x) else x in {'yes', 'positive', '+', '1'}
            ).astype('boolean')

    df_surgery = pd.get_dummies(df["type_of_breast_surgery"], prefix="breast_surgery")
    df = pd.concat([df, df_surgery], axis=1)
    thcolumns = [col for col in therapeutic_targets if col != "type_of_breast_surgery"]
    
    df["multicategory_therapeutic_target"] = pd.from_dummies(df[df_surgery.columns.tolist() + thcolumns].astype(str).agg(''.join, axis=1), prefix="multicategory_target")
    run_bias_analysis(df, "multicategory_therapeutic_target", report)

    df_non_gene = df[non_gene_expression_columns]
    df_gene = pd.concat([df.drop(columns=non_gene_expression_columns), df["multicategory_therapeutic_target"]], axis=1)

    run_correlation_analysis(df_non_gene, df_non_gene.columns, "multicategory_therapeutic_target", report)
    run_correlation_analysis(df_gene, df_gene.columns, "multicategory_therapeutic_target", report)
    run_pca_visualization(df, "multicategory_therapeutic_target", report)
    run_tsne_visualization(df, "multicategory_therapeutic_target", report)


    # diagnostic targets



    # prognostic targets
    if report:
        report.generate_report()

if "__init__" == __name__:
    argv = argparse.ArgumentParser()
    argv.add_argument("--id_column", type=str, required=True, help="Name of the ID column in the dataset")
    argv.add_argument("--main_file_path", type=str, required=True, help="Path to the main CSV file for EDA")
    argv.add_argument("--non_gene_expression_columns", type=str, nargs='*', default=[], help="List of non-gene expression columns to include in EDA")
    argv.add_argument("--therapeutic_target_columns", type=str, nargs='*', default=[], help="List of columns as therapeutic targets")
    argv.add_argument("--diagnostic_target_columns", type=str, nargs='*', default=[], help="List of columns as diagnostic targets")
    argv.add_argument("--pronostic_target_columns", type=str, nargs='*', default=[], help="List of columns as prognostic targets")
    argv.add_argument("--output_report_path", type=str, default=None, help="Path to save the EDA HTML report (optional)")
    args = argv.parse_args()
    
    run_eda(args.id_column, args.non_gene_expression_columns, args.therapeutic_target_columns, args.diagnostic_target_columns, args.pronostic_target_columns, args.main_file_path, args.output_report_path)