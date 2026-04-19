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
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
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


def get_numeric_feature_columns(df):
    """Return numeric columns excluding boolean dtypes."""
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    return [col for col in numeric_cols if not pd.api.types.is_bool_dtype(df[col])]


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


def move_legend_outside(ax, title=None):
    """Place the plot legend outside the axes on the right."""
    legend = ax.get_legend()
    if legend is None:
        return

    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return

    if legend is not None:
        legend.remove()

    ax.legend(
        handles,
        labels,
        title=title,
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
    )

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

    df = cast_true_false_categorical_columns(df)
    
    # - Handle outliers (simple example using IQR)
    for col in outlier_columns:
        if pd.api.types.is_bool_dtype(df[col]):
            continue
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

    categorical_cols = df.select_dtypes(include=['object', 'boolean']).columns.difference([id_column])
    numeric_cols = df.select_dtypes(include=['number']).columns

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
    numeric_cols = get_numeric_feature_columns(df)
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
    if target_col == pd.api.types.is_numeric_dtype(df[target_col]):
        #quantize
        df[target_col] = pd.qcut(df[target_col], q=4, labels=False, duplicates='drop')
    elif pd.api.types.is_bool_dtype(df[target_col]):
        df[target_col] = df[target_col].astype('string')
        
    numeric_cols = get_numeric_feature_columns(df)
    pca = PCA(n_components=2, random_state=42)
    pca_results = pca.fit_transform(df[numeric_cols].fillna(0))

    fig, ax = plt.subplots()
    if target_col:
        sns.scatterplot(x=pca_results[:, 0], y=pca_results[:, 1], hue=df[target_col], ax=ax, palette='viridis')
        ax.set_title(f"PCA Visualization colored by {target_col}")
        move_legend_outside(ax, title=target_col)
    else:
        sns.scatterplot(x=pca_results[:, 0], y=pca_results[:, 1], ax=ax)
        ax.set_title("PCA Visualization")

    fig.tight_layout(rect=(0, 0, 0.82, 1))

    if report:
        report.add_stage("PCA Visualization", [], fig)
    plt.close(fig)
    
def tsne_visualization(df, target_col, report=None):
    """Generate t-SNE visualization for the dataset."""
    if target_col == pd.api.types.is_numeric_dtype(df[target_col]):
        #quantize
        df[target_col] = pd.qcut(df[target_col], q=4, labels=False, duplicates='drop')
    elif pd.api.types.is_bool_dtype(df[target_col]):
        df[target_col] = df[target_col].astype('string')

    numeric_cols = get_numeric_feature_columns(df)
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(df[numeric_cols].fillna(0))

    fig, ax = plt.subplots()
    if target_col:
        sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=df[target_col], ax=ax, palette='viridis')
        ax.set_title(f"t-SNE Visualization colored by {target_col}")
        move_legend_outside(ax, title=target_col)
    else:
        sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], ax=ax)
        ax.set_title("t-SNE Visualization")

    fig.tight_layout(rect=(0, 0, 0.82, 1))

    if report:
        report.add_stage("t-SNE Visualization", [], fig)
    plt.close(fig)

def run_correlation_analysis(df, columns, target, report=None):
    """Generate correlation heatmap for numeric features."""
    numeric_cols = get_numeric_feature_columns(df)
    if len(numeric_cols) < 2:
        print("Not enough numeric columns for correlation analysis.")
        return

    corr_matrix = df[numeric_cols].corr()

    if target in corr_matrix.columns:
        target_corr = corr_matrix[target].drop(target).sort_values(ascending=False)
        print(f"Correlation of features with target '{target}':\n{target_corr}")
    anot = True if len(columns)>20 else False
    cluster_grid = sns.clustermap(corr_matrix, annot=anot, fmt=".2f", cmap='coolwarm', figsize=(10, 8))
    cluster_grid.ax_heatmap.set_title("Correlation Clustermap of Numeric Features")
    cluster_grid.fig.tight_layout()

    if report:
        report.add_stage("Correlation Analysis", [], cluster_grid.fig)
    plt.close(cluster_grid.fig)


def run_gene_expression_heatmap_and_clustermap(df, target=None, report=None, max_features=50):
    """Generate a heatmap and clustermap for gene expression columns."""
    gene_expression_columns = [
        col for col in get_numeric_feature_columns(df)
        if col != target and col != 'cohort'
    ]
    # Discretize target if numeric
    if target is not None and target in df.columns and pd.api.types.is_numeric_dtype(df[target]):
        df[target + '_binned'] = pd.qcut(df[target], q=4, labels=False, duplicates='drop')
        target_col_for_annot = target + '_binned'
    else:
        target_col_for_annot = target

    if len(gene_expression_columns) < 2:
        print("Not enough gene expression columns for heatmap/clustermap analysis.")
        return

    selected_columns = gene_expression_columns
    if len(gene_expression_columns) > max_features:
        selected_columns = (
            df[gene_expression_columns]
            .var()
            .sort_values(ascending=False)
            .head(max_features)
            .index
            .tolist()
        )

    correlation_matrix = df[selected_columns].corr()
    analysis_logs = [
        f"Gene expression columns available: {len(gene_expression_columns)}",
        f"Gene expression columns visualized: {len(selected_columns)}",
        "Selection strategy: top variance genes." if len(selected_columns) < len(gene_expression_columns) else "Selection strategy: all available gene columns.",
    ]

    heatmap_fig, heatmap_ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, ax=heatmap_ax)
    heatmap_ax.set_title("Gene Expression Correlation Heatmap")
    heatmap_fig.tight_layout()
    if report:
        report.add_stage("Gene Expression Heatmap", analysis_logs, heatmap_fig)
    plt.close(heatmap_fig)

    # Prepare col_colors for clustermap (target and cohort)
    col_colors = None
    if target_col_for_annot is not None or 'cohort' in df.columns:
        col_colors_dict = {}
        if target_col_for_annot is not None and target_col_for_annot in df.columns:
            # Use a categorical palette for the target
            unique_targets = df[target_col_for_annot].astype(str).unique()
            target_palette = sns.color_palette('Set2', n_colors=len(unique_targets))
            target_lut = dict(zip(unique_targets, target_palette))
            col_colors_dict['Target'] = df[target_col_for_annot].astype(str).map(target_lut)
        if 'cohort' in df.columns:
            unique_cohorts = df['cohort'].astype(str).unique()
            cohort_palette = sns.color_palette('Set1', n_colors=len(unique_cohorts))
            cohort_lut = dict(zip(unique_cohorts, cohort_palette))
            col_colors_dict['Cohort'] = df['cohort'].astype(str).map(cohort_lut)
        if col_colors_dict:
            import pandas as pd
            col_colors = pd.DataFrame(col_colors_dict)

    cluster_grid = sns.clustermap(
        correlation_matrix,
        cmap='coolwarm',
        center=0,
        figsize=(14, 12),
        col_colors=col_colors,
        dendrogram_ratio=(.1, .2),
        cbar_pos=(0.02, 0.8, 0.05, 0.18)
    )
    cluster_grid.ax_heatmap.set_title("Gene Expression Correlation Clustermap\n(Barra lateral: Target y Cohort)")
    cluster_grid.fig.tight_layout()
    if report:
        report.add_stage("Gene Expression Clustermap", analysis_logs, cluster_grid.fig)
    plt.close(cluster_grid.fig)

def visualize_distributions(df, id_column, columns, report=None):
    """Visualize distributions of numeric features and categorical features frequencies"""

    numeric_cols = get_numeric_feature_columns(df)
    for col in numeric_cols:
        if col != id_column:
            fig, ax = plt.subplots()
            sns.histplot(df[col].dropna(), kde=True, ax=ax)
            ax.set_title(f"Distribution of {col}")
            if report:
                report.add_stage(f"Distribution of {col}", [], fig)
        plt.close(fig)
    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols:
        if col != id_column:
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
            move_legend_outside(ax, title=target)
            plt.xticks(rotation=45, ha='right')
            fig.tight_layout(rect=(0, 0, 0.82, 1))

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


def run_mutual_information_analysis(df, target_columns, report=None, top_n=20):
    """Compute mutual information scores between processed features and target columns."""
    analysis_logs = []

    for target in target_columns:
        if target not in df.columns:
            analysis_logs.append(f"Skipped mutual information analysis: target column '{target}' not found.")
            continue

        target_feature_columns = resolve_processed_non_gene_columns(df, [target])
        feature_df = df.drop(columns=target_feature_columns + [target], errors='ignore')
        feature_df = feature_df.select_dtypes(include=['number', 'boolean']).copy()
        if feature_df.empty:
            analysis_logs.append(f"Skipped mutual information analysis for '{target}': no numeric or boolean features available.")
            continue

        target_series = df[target]
        if pd.api.types.is_numeric_dtype(target_series) and not pd.api.types.is_bool_dtype(target_series):
            target_values = pd.to_numeric(target_series, errors='coerce')
            valid_mask = target_values.notna()
            if valid_mask.sum() < 2:
                analysis_logs.append(f"Skipped mutual information analysis for '{target}': insufficient numeric target values.")
                continue
            scores = mutual_info_regression(
                feature_df.loc[valid_mask].fillna(0),
                target_values.loc[valid_mask],
                random_state=42,
            )
            analysis_type = 'regression'
        else:
            normalized_target = target_series.astype('string')
            valid_mask = normalized_target.notna()
            if valid_mask.sum() < 2:
                analysis_logs.append(f"Skipped mutual information analysis for '{target}': insufficient categorical target values.")
                continue
            encoded_target, _ = pd.factorize(normalized_target.loc[valid_mask])
            scores = mutual_info_classif(
                feature_df.loc[valid_mask].fillna(0),
                encoded_target,
                random_state=42,
            )
            analysis_type = 'classification'

        mutual_info = pd.Series(scores, index=feature_df.columns).sort_values(ascending=False)
        top_scores = mutual_info.head(top_n)
        analysis_logs.append(
            f"Top mutual information scores for target '{target}' ({analysis_type}):\n{top_scores.to_string()}"
        )

        fig, ax = plt.subplots(figsize=(10, max(4, min(top_n, len(top_scores)) * 0.4)))
        sns.barplot(x=top_scores.values, y=top_scores.index, ax=ax, orient='h')
        ax.set_title(f"Top Mutual Information Scores for {target}")
        ax.set_xlabel("Mutual Information")
        ax.set_ylabel("Feature")
        fig.tight_layout()

        if report:
            report.add_stage(
                f"Mutual Information - {target}",
                [f"Top {min(top_n, len(top_scores))} mutual information scores for target '{target}' ({analysis_type})."],
                fig,
            )
        plt.close(fig)

    if report and analysis_logs:
        report.add_stage("Mutual Information Analysis", analysis_logs)


def build_multicategory_therapeutic_target(df, surgery_dummy_columns, therapy_columns):
    """Build a readable multi-label target from surgery dummies and therapy flags."""

    def is_positive(value):
        if pd.isna(value):
            return False
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        return str(value).strip().lower() in {'yes', 'positive', '+', '1', 'true'}

    def label_row(row):
        active_labels = [
            column.replace('breast_surgery_', '')
            for column in surgery_dummy_columns
            if row[column] == 1
        ]
        active_labels.extend(
            column for column in therapy_columns if is_positive(row[column])
        )
        return '|'.join(active_labels) if active_labels else 'none'

    return df.apply(label_row, axis=1)


def resolve_processed_non_gene_columns(df, non_gene_expression_columns):
    """Map raw non-gene columns to the columns available after preprocessing."""
    resolved_columns = []
    for column in non_gene_expression_columns:
        if column in df.columns:
            resolved_columns.append(column)
            continue

        encoded_matches = [
            candidate for candidate in df.columns
            if candidate.startswith(f"{column}_")
        ]
        resolved_columns.extend(encoded_matches)

    return list(dict.fromkeys(resolved_columns))


def run_eda(id_column, non_gene_expression_columns, therapeutic_targets, diagnostic_targets, pronostic_targets, main_file_path, output_report_path=None):
    """Run exploratory data analysis on the dataset."""


    df = pd.read_csv(main_file_path)
    output_qc = run_qc(id_column, main_file_path, do_report=False, show_plots=False)
    
    report = EDAHTMLReport(output_report_path) if output_report_path else None
    visualize_distributions(df, id_column, df.columns, report)

    processed_non_gene_columns = resolve_processed_non_gene_columns(
        df,
        [column for column in non_gene_expression_columns if column != id_column]
    )


    # CASO DE USO 1 MEJOR TERAPIA
    ######################
    print("Exploracion de terapias en pacientes sin causa de muerte por cancer")
    analysis_df = df.copy()
    analysis_df = analysis_df[(analysis_df['death_from_cancer'] != 'Died of Other Causes')| (analysis_df["overall_survival"]==1)] # ensure we are modeling breast cancer specific survival and not overall survival, which would be a different use case with different target variable definition and modeling approach
    
    for col in therapeutic_targets:
        if col != "type_of_breast_surgery" and col in analysis_df.columns:
            analysis_df[col] = analysis_df[col].astype(str).str.strip().str.lower().map(
                lambda x: pd.NA if pd.isna(x) else x in {'yes', 'positive', '+', '1'}
            ).astype('boolean')

    df_surgery = pd.DataFrame()
    df_surgery["breast_surgery"] = ~df["type_of_breast_surgery"].isna()
    analysis_df = pd.concat([analysis_df, df_surgery], axis=1)
    thcolumns = [col for col in therapeutic_targets if col != "type_of_breast_surgery"]
    analysis_df["multicategory_therapeutic_target"] = build_multicategory_therapeutic_target(
        analysis_df,
        df_surgery.columns.tolist(),
        thcolumns,
    )

    df_p = run_preprocessing(
        analysis_df.copy(),
        id_column,
        output_qc
    )
    df_p["multicategory_therapeutic_target"] = analysis_df.loc[
        df_p.index,
        "multicategory_therapeutic_target",
    ]

    analysis_df = analysis_df.drop(columns=[id_column], errors='ignore')
    df_p = df_p.drop(columns=[id_column], errors='ignore')
    run_bias_analysis(analysis_df, ["multicategory_therapeutic_target"], report)


    if "multicategory_therapeutic_target" not in processed_non_gene_columns:
        processed_non_gene_columns.append("multicategory_therapeutic_target")

    df_non_gene = df_p[processed_non_gene_columns]
    df_gene = df_p.drop(columns=processed_non_gene_columns, errors='ignore')
    if "multicategory_therapeutic_target" not in df_gene.columns:
        df_gene = pd.concat([df_gene, df_p[["multicategory_therapeutic_target"]]], axis=1)


    mutual_info_df = df_p.copy()
    mutual_target_columns = []
    for target in therapeutic_targets:
        if target in analysis_df.columns and target not in mutual_info_df.columns:
            mutual_info_df[target] = analysis_df.loc[mutual_info_df.index, target]
        if target in mutual_info_df.columns and target not in mutual_target_columns:
            mutual_target_columns.append(target)

    if "multicategory_therapeutic_target" in df_p.columns and "multicategory_therapeutic_target" not in mutual_target_columns:
        mutual_target_columns.append("multicategory_therapeutic_target")

    run_correlation_analysis(df_non_gene, df_non_gene.columns, "multicategory_therapeutic_target", report)
    run_correlation_analysis(df_gene, df_gene.columns, "multicategory_therapeutic_target", report)
    run_gene_expression_heatmap_and_clustermap(df_gene, "multicategory_therapeutic_target", report)
    run_mutual_information_analysis(mutual_info_df, mutual_target_columns, report)
    run_pca_visualization(df_p, "multicategory_therapeutic_target", report)
    tsne_visualization(df_p, "multicategory_therapeutic_target", report)




    # CASO DE USO 2 prognostic targets
    df = pd.read_csv(main_file_path)
    df = df[df['overall_survival'] == 0] # filter to only patients who died to ensure we are modeling survival time and not a mix of survival time and censoring, which would be a different use case with different target variable definition and modeling approach

    df_p = run_preprocessing(
        df.copy(),
        id_column,
        output_qc
    )
    run_correlation_analysis(df_non_gene, df_non_gene.columns, pronostic_targets[0], report)
    run_correlation_analysis(df_gene, df_gene.columns, pronostic_targets[0], report)
    run_gene_expression_heatmap_and_clustermap(df_gene, pronostic_targets[0], report)
    run_mutual_information_analysis(mutual_info_df, pronostic_targets, report)
    run_pca_visualization(df_p, pronostic_targets[0], report)
    tsne_visualization(df_p, pronostic_targets[0], report)



    # CASO DE USO 3 diagnostic targets

    
    df = pd.read_csv(main_file_path)
    df = df[df['overall_survival'] == 0] # filter to only patients who died to ensure we are modeling survival time and not a mix of survival time and censoring, which would be a different use case with different target variable definition and modeling approach

    df_p = run_preprocessing(
        df.copy(),
        id_column,
        output_qc
    )

    mutual_info_df = df_p.copy()
    mutual_target_columns = []
    for target in diagnostic_targets:
        if target in analysis_df.columns and target not in mutual_info_df.columns:
            mutual_info_df[target] = analysis_df.loc[mutual_info_df.index, target]
        if target in mutual_info_df.columns and target not in mutual_target_columns:
            mutual_target_columns.append(target)

    run_correlation_analysis(df_non_gene, df_non_gene.columns, diagnostic_targets[0], report)
    run_correlation_analysis(df_gene, df_gene.columns, diagnostic_targets[0], report)
    run_gene_expression_heatmap_and_clustermap(df_gene, diagnostic_targets[0], report)
    run_mutual_information_analysis(mutual_info_df, diagnostic_targets, report)
    run_pca_visualization(df_p, diagnostic_targets[0], report)
    tsne_visualization(df_p, diagnostic_targets[0], report)







    if report:
        report.generate_report()

if __name__ == "__main__":
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