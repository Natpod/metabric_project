import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import io
import base64
import html

class QCHTMLReport:
    def __init__(self, output_path):
        self.output_path = output_path
        self.sections = []

    def add_section(self, title, content):
        self.sections.append((title, content))

    def generate_report(self):
        with open(self.output_path, 'w') as f:
            f.write("<html><head><title>QC Report</title></head><body>")
            f.write("<h1>Quality Control Report</h1>")
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


def _finalize_figure(fig, show_plots):
    if show_plots:
        plt.show()
    plt.close(fig)


def run_qc(id_column, main_file_path, do_report=False, output_report_path=None, show_plots=True):
    """Run quality control checks on the dataset.
    outputs:    
    - has_duplicates: bool
    - missing_value_columns: list of columns with missing values
    - boolean_cast_columns: list of columns that were cast to boolean
    - outlier_columns: list of numeric columns with outliers
    - high_cardinality_columns: list of categorical columns with high cardinality
    """

    df = pd.read_csv(main_file_path)
    report_path = output_report_path or "qc_report.html"
    report = QCHTMLReport(report_path) if do_report else None

    def log(message, stage_logs=None):
        print(message)
        if stage_logs is not None:
            stage_logs.append(str(message))

    # Duplicated patients
    duplicate_logs = []
    log("Checking for duplicates...", duplicate_logs)
    duplicates = df.duplicated(subset=id_column).sum()
    fig, ax = plt.subplots()
    sns.countplot(x=df.duplicated(subset=id_column), ax=ax)
    ax.set_title("Duplicate Entries Based on ID Column")
    duplicate_message = f"Number of duplicate entries based on {id_column}: {duplicates}"
    log(duplicate_message, duplicate_logs)
    if report:
        report.add_stage("Duplicate Check", duplicate_logs, fig)
    _finalize_figure(fig, show_plots)

    # Missing val
    missing_logs = []
    log("Checking for missing values...", missing_logs)
    missing_values = df.isnull().sum()
    fig, ax = plt.subplots()
    sns.barplot(x=missing_values.index, y=missing_values.values, ax=ax)
    ax.tick_params(axis='x', rotation=90)
    ax.set_title("Missing Values Per Column")

    log("Missing values per column:", missing_logs)
    log(missing_values.to_string(), missing_logs)
    if report:
        report.add_stage("Missing Values Check", missing_logs, fig)
    _finalize_figure(fig, show_plots)

    outlier_logs = []
    log("Checking for outliers in numeric columns...", outlier_logs)
    numeric_cols = df.select_dtypes(include=['float', 'int']).columns
    outlier_counts = {}
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_counts[col] = len(outliers)
        outlier_message = f"Outliers in {col}: {len(outliers)}"
        log(outlier_message, outlier_logs)
        fig, ax = plt.subplots()
        sns.boxplot(x=df[col], ax=ax)
        ax.set_title(f"Outliers in {col}")
        if report:
            report.add_stage(f"Outlier Check - {col}", [outlier_message], fig)
        _finalize_figure(fig, show_plots)

    cardinality_logs = []
    log("Checking cardinality of categorical columns...", cardinality_logs)
    categorical_cols = df.select_dtypes(include=['object']).columns

    cardinality_info = {}
    boolean_cast_cols = []
    positive_tokens = {"positive", "+", "yes", "1"}
    for col in categorical_cols:
        normalized_series = df[col].astype('string').str.strip().str.lower()
        non_null_values = normalized_series.dropna()
        cardinality = non_null_values.nunique()
        cardinality_message = f"Cardinality of {col}: {cardinality}"
        log(cardinality_message, cardinality_logs)
        cardinality_info[col] = cardinality

        unique_values = set(non_null_values.unique().tolist())
        if cardinality == 2 and unique_values.intersection(positive_tokens):
            boolean_series = normalized_series.map(
                lambda value: pd.NA if pd.isna(value) else value in positive_tokens
            )
            df[col] = boolean_series.astype('boolean')
            boolean_cast_cols.append(col)
            cast_message = f"Casted {col} to boolean using positive tokens {sorted(positive_tokens)}"
            log(cast_message, cardinality_logs)

        fig, ax = plt.subplots()
        sns.countplot(x=normalized_series, ax=ax)
        ax.set_title(f"Cardinality of {col}")
        ax.tick_params(axis='x', rotation=90)
        if report:
            report.add_stage(f"Cardinality Check - {col}", [cardinality_message], fig)
        _finalize_figure(fig, show_plots)

    # frequency of values in categorical columns
    frequency_logs = []
    log("Checking frequency of values in categorical columns...", frequency_logs)
    for col in categorical_cols:    
        normalized_series = df[col].astype('string').str.strip().str.lower()
        frequency_message = f"Value counts for {col}:\n{normalized_series.value_counts()}"
        log(frequency_message, frequency_logs)
        fig, ax = plt.subplots()
        sns.countplot(x=normalized_series, ax=ax)
        ax.set_title(f"Frequency of Values in {col}")
        ax.tick_params(axis='x', rotation=90)
        if report:
            report.add_stage(f"Frequency Check - {col}", [frequency_message], fig)
        _finalize_figure(fig, show_plots)

    preprocessing_logs = ["Recommended preprocessing strategy based on QC:"]
    if duplicates > 0:
        preprocessing_logs.append(f"- Remove or review {duplicates} duplicate rows based on {id_column}.")
    else:
        preprocessing_logs.append("- No duplicate row cleanup required.")

    missing_cols = missing_values[missing_values > 0]
    if not missing_cols.empty:
        preprocessing_logs.append(f"- Impute or handle missing values in columns: {', '.join(missing_cols.index.tolist())}.")
    else:
        preprocessing_logs.append("- No missing-value imputation required.")

    outlier_cols = [col for col, count in outlier_counts.items() if count > 0]
    if outlier_cols:
        preprocessing_logs.append(f"- Consider robust scaling/capping for outliers in: {', '.join(outlier_cols)}.")
    else:
        preprocessing_logs.append("- No strong outlier treatment indicated by IQR rule.")

    high_cardinality_cols = [col for col, card in cardinality_info.items() if card > 20]
    if high_cardinality_cols:
        preprocessing_logs.append(
            f"- High-cardinality categorical columns detected ({', '.join(high_cardinality_cols)}); consider target/frequency encoding."
        )
    else:
        preprocessing_logs.append("- Categorical cardinality is manageable for standard encoding.")

    if boolean_cast_cols:
        preprocessing_logs.append(f"- Converted to boolean: {', '.join(boolean_cast_cols)}.")
    else:
        preprocessing_logs.append("- No categorical columns met the boolean-cast rule.")

    for message in preprocessing_logs:
        print(message)

    if report:
        summary_logs = [
            f"Rows: {len(df)}",
            f"Columns: {len(df.columns)}",
            f"Duplicate rows by {id_column}: {duplicates}",
            f"Numeric columns checked for outliers: {len(numeric_cols)}",
            f"Categorical columns checked for cardinality: {len(categorical_cols)}"
        ]
        report.add_stage("Preprocessing Strategy", preprocessing_logs)
        report.add_stage("QC Summary", summary_logs)
        report.generate_report()
        print(f"QC HTML report generated at: {report_path}")
    return duplicates > 0, boolean_cast_cols, outlier_cols, high_cardinality_cols

if __name__ == "__main__":
    argv = argparse.ArgumentParser()
    argv.add_argument("--id_column", required=True, help="ID column for duplicate check")
    argv.add_argument("--main_file_path", required=True, help="Path to the main CSV file")
    argv.add_argument("--do-report", action="store_true", help="Generate an HTML report")
    argv.add_argument("--no-show-plots", action="store_true", help="Disable interactive matplotlib windows")
    argv.add_argument("--output_report_path", required=False, help="Path to save the QC report (optional)")
    args = argv.parse_args()
    run_qc(args.id_column, args.main_file_path, args.do_report, args.output_report_path, show_plots=not args.no_show_plots)