import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def standardize_text_value(value):
    if pd.isna(value):
        return value
    text = str(value).strip()
    replacements = {
        "m": "Male",
        "male": "Male",
        "f": "Female",
        "female": "Female",
    }
    return replacements.get(text.lower(), text.title())


def quality_report(df):
    return pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[col].dtype) for col in df.columns],
        "null_count": [df[col].isna().sum() for col in df.columns],
        "null_percent": [(df[col].isna().mean() * 100).round(2) for col in df.columns],
        "unique_values": [df[col].nunique(dropna=True) for col in df.columns],
    })


def cap_outliers_iqr(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return series.clip(lower=lower, upper=upper), lower, upper


def main(csv_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, encoding_errors="ignore")
    before_rows = len(df)
    before_duplicates = df.duplicated().sum()
    before_nulls = df.isna().sum().sum()

    print("\n--- Before Cleaning Quality Report ---")
    before_report = quality_report(df)
    print(before_report)
    before_report.to_csv(output_dir / "task3_before_quality_report.csv", index=False)

    cleaned = df.copy()
    cleaned.columns = (
        cleaned.columns
        .str.strip()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.lower()
    )

    duplicate_count = cleaned.duplicated().sum()
    cleaned = cleaned.drop_duplicates()
    print(f"\nDuplicates removed: {duplicate_count}")

    for col in cleaned.columns:
        if cleaned[col].dtype == "object":
            cleaned[col] = cleaned[col].replace(r"^\s*$", np.nan, regex=True)

    for col in cleaned.columns:
        low = col.lower()
        if "date" in low or "time" in low:
            converted = pd.to_datetime(cleaned[col], errors="coerce")
            if converted.notna().mean() >= 0.5:
                cleaned[col] = converted

    for col in cleaned.columns:
        if cleaned[col].dtype == "object":
            numeric_candidate = pd.to_numeric(
                cleaned[col].astype(str).str.replace(",", "", regex=False).str.replace("$", "", regex=False),
                errors="coerce",
            )
            if numeric_candidate.notna().mean() >= 0.7:
                cleaned[col] = numeric_candidate

    for col in cleaned.select_dtypes(include="object").columns:
        if "gender" in col or "sex" in col:
            cleaned[col] = cleaned[col].apply(standardize_text_value)
        else:
            cleaned[col] = cleaned[col].astype(str).str.strip()
            cleaned[col] = cleaned[col].replace("nan", np.nan)

    imputation_notes = []
    for col in cleaned.columns:
        missing = cleaned[col].isna().sum()
        if missing == 0:
            continue

        if pd.api.types.is_numeric_dtype(cleaned[col]):
            fill_value = cleaned[col].median()
            cleaned[col] = cleaned[col].fillna(fill_value)
            imputation_notes.append((col, missing, f"Filled with median: {fill_value}"))
        elif pd.api.types.is_datetime64_any_dtype(cleaned[col]):
            cleaned[col] = cleaned[col].ffill().bfill()
            imputation_notes.append((col, missing, "Filled with forward fill then backward fill"))
        else:
            mode = cleaned[col].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "Unknown"
            cleaned[col] = cleaned[col].fillna(fill_value)
            imputation_notes.append((col, missing, f"Filled with mode: {fill_value}"))

    imputation_df = pd.DataFrame(imputation_notes, columns=["column", "missing_before", "strategy"])
    print("\n--- Missing Data Handling ---")
    print(imputation_df)
    imputation_df.to_csv(output_dir / "task3_imputation_notes.csv", index=False)

    outlier_notes = []
    for col in cleaned.select_dtypes(include="number").columns:
        original = cleaned[col].copy()
        capped, lower, upper = cap_outliers_iqr(cleaned[col])
        changed = (original != capped).sum()
        cleaned[col] = capped
        outlier_notes.append((col, lower, upper, int(changed), "Capped using IQR limits"))

    outlier_df = pd.DataFrame(outlier_notes, columns=["column", "lower_limit", "upper_limit", "values_capped", "decision"])
    print("\n--- Outlier Handling ---")
    print(outlier_df)
    outlier_df.to_csv(output_dir / "task3_outlier_notes.csv", index=False)

    after_report = quality_report(cleaned)
    after_report.to_csv(output_dir / "task3_after_quality_report.csv", index=False)

    summary = pd.DataFrame({
        "metric": ["row_count", "duplicate_count", "total_null_count", "columns"],
        "before": [before_rows, before_duplicates, before_nulls, df.shape[1]],
        "after": [len(cleaned), cleaned.duplicated().sum(), cleaned.isna().sum().sum(), cleaned.shape[1]],
    })
    print("\n--- Before vs After Summary ---")
    print(summary)
    summary.to_csv(output_dir / "task3_before_after_summary.csv", index=False)

    output_file = output_dir / "task3_cleaned_dataset.csv"
    cleaned.to_csv(output_file, index=False)
    print(f"\nCleaned dataset saved to: {output_file.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to messy CSV file")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    main(args.csv, args.output_dir)
