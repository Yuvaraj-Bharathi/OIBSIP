import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")


def find_column(columns, keywords):
    lowered = {col.lower(): col for col in columns}
    for keyword in keywords:
        for low, original in lowered.items():
            if keyword in low:
                return original
    return None


def save_plot(output_dir, filename):
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=160)
    plt.close()


def main(csv_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    print("\n--- Initial Inspection ---")
    print("Shape:", df.shape)
    print("\nData types:\n", df.dtypes)
    print("\nNull values:\n", df.isna().sum())
    print("\nFirst rows:\n", df.head())

    numeric_df = df.select_dtypes(include="number")
    print("\n--- Descriptive Statistics ---")
    print(numeric_df.describe())
    print("\nMean:\n", numeric_df.mean(numeric_only=True))
    print("\nMedian:\n", numeric_df.median(numeric_only=True))
    print("\nMode:\n", numeric_df.mode().head(1))
    print("\nStandard Deviation:\n", numeric_df.std(numeric_only=True))

    date_col = find_column(df.columns, ["date", "order date", "invoice"])
    sales_col = find_column(df.columns, ["sales", "revenue", "amount", "total", "price"])
    product_col = find_column(df.columns, ["product", "item"])
    category_col = find_column(df.columns, ["category", "segment"])
    gender_col = find_column(df.columns, ["gender", "sex"])
    age_col = find_column(df.columns, ["age"])

    if date_col and sales_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        time_df = df.dropna(subset=[date_col]).copy()
        time_df = time_df.set_index(date_col).sort_index()

        monthly_sales = time_df[sales_col].resample("ME").sum()
        plt.figure(figsize=(12, 5))
        monthly_sales.plot(marker="o")
        plt.title("Monthly Sales Trend")
        plt.xlabel("Month")
        plt.ylabel("Sales")
        save_plot(output_dir, "task1_monthly_sales.png")

        quarterly_sales = time_df[sales_col].resample("QE").sum()
        plt.figure(figsize=(12, 5))
        quarterly_sales.plot(marker="o", color="darkorange")
        plt.title("Quarterly Sales Trend")
        plt.xlabel("Quarter")
        plt.ylabel("Sales")
        save_plot(output_dir, "task1_quarterly_sales.png")
    else:
        print("\nSkipped time-series charts: date or sales column not detected.")

    if age_col:
        plt.figure(figsize=(9, 5))
        sns.histplot(df[age_col].dropna(), bins=10, kde=True)
        plt.title("Customer Age Distribution")
        plt.xlabel("Age")
        save_plot(output_dir, "task1_age_distribution.png")

    if gender_col:
        plt.figure(figsize=(7, 5))
        sns.countplot(data=df, x=gender_col, order=df[gender_col].value_counts().index)
        plt.title("Gender Breakdown")
        plt.xlabel("Gender")
        plt.ylabel("Count")
        save_plot(output_dir, "task1_gender_breakdown.png")

    if product_col and sales_col:
        top_products = df.groupby(product_col)[sales_col].sum().sort_values(ascending=False).head(10)
        plt.figure(figsize=(12, 6))
        sns.barplot(x=top_products.values, y=top_products.index)
        plt.title("Top 10 Best-Selling Products by Revenue")
        plt.xlabel("Revenue")
        plt.ylabel("Product")
        save_plot(output_dir, "task1_top_products.png")

    if category_col and sales_col:
        category_revenue = df.groupby(category_col)[sales_col].sum().sort_values(ascending=False)
        plt.figure(figsize=(10, 5))
        sns.barplot(x=category_revenue.index, y=category_revenue.values)
        plt.title("Revenue by Product Category")
        plt.xlabel("Category")
        plt.ylabel("Revenue")
        plt.xticks(rotation=30, ha="right")
        save_plot(output_dir, "task1_category_revenue.png")

    if numeric_df.shape[1] >= 2:
        plt.figure(figsize=(10, 7))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Matrix")
        save_plot(output_dir, "task1_correlation_heatmap.png")

    if category_col and sales_col and date_col:
        temp = df.copy()
        temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
        temp["month"] = temp[date_col].dt.to_period("M").astype(str)
        pivot = temp.pivot_table(index="month", columns=category_col, values=sales_col, aggfunc="sum")
        plt.figure(figsize=(12, 6))
        pivot.plot(ax=plt.gca())
        plt.title("Monthly Revenue by Category")
        plt.xlabel("Month")
        plt.ylabel("Revenue")
        plt.xticks(rotation=45, ha="right")
        save_plot(output_dir, "task1_extra_monthly_category_revenue.png")

    print(f"\nCharts saved to: {output_dir.resolve()}")
    print("\nConclusion template:")
    print("1. Focus inventory and promotion on the highest-revenue products/categories.")
    print("2. Use seasonal/monthly sales peaks to plan discounts and stock levels.")
    print("3. Target customer groups with personalized offers based on age/gender patterns.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to retail sales CSV file")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    main(args.csv, args.output_dir)
