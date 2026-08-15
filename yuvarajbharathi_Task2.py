import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


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

    df = pd.read_csv(csv_path, encoding_errors="ignore")
    print("\n--- Initial Inspection ---")
    print("Shape:", df.shape)
    print(df.head())
    print("\nNull values:\n", df.isna().sum())
    print("\nData types:\n", df.dtypes)

    customer_col = find_column(df.columns, ["customerid", "customer id", "customer", "client"])
    date_col = find_column(df.columns, ["invoice date", "date"])
    quantity_col = find_column(df.columns, ["quantity", "qty"])
    price_col = find_column(df.columns, ["unitprice", "unit price", "price", "amount"])
    total_col = find_column(df.columns, ["total", "sales", "revenue", "amount"])

    if customer_col is None:
        raise ValueError("Customer column not found. Rename your customer ID column or update customer_col manually.")

    work = df.copy()
    work = work.dropna(subset=[customer_col])

    if total_col:
        work["TotalAmount"] = pd.to_numeric(work[total_col], errors="coerce")
    elif quantity_col and price_col:
        work["TotalAmount"] = pd.to_numeric(work[quantity_col], errors="coerce") * pd.to_numeric(work[price_col], errors="coerce")
    else:
        raise ValueError("Could not calculate monetary value. Need total/sales column or quantity and price columns.")

    work = work.dropna(subset=["TotalAmount"])
    work = work[work["TotalAmount"] > 0]

    if date_col:
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
        snapshot_date = work[date_col].max() + pd.Timedelta(days=1)
        rfm = work.groupby(customer_col).agg(
            Recency=(date_col, lambda x: (snapshot_date - x.max()).days),
            Frequency=(date_col, "count"),
            Monetary=("TotalAmount", "sum"),
        )
    else:
        rfm = work.groupby(customer_col).agg(
            Frequency=("TotalAmount", "count"),
            Monetary=("TotalAmount", "sum"),
        )
        rfm["Recency"] = 0

    rfm = rfm.reset_index()
    rfm["AveragePurchaseValue"] = rfm["Monetary"] / rfm["Frequency"]
    print("\n--- RFM Summary ---")
    print(rfm.describe())

    features = ["Recency", "Frequency", "Monetary"]
    X = rfm[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    inertias = []
    k_values = range(1, 11)
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X_scaled)
        inertias.append(model.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, inertias, marker="o")
    plt.title("Elbow Method for Optimal K")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    save_plot(output_dir, "task2_elbow_method.png")

    optimal_k = 4
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    rfm["Cluster"] = kmeans.fit_predict(X_scaled)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=rfm, x="Recency", y="Monetary", hue="Cluster", palette="tab10")
    plt.title("Customer Clusters: Recency vs Monetary")
    save_plot(output_dir, "task2_clusters_recency_monetary.png")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=rfm, x="Frequency", y="Monetary", hue="Cluster", palette="tab10")
    plt.title("Customer Clusters: Frequency vs Monetary")
    save_plot(output_dir, "task2_clusters_frequency_monetary.png")

    plt.figure(figsize=(8, 5))
    sns.countplot(data=rfm, x="Cluster")
    plt.title("Number of Customers per Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Customers")
    save_plot(output_dir, "task2_customers_per_cluster.png")

    cluster_profile = rfm.groupby("Cluster")[features + ["AveragePurchaseValue"]].mean().round(2)
    print("\n--- Cluster Profile ---")
    print(cluster_profile)
    cluster_profile.to_csv(output_dir / "task2_cluster_profile.csv")
    rfm.to_csv(output_dir / "task2_customer_segments.csv", index=False)

    print(f"\nOutputs saved to: {output_dir.resolve()}")
    print("\nMarketing action template:")
    print("- Low recency, high frequency, high monetary: reward loyal VIP customers.")
    print("- High recency, low frequency: run win-back campaigns.")
    print("- Low monetary, high frequency: promote bundles or upsell offers.")
    print("- New/low activity customers: send onboarding and first-discount campaigns.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to customer transaction CSV file")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    main(args.csv, args.output_dir)
