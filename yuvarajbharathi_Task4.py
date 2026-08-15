import argparse
import re
import string
from pathlib import Path

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud


sns.set_theme(style="whitegrid")


def prepare_nltk():
    for package in ["stopwords", "wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"corpora/{package}")
        except LookupError:
            nltk.download(package)


def clean_text(text, stop_words, lemmatizer):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words and not word.isdigit()]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)


def save_plot(output_dir, filename):
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=160)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, labels, title, output_dir, filename):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    save_plot(output_dir, filename)


def load_sentiment_data(file_path, text_column=None, label_column=None, sample_size=None):
    file_path = Path(file_path)
    if file_path.suffix.lower() == ".txt":
        rows = []
        with file_path.open("r", encoding="utf-8", errors="ignore") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("__label__"):
                    label, text = line.split(" ", 1)
                    label = label.replace("__label__", "")
                    label = "positive" if label == "2" else "negative"
                    rows.append({"text": text, "sentiment": label})
                if sample_size and len(rows) >= sample_size:
                    break
        return pd.DataFrame(rows), "text", "sentiment"

    df = pd.read_csv(file_path, encoding_errors="ignore")
    if text_column not in df.columns or label_column not in df.columns:
        raise ValueError(f"Columns not found. Available columns: {list(df.columns)}")
    return df[[text_column, label_column]].dropna(), text_column, label_column


def main(csv_path, text_column, label_column, output_dir, sample_size):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prepare_nltk()

    df, text_column, label_column = load_sentiment_data(csv_path, text_column, label_column, sample_size)
    df = df[[text_column, label_column]].dropna()
    df[label_column] = df[label_column].astype(str).str.strip().str.lower()

    print("\n--- Class Distribution ---")
    print(df[label_column].value_counts())

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=label_column, order=df[label_column].value_counts().index)
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    save_plot(output_dir, "task4_sentiment_distribution.png")

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    df["clean_text"] = df[text_column].apply(lambda x: clean_text(x, stop_words, lemmatizer))

    for sentiment in df[label_column].unique():
        text = " ".join(df.loc[df[label_column] == sentiment, "clean_text"])
        if not text.strip():
            continue
        wordcloud = WordCloud(width=900, height=450, background_color="white").generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"WordCloud: {sentiment}")
        save_plot(output_dir, f"task4_wordcloud_{sentiment}.png")

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df[label_column],
        test_size=0.2,
        random_state=42,
        stratify=df[label_column] if df[label_column].value_counts().min() >= 2 else None,
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
    }

    labels = sorted(df[label_column].unique())
    results = []
    predictions = {}

    for model_name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        predictions[model_name] = y_pred

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        results.append({
            "model": model_name,
            "accuracy": accuracy,
            "macro_precision": report["macro avg"]["precision"],
            "macro_recall": report["macro avg"]["recall"],
            "macro_f1": report["macro avg"]["f1-score"],
        })

        print(f"\n--- {model_name} ---")
        print("Accuracy:", round(accuracy, 4))
        print(classification_report(y_test, y_pred, zero_division=0))
        plot_confusion_matrix(
            y_test,
            y_pred,
            labels,
            f"Confusion Matrix: {model_name}",
            output_dir,
            f"task4_confusion_matrix_{model_name.lower().replace(' ', '_')}.png",
        )

    results_df = pd.DataFrame(results).sort_values("macro_f1", ascending=False)
    results_df.to_csv(output_dir / "task4_model_results.csv", index=False)
    print("\n--- Model Comparison ---")
    print(results_df)

    best_model_name = results_df.iloc[0]["model"]
    best_predictions = predictions[best_model_name]
    error_df = pd.DataFrame({
        "text": X_test.values,
        "actual": y_test.values,
        "predicted": best_predictions,
    })
    error_df = error_df[error_df["actual"] != error_df["predicted"]].head(5)
    error_df.to_csv(output_dir / "task4_error_analysis_examples.csv", index=False)
    print("\n--- 5 Misclassified Examples ---")
    print(error_df)

    print(f"\nOutputs saved to: {output_dir.resolve()}")
    print(f"\nBest model: {best_model_name}")
    print("Real-world use: monitor customer reviews/social media to identify negative feedback early.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to sentiment CSV or fastText TXT file")
    parser.add_argument("--text-column", help="Name of the text/review/tweet column for CSV files")
    parser.add_argument("--label-column", help="Name of the sentiment label column for CSV files")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--sample-size", type=int, default=50000, help="Rows to read from large TXT datasets")
    args = parser.parse_args()
    main(args.csv, args.text_column, args.label_column, args.output_dir, args.sample_size)
