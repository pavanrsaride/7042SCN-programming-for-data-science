"""7042SCN Task 4B: reproducible, leakage-safe text classification comparison.

Student: Saride Pavan Rahul (17509497)
Input: 7042SCN_Text_Classification_Dataset.txt (tab-separated, text/label)
Methods:
    1. scikit-learn: TF-IDF word 1-2 grams + multinomial logistic regression.
    2. NLTK: binary word-presence features + Naive Bayes.
Evaluation: same five stratified, shuffled folds; each record predicted once.

Requirements: Python 3.10+, scikit-learn, nltk.
No external model files or NLTK corpus downloads are needed.

Run with:
    python task4_text_classification.py \
        7042SCN_Text_Classification_Dataset.txt
"""

import argparse
import csv
import json
import os
import re
from collections import Counter
from pathlib import Path

import nltk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

CATEGORIES = ("family", "other", "study", "visit", "work")
FOLDS = 5
SEED = 42


def load_data(path: Path):
    """Read and check the provided tab-separated dataset."""
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        if reader.fieldnames is None or set(reader.fieldnames) != {"text", "label"}:
            raise ValueError("Expected exactly two columns: text and label.")
        rows = list(reader)

    if not rows:
        raise ValueError("The input file has no records.")
    texts = [row["text"].strip() for row in rows]
    labels = [row["label"].strip() for row in rows]

    if any(not text or not label for text, label in zip(texts, labels)):
        raise ValueError("Empty text or label found.")
    if set(labels) != set(CATEGORIES):
        raise ValueError(f"Labels must be precisely: {', '.join(CATEGORIES)}")
    if len(set(text.casefold() for text in texts)) != len(texts):
        raise ValueError("Duplicate text detected; inspect before cross-validation.")

    counts = Counter(labels)
    if min(counts.values()) < FOLDS:
        raise ValueError("Each class must have at least five records.")
    return texts, labels, counts


def tokens(text):
    """Tokenizer independent of downloadable NLTK linguistic resources."""
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


def nltk_features(text, vocabulary):
    """Presence/absence of each training-vocabulary word; no test vocabulary fit."""
    observed = set(tokens(text))
    return {f"has_{word}": word in observed for word in vocabulary}


def predict_nltk(train_texts, train_labels, test_texts):
    """Train an NLTK Naive Bayes classifier on this fold only."""
    vocabulary = sorted({word for text in train_texts for word in tokens(text)})
    training = [
        (nltk_features(text, vocabulary), label)
        for text, label in zip(train_texts, train_labels)
    ]
    classifier = nltk.NaiveBayesClassifier.train(training)
    return [classifier.classify(nltk_features(text, vocabulary)) for text in test_texts]


def evaluate(texts, labels, seed=SEED):
    """Compute out-of-fold predictions from matching stratified splits."""
    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
    predictions = {"scikit-learn": [None] * len(texts), "NLTK": [None] * len(texts)}
    fold_ids = [0] * len(texts)
    fold_results = []

    for fold_number, (train_indices, test_indices) in enumerate(
        splitter.split(texts, labels), start=1
    ):
        train_texts = [texts[int(i)] for i in train_indices]
        train_labels = [labels[int(i)] for i in train_indices]
        test_texts = [texts[int(i)] for i in test_indices]
        test_labels = [labels[int(i)] for i in test_indices]

        # IMPORTANT: fit the TF-IDF vocabulary and model only on training records.
        sklearn_model = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
            LogisticRegression(max_iter=2000, random_state=seed),
        )
        sklearn_model.fit(train_texts, train_labels)
        sklearn_predictions = sklearn_model.predict(test_texts).tolist()

        # IMPORTANT: build NLTK's feature vocabulary from training records only.
        nltk_predictions = predict_nltk(train_texts, train_labels, test_texts)

        for index, sklearn_label, nltk_label in zip(
            test_indices, sklearn_predictions, nltk_predictions
        ):
            i = int(index)
            fold_ids[i] = fold_number
            predictions["scikit-learn"][i] = sklearn_label
            predictions["NLTK"][i] = nltk_label

        fold_results.append(
            (
                fold_number,
                accuracy_score(test_labels, sklearn_predictions),
                f1_score(test_labels, sklearn_predictions, average="macro"),
                accuracy_score(test_labels, nltk_predictions),
                f1_score(test_labels, nltk_predictions, average="macro"),
            )
        )

    if any(pred is None for values in predictions.values() for pred in values):
        raise AssertionError("Every record must receive exactly one OOF prediction.")
    return predictions, fold_ids, fold_results


def save_predictions(path, texts, labels, fold_ids, predictions):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["row_id", "fold", "text", "true_label", "sklearn_pred", "nltk_pred"]
        )
        for i, (text, label) in enumerate(zip(texts, labels)):
            writer.writerow(
                [i + 1, fold_ids[i], text, label,
                 predictions["scikit-learn"][i], predictions["NLTK"][i]]
            )


def export_performance(path, labels, predictions, fold_results):
    """Machine-readable metrics supporting reproducible report tables."""
    summary = {
        "evaluation": "5-fold stratified out-of-fold",
        "folds": FOLDS,
        "seed": SEED,
        "n_records": len(labels),
        "classes": list(CATEGORIES),
        "methods": {},
        "fold_results": [
            {"fold": fold, "sklearn_accuracy": acc_sk, "sklearn_macro_f1": f1_sk,
             "nltk_accuracy": acc_nl, "nltk_macro_f1": f1_nl}
            for fold, acc_sk, f1_sk, acc_nl, f1_nl in fold_results
        ],
    }
    for method, pred in predictions.items():
        summary["methods"][method] = {
            "accuracy": float(accuracy_score(labels, pred)),
            "macro_f1": float(f1_score(labels, pred, average="macro")),
            "correct": sum(actual == forecast for actual, forecast in zip(labels, pred)),
            "incorrect": sum(actual != forecast for actual, forecast in zip(labels, pred)),
            "report": classification_report(labels, pred, labels=CATEGORIES,
                                            output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(labels, pred, labels=CATEGORIES).tolist(),
        }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def save_confusion_figures(figures_directory, labels, predictions):
    """Generate reproducible image figures from observed, out-of-fold results."""
    figures_directory.mkdir(parents=True, exist_ok=True)
    for method, pred in predictions.items():
        matrix = confusion_matrix(labels, pred, labels=CATEGORIES)
        fig, ax = plt.subplots(figsize=(7, 6))
        figure = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=15)
        fig.colorbar(figure, ax=ax, label="Number of enquiries")
        for i, row in enumerate(matrix):
            for j, count in enumerate(row):
                ax.text(j, i, str(int(count)), ha="center", va="center",
                        color="white" if count > 8 else "black", fontsize=10)
        ax.set_xticks(range(len(CATEGORIES)), CATEGORIES, rotation=30, ha="right")
        ax.set_yticks(range(len(CATEGORIES)), CATEGORIES)
        ax.set_xlabel("Predicted category")
        ax.set_ylabel("Actual category")
        ax.set_title(f"{method}: out-of-fold confusion matrix")
        fig.tight_layout()
        path = figures_directory / f"task4_confusion_{method.replace('-', '_')}.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data_file",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("7042SCN_Text_Classification_Dataset.txt"),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(__file__).parent / "outputs" / "task4_predictions.csv",
    )
    parser.add_argument("--metrics", type=Path,
                        default=Path(__file__).parent / "outputs" / "task4_metrics.json")
    parser.add_argument("--figures-dir", type=Path,
                        default=Path(__file__).parent / "figures")
    args = parser.parse_args()
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    texts, labels, counts = load_data(args.data_file)

    print("TASK 4B: FIVE-CLASS IMMIGRATION ENQUIRY ROUTING")
    print("Dataset rows:", len(texts))
    print("Class counts:", dict(sorted(counts.items())))
    print("Duplicate texts: 0; missing values: 0")
    print("Evaluation: 5-fold stratified out-of-fold prediction, random_state=42")
    print("Vectorisers and NLTK vocabularies fitted within each training fold")

    predictions, fold_ids, fold_results = evaluate(texts, labels)

    print("\nFOLD RESULTS")
    print("Fold  sklearn_accuracy  sklearn_macro_f1  nltk_accuracy  nltk_macro_f1")
    for fold, acc_sk, f1_sk, acc_nl, f1_nl in fold_results:
        print(f"{fold:>4}  {acc_sk:>16.3f}  {f1_sk:>16.3f}  {acc_nl:>13.3f}  {f1_nl:>13.3f}")

    for library, pred in predictions.items():
        print(f"\n{library.upper()} — OUT-OF-FOLD PERFORMANCE")
        print(f"Accuracy: {accuracy_score(labels, pred):.3f}")
        print(f"Macro F1: {f1_score(labels, pred, average='macro'):.3f}")
        print(classification_report(labels, pred, labels=CATEGORIES, digits=3,
                                    zero_division=0))
        print("Confusion matrix: rows=true, columns=predicted")
        print("Label order:", ", ".join(CATEGORIES))
        for row in confusion_matrix(labels, pred, labels=CATEGORIES):
            print(" ".join(f"{int(value):3d}" for value in row))
        errors = [(i + 1, actual, predicted) for i, (actual, predicted)
                  in enumerate(zip(labels, pred)) if actual != predicted]
        print(f"Incorrectly routed records: {len(errors)}")
        print("First 10 error records (row, true label, predicted label):")
        for error in errors[:10]:
            print(" ", error)

    save_predictions(args.predictions, texts, labels, fold_ids, predictions)
    export_performance(args.metrics, labels, predictions, fold_results)
    save_confusion_figures(args.figures_dir, labels, predictions)
    print("\nSaved out-of-fold predictions:", os.path.relpath(args.predictions))
    print("Saved machine-readable metrics:", os.path.relpath(args.metrics))
    print("Saved two confusion-matrix PNG figures in:", os.path.relpath(args.figures_dir))
    print("NOTE: Scores describe this tiny supplied dataset, not deployment accuracy.")
    print("This demonstration routes queries only; it provides no legal advice.")


if __name__ == "__main__":
    main()
