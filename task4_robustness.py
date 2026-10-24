"""Supplementary Task 4 analysis: sensitivity to stratified CV split seed.

Repeats the identical two-model protocol with 10 distinct split seeds. Scores
from these repetitions are NOT independent estimates or real-world accuracy.
"""

import csv
from pathlib import Path
from statistics import mean, stdev

from sklearn.metrics import accuracy_score, f1_score
from task4_text_classification import load_data, evaluate

ROOT = Path(__file__).resolve().parent
SEEDS = (2, 7, 11, 19, 23, 42, 59, 73, 101, 2026)


def main():
    texts, labels, _ = load_data(ROOT / "7042SCN_Text_Classification_Dataset.txt")
    rows = []
    for seed in SEEDS:
        prediction, _, _ = evaluate(texts, labels, seed=seed)
        for library, pred in prediction.items():
            rows.append({"seed": seed, "library": library,
                         "accuracy": accuracy_score(labels, pred),
                         "macro_f1": f1_score(labels, pred, average="macro")})

    path = ROOT / "outputs" / "task4_split_sensitivity.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "library", "accuracy", "macro_f1"])
        writer.writeheader()
        writer.writerows(rows)

    print("TASK 4: SPLIT-SENSITIVITY CHECK (10 REPEATED FIVE-FOLD PROTOCOLS)")
    print("Seeds:", SEEDS)
    for library in ("scikit-learn", "NLTK"):
        xs = [row for row in rows if row["library"] == library]
        accuracies = [row["accuracy"] for row in xs]
        f1s = [row["macro_f1"] for row in xs]
        print(f"{library}: accuracy mean {mean(accuracies):.3f}, sd {stdev(accuracies):.3f}, "
              f"range {min(accuracies):.3f}–{max(accuracies):.3f}; "
              f"macro-F1 mean {mean(f1s):.3f}, range {min(f1s):.3f}–{max(f1s):.3f}")
    print("Scores are resampling sensitivity diagnostics on the SAME 75 messages;")
    print("they must not be presented as independent observations or deployment estimates.")
    print("Saved: outputs/task4_split_sensitivity.csv")


if __name__ == "__main__":
    main()
