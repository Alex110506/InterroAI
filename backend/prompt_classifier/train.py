"""
Train the local prompt-complexity classifier that powers the Auto router.

Pipeline: TF-IDF (word 1-2 grams + char 3-5 grams) → Logistic Regression.

Dataset:   backend/prompt_classifier/prompt_examples_dataset.csv
           1450 rows, 3-class label `complexity` ∈ {low, medium, high}.
           For each row we stack `task_description`, `bad_prompt`, and
           `good_prompt` as separate training examples sharing the row's
           label — this triples the corpus and exposes the classifier to
           both terse (label-like) and verbose (engineered) phrasings.

Output:    backend/prompt_classifier/router_model.joblib   (commit this)

Run:       cd backend && python -m prompt_classifier.train
"""
from __future__ import annotations

import csv
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

_HERE = Path(__file__).parent
_DATA_PATH = _HERE / "prompt_examples_dataset.csv"
_MODEL_PATH = _HERE / "router_model.joblib"

_COMPLEXITY_TO_TIER: dict[str, str] = {
    "low":    "gpt-5.4-mini",
    "medium": "gpt-5.4-high-effort",
    "high":   "gpt-5.5-high-effort",
}

# Stack these columns as separate examples sharing the same label.
_TEXT_COLUMNS = ("task_description", "bad_prompt", "good_prompt")


def _load_rows() -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    with _DATA_PATH.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            complexity = (row.get("complexity") or "").strip().lower()
            tier = _COMPLEXITY_TO_TIER.get(complexity)
            if tier is None:
                continue
            for col in _TEXT_COLUMNS:
                text = (row.get(col) or "").strip()
                if text:
                    texts.append(text)
                    labels.append(tier)
    return texts, labels


def _build_pipeline() -> Pipeline:
    # Two TF-IDF views: word-level captures lexical content, char-level catches
    # morphology and short/noisy prompts. FeatureUnion concatenates them.
    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=20_000,
        sublinear_tf=True,
        min_df=2,
        stop_words="english",
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=30_000,
        sublinear_tf=True,
        min_df=2,
    )
    return Pipeline([
        ("features", FeatureUnion([("word", word_vec), ("char", char_vec)])),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            C=2.0,
            solver="lbfgs",
        )),
    ])


def main() -> None:
    if not _DATA_PATH.exists():
        raise SystemExit(f"Dataset not found at {_DATA_PATH}")

    X, y = _load_rows()
    print(f"Loaded {len(X)} examples across {len(set(y))} classes.")
    for cls in sorted(set(y)):
        print(f"  {cls:<24s} {y.count(cls):>5d}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y,
    )

    pipeline = _build_pipeline()
    pipeline.fit(X_train, y_train)

    print(f"\nTRAIN accuracy: {pipeline.score(X_train, y_train):.3f}")
    print(f"TEST  accuracy: {pipeline.score(X_test, y_test):.3f}")
    print(classification_report(y_test, pipeline.predict(X_test), digits=3))

    joblib.dump(pipeline, _MODEL_PATH, compress=3)
    size_kb = _MODEL_PATH.stat().st_size / 1024
    print(f"Saved → {_MODEL_PATH}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
