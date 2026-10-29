"""News preparation and models shared by both NLP walkthroughs.

TF-IDF is fitted on training text only. No downloads or pretrained vectors
are required. The small stopword list is illustrative, not comprehensive.
"""

from pathlib import Path
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
STOPWORDS = frozenset("र वा को का की मा ले लाई बाट तथा पनि यो त्यो एक छ छन् हो".split())
CATEGORIES = ("खेलकुद समाचार", "बिजनेस समाचार", "मनोरञ्जन एप", "देश")


def load_suffixes(path=None):
    """Longest suffixes first; UTF-8 is explicit for Windows."""
    path = Path(path) if path else ROOT / "NLP" / "suffix.txt"
    return tuple(sorted(set(path.read_text(encoding="utf-8").split()), key=lambda s: (-len(s), s)))


def clean_text(text, suffixes=()):
    """Keep Devanagari words, remove digits/punctuation and basic stopwords.

    Suffix removal is a simple teaching heuristic, not a linguistic analyzer.
    At least two characters of the stem must remain.
    """
    text = re.sub(r"[^\u0900-\u097f\s]", " ", str(text))
    text = re.sub(r"[\u0964-\u096f]", " ", text)
    words = []
    for word in text.split():
        if word in STOPWORDS:
            continue
        for suffix in suffixes:
            if len(word) > len(suffix) + 1 and word.endswith(suffix):
                word = word[:-len(suffix)]
                break
        words.append(word)
    return " ".join(words)


def load_news(path=None, max_per_class=200, random_state=42):
    """Use a balanced, bounded sample of the four original workshop topics.

    Scores describe this sample, not the original population distribution.
    Exact duplicate articles are removed before splitting to prevent leakage.
    """
    path = Path(path) if path else ROOT / "NLP" / "data.csv"
    frame = pd.read_csv(path, usecols=["News", "Category", "Language"])
    frame = frame.dropna().loc[lambda df: df["Language"].eq("ne")].copy()
    frame = frame.drop_duplicates("News")
    frame = frame.loc[frame["Category"].isin(CATEGORIES)]
    # Bound preprocessing time too, not just model training time.
    frame = pd.concat([
        group.sample(n=min(len(group), max_per_class * 2), random_state=random_state)
        for _, group in frame.groupby("Category", sort=True)
    ]) if len(frame) else frame
    suffixes = load_suffixes()
    frame["clean_news"] = frame["News"].map(lambda text: clean_text(text, suffixes))
    frame = frame.loc[frame["clean_news"].str.len().gt(0)]
    frame = frame.drop_duplicates("clean_news")
    categories = frame["Category"].unique()
    if len(categories) < 2:
        raise ValueError("News data must contain at least two Nepali categories")
    count = min(max_per_class, int(frame["Category"].value_counts().min()))
    if count < 4:
        raise ValueError("Each category needs at least four distinct articles")
    return pd.concat([
        group.sample(n=count, random_state=random_state)
        for _, group in frame.groupby("Category", sort=True)
    ]).reset_index(drop=True)


def split_news(frame):
    return train_test_split(
        frame["clean_news"], frame["Category"], test_size=0.25,
        stratify=frame["Category"], random_state=42,
    )


def build_model(neural=False):
    # Whitespace tokenization preserves Devanagari combining marks.
    vectorizer = TfidfVectorizer(tokenizer=str.split, token_pattern=None,
                                 lowercase=False, max_features=2000, sublinear_tf=True)
    estimator = (MLPClassifier(hidden_layer_sizes=(32,), solver="lbfgs", alpha=1,
                               max_iter=200, random_state=42)
                 if neural else LogisticRegression(max_iter=1000, random_state=42))
    return make_pipeline(vectorizer, estimator)


def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "report": classification_report(y_test, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=model.classes_),
        "predictions": predictions,
    }
