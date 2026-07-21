import os
import re
from typing import Optional, Sequence

import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer


# Ensure the tokenizer resources are available (safe to call repeatedly).
nltk.download("punkt", quiet=True)

DATASET_DIR = os.path.join("dataset")
UNPREPARED_DIR = os.path.join(DATASET_DIR, "unprepared")
ENGLISH_PATH = os.path.join(UNPREPARED_DIR, "english_spam_dataset.csv")
FILIPINO_PATH = os.path.join(UNPREPARED_DIR, "filipino_spam_dataset.csv")
OUTPUT_PATH = os.path.join(DATASET_DIR, "final_spam_ham_dataset.csv")

LABEL_CANDIDATES: Sequence[str] = (
    "label",
    "category",
    "class",
    "target",
    "spam",
    "is_spam",
    "v1",
)

TEXT_CANDIDATES: Sequence[str] = (
    "text",
    "message",
    "sms",
    "content",
    "body",
    "v2",
)

STEMMER = SnowballStemmer("english")


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/punctuation, tokenize, and stem."""
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)

    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = word_tokenize(text)
    stemmed = [STEMMER.stem(tok) for tok in tokens if tok.isalnum()]
    return " ".join(stemmed)


def normalize_label(raw_label) -> int:
    """Normalize different label notations to {0: ham, 1: spam}."""
    if isinstance(raw_label, str):
        raw_label = raw_label.strip().lower()

    mapping = {"spam": 1, "ham": 0, "1": 1, "0": 0}
    return mapping.get(raw_label, 1 if str(raw_label).strip() == "1" else 0)


def read_csv_with_fallback(path: str) -> pd.DataFrame:
    """Try multiple encodings to deal with mixed-source CSV files."""
    encodings = (None, "utf-8", "latin-1", "ISO-8859-1")
    last_error: Optional[Exception] = None

    for encoding in encodings:
        try:
            if encoding is None:
                return pd.read_csv(path)
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc

    raise UnicodeDecodeError(
        last_error.encoding if isinstance(last_error, UnicodeDecodeError) else "utf-8",
        b"",
        0,
        0,
        f"Unable to decode file {path} with tried encodings.",
    )


def load_and_clean_dataset(
    path: str,
    language: str,
    default_label: Optional[int] = None,
) -> pd.DataFrame:
    """Load CSV, detect columns, clean text, and normalize labels."""
    df = read_csv_with_fallback(path)
    df.columns = [col.strip().lower() for col in df.columns]

    label_col = next((c for c in df.columns if c in LABEL_CANDIDATES), None)
    text_col = next((c for c in df.columns if c in TEXT_CANDIDATES), None)

    if text_col is None:
        raise ValueError(
            f"Unable to identify the text column in {path}. "
            f"Columns found: {df.columns.tolist()}"
        )

    if label_col is None:
        if default_label is None:
            raise ValueError(
                f"No label column detected in {path} and no default_label provided."
            )
        df["label"] = default_label
    else:
        df["label"] = df[label_col].apply(normalize_label)

    df["text"] = df[text_col].astype(str).apply(clean_text)
    df["language"] = language

    # Keep only the unified schema plus optional metadata
    df = df[["label", "text", "language"]].copy()
    df = df[df["text"].str.strip().ne("")]
    return df


def print_summary(name: str, df: pd.DataFrame) -> None:
    """Print simple summary stats for a dataset."""
    print(f"\n{name} dataset summary")
    print("-" * 30)
    label_counts = df["label"].value_counts().rename({1: "spam", 0: "ham"})
    print(label_counts)
    sample_count = min(3, len(df))
    if sample_count:
        print("\nSample cleaned rows:")
        print(df[["label", "text"]].sample(sample_count, random_state=42))


def main() -> None:
    os.makedirs(DATASET_DIR, exist_ok=True)

    if not os.path.exists(ENGLISH_PATH):
        raise FileNotFoundError(f"Missing English dataset at {ENGLISH_PATH}")
    if not os.path.exists(FILIPINO_PATH):
        raise FileNotFoundError(f"Missing Filipino dataset at {FILIPINO_PATH}")

    english_df = load_and_clean_dataset(ENGLISH_PATH, language="english")
    filipino_df = load_and_clean_dataset(
        FILIPINO_PATH,
        language="filipino",
        default_label=1,  # Filipino dataset contains spam alerts only.
    )

    print_summary("English", english_df)
    print_summary("Filipino", filipino_df)

    final_df = pd.concat([english_df, filipino_df], ignore_index=True)
    final_df = final_df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    # Only persist the required columns.
    final_df[["label", "text"]].to_csv(OUTPUT_PATH, index=False)

    print("\nMerged dataset summary")
    print("-" * 30)
    merged_counts = final_df["label"].value_counts().rename({1: "spam", 0: "ham"})
    print(merged_counts)

    sample_rows = min(5, len(final_df))
    if sample_rows:
        print("\nSample cleaned rows from merged dataset:")
        print(final_df[["label", "text"]].sample(sample_rows, random_state=123))

    print(f"\nSaved final dataset to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

