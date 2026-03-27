import os
from typing import List

import pandas as pd


def get_language_data(language_code: str, test_folder: str) -> str:
    """
    Read text data for a language code from the data files.

    Returns an empty string if the file is not found or an error occurs.
    """
    filename = os.path.join(test_folder, language_code)
    if not os.path.isfile(filename):
        print(
            f"Error: File not found for language code {language_code} at {filename}"
        )
        return ""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(filename, "r", encoding="latin-1") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return ""


def load_languages(test_folder: str = "Test/data") -> pd.DataFrame:
    """
    Load lang_code.csv and filter to languages that have a matching data file.
    """
    df = pd.read_csv("lang_code.csv")
    df.columns = df.columns.str.strip()
    df["code"] = df["code"].astype(str).str.strip()
    df["language"] = df["language"].astype(str).str.strip()
    df.dropna(subset=["code", "language"], inplace=True)

    files = [
        f
        for f in os.listdir(test_folder)
        if os.path.isfile(os.path.join(test_folder, f))
    ]
    return df[df["code"].isin(files)]


def preprocess_text(text: str) -> List[str]:
    """
    Lowercase the text and split into words, keeping only alphabetic characters.
    """
    words = text.lower().split()
    return [
        "".join(ch for ch in word if ch.isalpha())
        for word in words
        if any(ch.isalpha() for ch in word)
    ]
