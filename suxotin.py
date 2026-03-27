"""
Sukhotin's algorithm for vowel/consonant classification.

Uses character adjacency frequencies to iteratively identify vowels
based on the observation that vowels and consonants tend to alternate.

Key improvement: builds the adjacency matrix from within-word character
bigrams only (no spaces/whitespace), which prevents word-boundary
consonants from being inflated.
"""

import os
import time
from typing import Dict, List, Set, Tuple

import numpy as np
from scipy.sparse import coo_matrix

from utils import get_language_data, load_languages


def create_frequency_matrix(
    text: str,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Build a symmetric adjacency matrix from within-word character bigrams.

    Only counts bigrams between alphabetic characters within the same word.
    Self-adjacencies (e.g. 'aa') are excluded.
    """
    words = text.lower().split()
    all_chars: Set[str] = set()
    for word in words:
        for ch in word:
            if ch.isalpha():
                all_chars.add(ch)

    unique_chars = sorted(all_chars)
    char_to_index = {ch: i for i, ch in enumerate(unique_chars)}
    size = len(unique_chars)

    if size < 2:
        return np.zeros((size, size), dtype=int), char_to_index

    left_indices: List[int] = []
    right_indices: List[int] = []
    for word in words:
        alpha = [ch for ch in word if ch.isalpha()]
        for i in range(len(alpha) - 1):
            a, b = char_to_index[alpha[i]], char_to_index[alpha[i + 1]]
            if a != b:
                left_indices.append(a)
                right_indices.append(b)

    if not left_indices:
        return np.zeros((size, size), dtype=int), char_to_index

    data = np.ones(len(left_indices), dtype=int)
    row = np.array(left_indices)
    col = np.array(right_indices)
    adj = coo_matrix((data, (row, col)), shape=(size, size))
    adj = adj + adj.T
    return adj.toarray(), char_to_index


def classify_vowels(
    sums: np.ndarray,
    matrix: np.ndarray,
    index_to_char: Dict[int, str],
) -> Tuple[List[str], List[str], np.ndarray]:
    """
    Iteratively classify characters as vowels or consonants.

    At each step, the character with the highest remaining adjacency sum
    is classified as a vowel (if its sum is positive). Its column
    contributions are then subtracted. Stops when no character has a
    positive remaining sum.

    Returns (vowels, consonants, adjusted_sums).
    """
    num_chars = len(sums)
    remaining_sums = sums.astype(float)
    adjusted_sums = np.zeros(num_chars, dtype=float)
    vowels: List[str] = []
    consonants: List[str] = []

    for _ in range(num_chars):
        max_idx = int(np.argmax(remaining_sums))
        adjusted_sums[max_idx] = remaining_sums[max_idx]

        if remaining_sums[max_idx] <= 0:
            # All remaining characters are consonants
            for i in range(num_chars):
                if remaining_sums[i] != -np.inf:
                    consonants.append(index_to_char[i])
            break

        vowels.append(index_to_char[max_idx])
        remaining_sums -= matrix[:, max_idx] * 2
        remaining_sums[max_idx] = -np.inf

    return vowels, consonants, adjusted_sums


def visualize_classification_scores(
    adjusted_sums: np.ndarray,
    index_to_char: Dict[int, str],
    vowels: List[str],
    language_name: str,
    output_folder: str,
) -> None:
    """Save a bar chart of classification scores to output_folder."""
    import matplotlib.pyplot as plt

    chars = [index_to_char[idx] for idx in range(len(adjusted_sums))]

    plt.figure(figsize=(15, 8))
    bars = plt.bar(range(len(adjusted_sums)), adjusted_sums, color="blue")
    for idx, char in enumerate(chars):
        if char in vowels:
            bars[idx].set_color("red")

    plt.axhline(y=0, color="green", linestyle="--", label="Threshold = 0")
    plt.title(
        f"Character Classification Scores for {language_name}\n"
        "Red: Vowels, Blue: Consonants"
    )
    plt.xlabel("Characters")
    plt.ylabel("Adjusted Sum Scores")
    plt.xticks(range(len(chars)), chars, rotation=45)
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()

    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(
        os.path.join(output_folder, f"{language_name}_scores.png"),
        bbox_inches="tight",
    )
    plt.close()


def run(
    language_code: str,
    language_name: str,
    test_folder: str,
) -> str:
    """Run Sukhotin's algorithm for a single language and return formatted results."""
    lines = [f"\nProcessing language: {language_name} ({language_code})"]

    text = get_language_data(language_code, test_folder)
    if not text:
        lines.append(f"Error: No text data available for {language_name}")
        return "\n".join(lines)

    start_time = time.perf_counter()

    matrix, char_to_index = create_frequency_matrix(text)
    sums = matrix.sum(axis=1)
    index_to_char = {idx: ch for ch, idx in char_to_index.items()}

    vowels, consonants, adjusted_sums = classify_vowels(
        sums, matrix, index_to_char
    )

    visualize_classification_scores(
        adjusted_sums, index_to_char, vowels, language_name, "visualizations"
    )

    # Filter whitespace-only characters and sort
    vowels = sorted(v for v in vowels if v.strip())
    consonants = sorted(c for c in consonants if c.strip())

    elapsed = time.perf_counter() - start_time
    lines.append(f"Vowels in {language_name}: {', '.join(vowels)}")
    lines.append(f"Consonants in {language_name}: {', '.join(consonants)}")
    lines.append(f"Execution time: {elapsed:.4f} seconds")
    lines.append(
        f"Visualization saved as: visualizations/{language_name}_scores.png"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    test_folder = os.path.join("Test", "data")
    lang_df = load_languages(test_folder)

    output_filename = "suxotin_algorithm_output.txt"
    with open(output_filename, "w", encoding="utf-8") as f:
        results: List[str] = []
        for _, row in lang_df.iterrows():
            results.append(run(row["code"], row["language"], test_folder))
        f.write("\n".join(results))

    print(f"Processing complete. Results saved to {output_filename}")
