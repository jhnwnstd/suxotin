"""
Spectral Decomposition Algorithm for vowel/consonant classification.

Based on: Thaine & Penn (2017), "Vowel and Consonant Classification
through Spectral Decomposition".
"""

import os
from typing import List, Optional, Tuple

import numpy as np

from utils import get_language_data, load_languages, preprocess_text


def vowel_consonant_classification(
    V: np.ndarray,
    letters: List[str],
    letters_count: np.ndarray,
) -> Tuple[List[str], List[str]]:
    """
    Classify letters by the sign of the second right singular vector.

    Uses the most frequent letter to orient the clusters: the cluster
    containing it is labeled vowels (the most frequent letter is usually
    a vowel across languages). Cluster size is used as a tiebreaker.
    """
    V1 = V[:, 1]
    most_freq_letter = letters[int(np.argmax(letters_count))]

    cluster_pos = [letter for idx, letter in enumerate(letters) if V1[idx] > 0]
    cluster_neg = [letter for idx, letter in enumerate(letters) if V1[idx] < 0]

    if not cluster_pos or not cluster_neg:
        return cluster_pos, cluster_neg

    # Primary: cluster containing the most frequent letter = vowels
    if most_freq_letter in cluster_pos:
        return cluster_pos, cluster_neg
    return cluster_neg, cluster_pos


def algorithm1(
    corpus: List[str], max_words: Optional[int] = None
) -> Tuple[List[str], List[str]]:
    """
    Classify letters via spectral decomposition of a letter-by-p-frame matrix.

    Builds a binary matrix A where rows are unique p-frames (trigram contexts)
    and columns are letters. SVD of A separates vowels from consonants via the
    second right singular vector.
    """
    if max_words is not None:
        corpus = corpus[:max_words]

    letters = sorted(set("".join(corpus)))
    num_letters = len(letters)
    letter_to_index = {letter: idx for idx, letter in enumerate(letters)}
    letters_count = np.zeros(num_letters, dtype=int)

    p_frame_indices: dict[tuple[str, str, str], int] = {}
    A_entries: List[Tuple[int, int]] = []

    for word in corpus:
        padded = f" {word} "
        for i in range(1, len(padded) - 1):
            p_frame = (padded[i - 1], "*", padded[i + 1])
            if p_frame not in p_frame_indices:
                p_frame_indices[p_frame] = len(p_frame_indices)

            letter_idx = letter_to_index.get(padded[i])
            if letter_idx is not None:
                letters_count[letter_idx] += 1
                A_entries.append((p_frame_indices[p_frame], letter_idx))

    if not A_entries:
        return [], []

    num_p_frames = len(p_frame_indices)
    A = np.zeros((num_p_frames, num_letters), dtype=float)
    rows, cols = zip(*A_entries)
    A[rows, cols] = 1

    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T

    return vowel_consonant_classification(V, letters, letters_count)


def run(
    language_code: str,
    language_name: str,
    test_folder: str,
    max_words: Optional[int] = None,
) -> str:
    """Run the algorithm for a single language and return formatted results."""
    lines = [f"\nProcessing language: {language_name} ({language_code})"]

    text = get_language_data(language_code, test_folder)
    if not text:
        lines.append(f"Error: No text data available for {language_name}")
        return "\n".join(lines)

    corpus = preprocess_text(text)
    if not corpus:
        lines.append(f"No valid words found in {language_name}. Skipping.")
        return "\n".join(lines)

    vowels, consonants = algorithm1(corpus, max_words)
    if not vowels and not consonants:
        lines.append(f"No vowels or consonants identified in {language_name}.")
        return "\n".join(lines)

    lines.append(
        f"Vowels in {language_name}: {', '.join(sorted(set(vowels)))}"
    )
    lines.append(
        f"Consonants in {language_name}: {', '.join(sorted(set(consonants)))}"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    test_folder = os.path.join("Test", "data")
    lang_df = load_languages(test_folder)

    output_filename = "algorithm1_output.txt"
    with open(output_filename, "w", encoding="utf-8") as f:
        for _, row in lang_df.iterrows():
            f.write(run(row["code"], row["language"], test_folder) + "\n")

    print(f"Processing complete. Results saved to {output_filename}")
