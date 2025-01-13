import time
from typing import Dict, List, Tuple
import os

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix


def get_language_data(language_code: str, test_folder: str) -> str:
    """
    Retrieve text data for the specified language code from the data files.

    Parameters
    ----------
    language_code : str
        The code for the language (e.g., "eng" for English).
    test_folder : str
        Path to the folder containing the language text files.

    Returns
    -------
    text : str
        The text data for the specified language. Returns an empty string if file not found or an error occurs.
    """
    filename = os.path.join(test_folder, language_code)
    if not os.path.isfile(filename):
        print(f"Error: File not found for language code {language_code} at {filename}")
        return ""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()
    except UnicodeDecodeError:
        # Fallback to 'latin-1'
        with open(filename, 'r', encoding='latin-1') as f:
            text = f.read()
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return ""
    return text


def preprocess_text(text: str) -> str:
    """
    Convert the input text to lowercase and remove all non-alphabetic characters except spaces.

    Parameters
    ----------
    text : str
        Original text data.

    Returns
    -------
    processed : str
        Preprocessed text (lowercase, only alphabets and spaces).
    """
    return ''.join(
        char.lower() if char.isalpha() or char.isspace() else ' '
        for char in text
    )


def create_frequency_matrix(text: str) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Create a symmetric frequency matrix of adjacent characters in the text.

    Parameters
    ----------
    text : str
        Preprocessed text containing only alphabets and spaces.

    Returns
    -------
    matrix : np.ndarray
        Symmetric adjacency matrix of size (N, N), where N is the number of unique characters.
    char_to_index : Dict[str, int]
        Mapping from each unique character to its index in the adjacency matrix.
    """
    # Get sorted list of unique characters for consistent indexing
    unique_chars = sorted(set(text))
    char_to_index = {char: idx for idx, char in enumerate(unique_chars)}
    size = len(unique_chars)

    # Convert text to a list of indices
    text_indices = [char_to_index[ch] for ch in text]
    # Edge case: if text_indices is too short, just return empty adjacency matrix
    if len(text_indices) < 2:
        return np.zeros((size, size), dtype=int), char_to_index

    # Build index pairs for adjacent characters, excluding repeats (a -> a)
    left_indices, right_indices = [], []
    for i in range(len(text_indices) - 1):
        if text_indices[i] != text_indices[i + 1]:
            left_indices.append(text_indices[i])
            right_indices.append(text_indices[i + 1])

    data = np.ones(len(left_indices), dtype=int)
    adjacency_matrix = coo_matrix(
        (data, (left_indices, right_indices)), shape=(size, size)
    )
    # Make the matrix symmetric
    adjacency_matrix = adjacency_matrix + adjacency_matrix.T
    # Convert to dense NumPy array
    matrix = adjacency_matrix.toarray()
    return matrix, char_to_index


def classify_vowels(
    sums: np.ndarray,
    matrix: np.ndarray,
    index_to_char: Dict[int, str],
    threshold_factor: float = 1.0
) -> Tuple[List[str], List[str]]:
    """
    Classify characters as vowels or consonants using an iterative adjacency-based approach 
    (also referred to as Sukhotin’s algorithm in some literature).

    Parameters
    ----------
    sums : np.ndarray
        Sum of adjacency counts for each character (size N).
    matrix : np.ndarray
        Adjacency (frequency) matrix of size (N, N).
    index_to_char : Dict[int, str]
        Mapping from matrix indices to the corresponding characters.
    threshold_factor : float, optional
        Factor to adjust the threshold for distinguishing vowels from consonants.

    Returns
    -------
    vowels : List[str]
        List of characters classified as vowels.
    consonants : List[str]
        List of characters classified as consonants.
    """
    num_chars = len(sums)
    remaining_sums = sums.astype(float)
    adjusted_sums = np.zeros(num_chars, dtype=float)
    vowels, consonants = [], []

    for _ in range(num_chars):
        max_idx = np.argmax(remaining_sums)
        char = index_to_char[max_idx]
        adjusted_sums[max_idx] = remaining_sums[max_idx]

        if adjusted_sums[max_idx] > 0:
            vowels.append(char)
        else:
            consonants.append(char)

        # Remove this character from further consideration
        remaining_sums -= matrix[:, max_idx] * 2
        remaining_sums[max_idx] = -np.inf

    # Determine threshold based on minimum positive vowel sum
    if vowels:
        vowel_indices = [idx for idx, c in index_to_char.items() if c in vowels]
        min_vowel_sum = adjusted_sums[vowel_indices].min()
        threshold = min_vowel_sum - threshold_factor * abs(min_vowel_sum)
    else:
        # If no vowels found, just use the global min as threshold
        threshold = adjusted_sums.min()

    # Reclassify consonants that exceed threshold
    reclassified_vowels, new_consonants = [], []
    for char in consonants:
        idx = next(k for k, c in index_to_char.items() if c == char)
        if adjusted_sums[idx] >= threshold:
            reclassified_vowels.append(char)
        else:
            new_consonants.append(char)

    vowels.extend(reclassified_vowels)
    consonants = new_consonants
    return vowels, consonants


def visualize_classification_scores(
    adjusted_sums: np.ndarray,
    index_to_char: Dict[int, str],
    vowels: List[str],
    threshold: float,
    language_name: str,
    output_folder: str
) -> None:
    """
    Visualize character classification scores and threshold for each character.

    Parameters
    ----------
    adjusted_sums : np.ndarray
        The array of final scores for each character.
    index_to_char : Dict[int, str]
        Mapping from matrix indices to characters.
    vowels : List[str]
        List of characters classified as vowels.
    threshold : float
        Computed threshold for classification.
    language_name : str
        A descriptive string for the language name (e.g., "English").
    output_folder : str
        Path to the folder where the visualization will be saved.
    """
    import matplotlib.pyplot as plt

    chars = [index_to_char[idx] for idx in range(len(adjusted_sums))]
    scores = adjusted_sums

    plt.figure(figsize=(15, 8))
    bars = plt.bar(range(len(scores)), scores, color='blue')

    # Color vowel bars in red
    for idx, char in enumerate(chars):
        if char in vowels:
            bars[idx].set_color('red')

    plt.axhline(y=threshold, color='green', linestyle='--', label=f'Threshold = {threshold:.2f}')
    plt.title(f'Character Classification Scores for {language_name}\nRed: Vowels, Blue: Consonants')
    plt.xlabel('Characters')
    plt.ylabel('Adjusted Sum Scores')
    plt.xticks(range(len(chars)), chars, rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f'{language_name}_scores.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def run_analysis_for_language(
    language_code: str,
    language_name: str,
    test_folder: str,
    output_lines: List[str],
    threshold_factor: float = 1.0
) -> None:
    """
    Run adjacency-based (Sukhotin's) classification algorithm for a given language 
    and append the classification results to output_lines.

    Parameters
    ----------
    language_code : str
        The code for the language (e.g., "eng").
    language_name : str
        The full descriptive name of the language (e.g., "English").
    test_folder : str
        Path to the folder containing language data files.
    output_lines : List[str]
        List to accumulate the output/logging lines.
    threshold_factor : float
        Factor for adjusting the vowel threshold determination.
    """
    output_lines.append(f"\nProcessing language: {language_name} ({language_code})")
    text = get_language_data(language_code, test_folder)
    if not text:
        output_lines.append(f"Error: No text data available for {language_name}")
        return

    start_time = time.perf_counter()
    processed_text = preprocess_text(text)
    matrix, char_to_index = create_frequency_matrix(processed_text)
    sums = matrix.sum(axis=1)

    index_to_char = {idx: ch for ch, idx in char_to_index.items()}
    vowels, consonants = classify_vowels(sums, matrix, index_to_char, threshold_factor)

    # Prepare adjusted sums for visualization
    adjusted_sums = np.zeros_like(sums, dtype=float)
    remaining_sums = sums.astype(float)

    for _ in range(len(sums)):
        max_idx = np.argmax(remaining_sums)
        adjusted_sums[max_idx] = remaining_sums[max_idx]
        remaining_sums -= matrix[:, max_idx] * 2
        remaining_sums[max_idx] = -np.inf

    if vowels:
        vowel_indices = [idx for idx, c in index_to_char.items() if c in vowels]
        min_vowel_sum = adjusted_sums[vowel_indices].min()
        threshold = min_vowel_sum - threshold_factor * abs(min_vowel_sum)
    else:
        threshold = adjusted_sums.min()

    visualize_classification_scores(
        adjusted_sums, index_to_char, vowels, threshold, language_name, "visualizations"
    )

    # Filter out whitespace-only chars and sort
    vowels = sorted([v for v in vowels if v.strip()])
    consonants = sorted([c for c in consonants if c.strip()])

    output_lines.append(f"Vowels in {language_name}: {', '.join(vowels)}")
    output_lines.append(f"Consonants in {language_name}: {', '.join(consonants)}")

    end_time = time.perf_counter()
    output_lines.append(f"Execution time: {end_time - start_time:.4f} seconds")
    output_lines.append(f"Visualization saved as: visualizations/{language_name}_scores.png")


def main() -> None:
    """
    Main function to read language codes and names, process each language, 
    and write the results to a file.
    """
    lang_code_df = pd.read_csv('lang_code.csv')
    lang_code_df.columns = lang_code_df.columns.str.strip()
    lang_code_df['code'] = lang_code_df['code'].astype(str).str.strip()
    lang_code_df['language'] = lang_code_df['language'].astype(str).str.strip()

    # Drop rows with missing code or language
    lang_code_df.dropna(subset=['code', 'language'], inplace=True)

    test_folder = 'Test/data'
    files_in_test_folder = [
        f for f in os.listdir(test_folder)
        if os.path.isfile(os.path.join(test_folder, f))
    ]
    # Keep only rows whose code is in the test folder
    lang_code_df = lang_code_df[lang_code_df['code'].isin(files_in_test_folder)]

    output_filename = 'suxotin_algorithm_output.txt'
    with open(output_filename, 'w', encoding='utf-8') as output_file:
        output_lines = []
        for _, row in lang_code_df.iterrows():
            run_analysis_for_language(
                language_code=row['code'],
                language_name=row['language'],
                test_folder=test_folder,
                output_lines=output_lines,
                threshold_factor=1.0
            )
        output_file.write('\n'.join(output_lines))

    print(f"Processing complete. Results saved to {output_filename}")


if __name__ == '__main__':
    main()
