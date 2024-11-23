import time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix

def get_language_data(language_code: str, test_folder: str) -> str:
    """
    Get text data for the specified language code from the data files.
    """
    # Construct the path to the language text file
    filename = os.path.join(test_folder, f"{language_code}")
    if not os.path.isfile(filename):
        # If the file does not exist, log an error and return an empty string
        print(f"Error: File not found for language code {language_code} at {filename}")
        return ""
    try:
        # Try reading the file with UTF-8 encoding
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()
    except UnicodeDecodeError:
        # If UTF-8 fails, try with 'latin-1' encoding
        with open(filename, 'r', encoding='latin-1') as f:
            text = f.read()
    except Exception as e:
        # Catch any other exceptions during file reading
        print(f"Error reading file {filename}: {e}")
        return ""
    return text

def preprocess_text(text: str) -> str:
    """
    Convert the input text to lowercase and remove all non-alphabetic characters except spaces.
    """
    # Use a generator expression to filter out unwanted characters
    return ''.join(
        char.lower() if char.isalpha() or char.isspace() else ' '
        for char in text
    )

def create_frequency_matrix(text: str) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Create a symmetric frequency matrix of adjacent characters in the text.
    """
    # Get sorted list of unique characters for consistent indexing
    unique_chars = sorted(set(text))
    # Map each character to a unique index
    char_to_index = {char: idx for idx, char in enumerate(unique_chars)}
    size = len(unique_chars)
    # Convert the text into indices based on the mapping
    text_indices = [char_to_index[char] for char in text]
    # Get pairs of adjacent indices
    left_indices = text_indices[:-1]
    right_indices = text_indices[1:]
    # Create a mask to exclude pairs where the same character repeats
    mask = np.array(left_indices) != np.array(right_indices)
    # Apply the mask to filter out self-adjacency
    left_indices = np.array(left_indices)[mask]
    right_indices = np.array(right_indices)[mask]
    data = np.ones(len(left_indices), dtype=int)  # Initialize data with ones
    # Create a sparse matrix for adjacency counts
    adjacency_matrix = coo_matrix((data, (left_indices, right_indices)), shape=(size, size))
    # Make the matrix symmetric by adding its transpose
    adjacency_matrix = adjacency_matrix + adjacency_matrix.transpose()
    # Convert the sparse matrix to a dense NumPy array
    matrix = adjacency_matrix.toarray()
    return matrix, char_to_index

def classify_vowels(
    sums: np.ndarray,
    matrix: np.ndarray,
    index_to_char: Dict[int, str],
    threshold_factor: float = 1.0
) -> Tuple[List[str], List[str]]:
    """
    Classify characters as vowels or consonants based on adjacency counts using Sukhotin's algorithm.
    """
    num_chars = len(sums)
    # Copy the sums to remaining_sums for adjustment during iteration
    remaining_sums = sums.astype(np.float64)
    # Initialize an array to store adjusted sums for each character
    adjusted_sums = np.zeros(num_chars, dtype=np.float64)
    vowels = []       # List to hold identified vowels
    consonants = []   # List to hold identified consonants

    # Iteratively classify characters
    for _ in range(num_chars):
        # Find the index of the character with the maximum remaining sum
        max_idx = np.argmax(remaining_sums)
        # Get the character corresponding to this index
        char = index_to_char[max_idx]
        # Store the adjusted sum for later threshold calculation
        adjusted_sums[max_idx] = remaining_sums[max_idx]
        if remaining_sums[max_idx] > 0:
            # If the sum is positive, classify as vowel
            vowels.append(char)
        else:
            # If the sum is negative or zero, classify as consonant
            consonants.append(char)
        # Subtract twice the adjacency counts of this character from all remaining sums
        adjustment = matrix[:, max_idx] * 2
        remaining_sums -= adjustment
        # Set the current character's sum to negative infinity to exclude it from further consideration
        remaining_sums[max_idx] = -np.inf

    # Calculate the threshold for reclassification
    if vowels:
        # Get indices of identified vowels
        vowel_indices = [idx for idx, c in index_to_char.items() if c in vowels]
        # Find the minimum adjusted sum among vowels
        min_vowel_sum = adjusted_sums[vowel_indices].min()
        # Calculate the threshold using the threshold factor
        threshold = min_vowel_sum - threshold_factor * abs(min_vowel_sum)
    else:
        # If no vowels identified, set threshold to minimum adjusted sum
        threshold = adjusted_sums.min()

    # Reclassify consonants that have adjusted sums above the threshold
    reclassified_vowels = []
    new_consonants = []
    for char in consonants:
        idx = next(idx for idx, c in index_to_char.items() if c == char)
        if adjusted_sums[idx] >= threshold:
            reclassified_vowels.append(char)
        else:
            new_consonants.append(char)

    # Update the vowels and consonants lists with reclassified characters
    vowels.extend(reclassified_vowels)
    consonants = new_consonants
    return vowels, consonants

def suxotins_algorithm(
    text: str,
    preprocess: bool = True,
    threshold_factor: float = 1.0
) -> Tuple[List[str], List[str]]:
    """
    Apply Sukhotin's algorithm to classify characters into vowels and consonants.
    """
    if preprocess:
        # Preprocess the text to remove non-alphabetic characters and convert to lowercase
        text = preprocess_text(text)
    # Create the frequency (adjacency) matrix and character-to-index mapping
    matrix, char_to_index = create_frequency_matrix(text)
    # Calculate the sum of adjacency counts for each character
    sums = matrix.sum(axis=1)
    # Create a reverse mapping from indices to characters
    index_to_char = {idx: char for char, idx in char_to_index.items()}
    # Classify vowels and consonants using the adjacency data
    vowels, consonants = classify_vowels(sums, matrix, index_to_char, threshold_factor)
    return vowels, consonants

def visualize_classification_scores(
    adjusted_sums: np.ndarray,
    index_to_char: Dict[int, str],
    vowels: List[str],
    threshold: float,
    language_name: str,
    output_folder: str
):
    """
    Visualize the classification scores and threshold for each character.
    """
    # Get the list of characters in order
    chars = [index_to_char[idx] for idx in range(len(adjusted_sums))]
    scores = adjusted_sums  # Adjusted sums are the scores used for classification
    plt.figure(figsize=(15, 8))
    # Create a bar chart of the scores
    bars = plt.bar(range(len(scores)), scores, color='blue')

    # Color the bars based on classification
    for idx, char in enumerate(chars):
        if char in vowels:
            bars[idx].set_color('red')  # Vowels in red

    # Add a horizontal line representing the threshold
    plt.axhline(y=threshold, color='green', linestyle='--', label=f'Threshold = {threshold:.2f}')

    # Customize the plot with titles and labels
    plt.title(f'Character Classification Scores for {language_name}\nRed: Vowels, Blue: Consonants')
    plt.xlabel('Characters')
    plt.ylabel('Adjusted Sum Scores')
    plt.xticks(range(len(chars)), chars, rotation=45)  # Set x-axis labels as characters
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    # Save the plot to the specified output folder
    output_path = os.path.join(output_folder, f'{language_name}_scores.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def run_analysis_for_language(
    language_code: str,
    language_name: str,
    test_folder: str,
    output_lines: List[str],
    threshold_factor: float = 1.0
):
    """
    Run Sukhotin's algorithm for a given language and append results to output_lines.
    """
    output_lines.append(f"\nProcessing language: {language_name} ({language_code})")
    # Retrieve the text data for the language
    text = get_language_data(language_code, test_folder)
    if not text:
        # If no text is returned, log an error and return
        output_lines.append(f"Error: No text data available for {language_name}")
        return

    start_time = time.perf_counter()
    # Preprocess the text
    processed_text = preprocess_text(text)
    # Create the frequency matrix and character mappings
    matrix, char_to_index = create_frequency_matrix(processed_text)
    # Calculate the sum of adjacency counts for each character
    sums = matrix.sum(axis=1)
    # Create reverse mapping from indices to characters
    index_to_char = {idx: char for char, idx in char_to_index.items()}
    # Apply Sukhotin's algorithm to classify vowels and consonants
    vowels, consonants = classify_vowels(sums, matrix, index_to_char, threshold_factor)

    # Visualization preparation
    adjusted_sums = np.zeros(len(sums), dtype=np.float64)
    remaining_sums = sums.astype(np.float64)
    for _ in range(len(sums)):
        max_idx = np.argmax(remaining_sums)
        adjusted_sums[max_idx] = remaining_sums[max_idx]
        adjustment = matrix[:, max_idx] * 2
        remaining_sums -= adjustment
        remaining_sums[max_idx] = -np.inf

    # Recalculate threshold for visualization
    if vowels:
        vowel_indices = [idx for idx, c in index_to_char.items() if c in vowels]
        min_vowel_sum = adjusted_sums[vowel_indices].min()
        threshold = min_vowel_sum - threshold_factor * abs(min_vowel_sum)
    else:
        threshold = adjusted_sums.min()

    # Generate and save the visualization
    visualize_classification_scores(
        adjusted_sums, index_to_char, vowels, threshold, language_name, "visualizations"
    )

    # Filter out any whitespace characters and sort the results
    vowels = sorted([v for v in vowels if v.strip()])
    consonants = sorted([c for c in consonants if c.strip()])
    # Append the classification results to output_lines
    output_lines.append(f"Vowels in {language_name}: {', '.join(vowels)}")
    output_lines.append(f"Consonants in {language_name}: {', '.join(consonants)}")
    end_time = time.perf_counter()
    # Record the execution time
    output_lines.append(f"Execution time: {end_time - start_time:.4f} seconds")
    output_lines.append(f"Visualization saved as: visualizations/{language_name}_scores.png")

def main():
    """
    Main function to analyze languages and output results to a file.
    """
    # Read the language codes and names from 'lang_code.csv'
    lang_code_df = pd.read_csv('lang_code.csv')
    # Clean up the DataFrame by stripping whitespace
    lang_code_df.columns = lang_code_df.columns.str.strip()
    lang_code_df['code'] = lang_code_df['code'].astype(str).str.strip()
    lang_code_df['language'] = lang_code_df['language'].astype(str).str.strip()
    # Drop any rows with missing 'code' or 'language' values
    lang_code_df = lang_code_df.dropna(subset=['code', 'language'])

    # Set the path to the folder containing language data files
    test_folder = 'Test/data'
    # Get a list of files in the test folder
    files_in_test_folder = os.listdir(test_folder)
    # Filter out directories, keeping only files
    files_in_test_folder = [f for f in files_in_test_folder if os.path.isfile(os.path.join(test_folder, f))]
    # The available language codes are the filenames themselves
    available_codes = files_in_test_folder
    # Keep only languages for which data files are available
    lang_code_df = lang_code_df[lang_code_df['code'].isin(available_codes)]

    # Open the output file for writing results
    output_filename = 'suxotin_algorithm_output.txt'
    with open(output_filename, 'w', encoding='utf-8') as output_file:
        output_lines = []
        # Process each language in the DataFrame
        for idx, row in lang_code_df.iterrows():
            language_code = row['code']
            language_name = row['language']
            # Run analysis for the language and collect results
            run_analysis_for_language(language_code, language_name, test_folder, output_lines)
        # Write all collected output lines to the file
        output_file.write('\n'.join(output_lines))

    print(f"Processing complete. Results saved to {output_filename}")

if __name__ == '__main__':
    main()