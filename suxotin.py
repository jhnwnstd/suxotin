import time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import os

def get_language_data(language_code: str, test_folder: str) -> str:
    """
    Get text data for the specified language code from the data files.

    Parameters:
    - language_code: The code of the language (e.g., 'eng' for English).
    - test_folder: The folder where the language text files are located.

    Returns:
    - text: The content of the language text file as a string.
    """
    # Construct the filename for the language text file
    filename = os.path.join(test_folder, f"{language_code}")
    if not os.path.isfile(filename):
        # If the file does not exist, log an error and return an empty string
        print(f"Error: File not found for language code {language_code} at {filename}")
        return ""
    # Read the text data from the file
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()
    except UnicodeDecodeError:
        # If there's an encoding issue with utf-8, try 'latin-1' encoding
        with open(filename, 'r', encoding='latin-1') as f:
            text = f.read()
    except Exception as e:
        # Log any other exceptions that occur during file reading
        print(f"Error reading file {filename}: {e}")
        return ""
    return text

def preprocess_text(text: str) -> str:
    """
    Convert the input text to lowercase and remove all non-alphabetic characters except spaces.

    Parameters:
    - text: The original text string.

    Returns:
    - A cleaned text string containing only lowercase alphabetic characters and spaces.
    """
    # Use a generator expression to filter text
    # Keep only alphabetic characters and spaces; convert to lowercase
    return ''.join(
        char.lower() if char.isalpha() or char.isspace() else ''
        for char in text
    )

def create_frequency_matrix(text: str) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Create a symmetric frequency matrix of adjacent characters in the text.

    Parameters:
    - text: The preprocessed text string.

    Returns:
    - matrix: A symmetric adjacency matrix where matrix[i, j] counts the number of times
      character i appears next to character j in the text.
    - char_to_index: A dictionary mapping characters to their indices in the matrix.
    """
    # Get a sorted list of unique characters in the text for consistent ordering
    unique_chars = sorted(set(text))
    # Create a mapping from each character to a unique index
    char_to_index = {char: idx for idx, char in enumerate(unique_chars)}
    size = len(unique_chars)
    # Convert the text into a NumPy array of indices based on the char_to_index mapping
    text_indices = np.array([char_to_index[char] for char in text], dtype=np.int32)
    # Get pairs of adjacent indices
    left_indices = text_indices[:-1]
    right_indices = text_indices[1:]
    # Create a mask to exclude self-adjacency (e.g., repeated characters like 'aa')
    mask = left_indices != right_indices
    # Filter out self-adjacency pairs based on the mask
    left_indices = left_indices[mask]
    right_indices = right_indices[mask]
    # Initialize the frequency matrix with zeros
    matrix = np.zeros((size, size), dtype=np.int32)
    # Increment counts for each observed pair to create the adjacency matrix
    np.add.at(matrix, (left_indices, right_indices), 1)
    np.add.at(matrix, (right_indices, left_indices), 1)  # Keep matrix symmetric
    return matrix, char_to_index

def classify_vowels(
    sums: np.ndarray, matrix: np.ndarray, index_to_char: Dict[int, str]
) -> Tuple[List[str], List[str]]:
    """
    Classify characters as vowels or consonants based on adjacency counts.

    Parameters:
    - sums: The sum of adjacency counts for each character.
    - matrix: The adjacency matrix of character pairs.
    - index_to_char: A mapping from indices to characters.

    Returns:
    - vowels: List of characters classified as vowels.
    - consonants: List of characters classified as consonants.
    """
    num_chars = len(sums)
    # Copy sums to be adjusted during the classification
    remaining_sums = sums.astype(np.float64)
    # Initialize an array to store the final adjusted sums for thresholding
    adjusted_sums = np.zeros(num_chars, dtype=np.float64)
    vowels = []      # List to hold classified vowels
    consonants = []  # List to hold classified consonants

    # Classify each character one at a time based on its remaining sum
    for _ in range(num_chars):
        # Find the index of the character with the maximum remaining sum
        max_idx = np.argmax(remaining_sums)
        # Get the character corresponding to the max index
        char = index_to_char[max_idx]
        # Store the adjusted sum for thresholding later
        adjusted_sums[max_idx] = remaining_sums[max_idx]
        # Classify the character based on whether the sum is positive
        if remaining_sums[max_idx] > 0:
            vowels.append(char)
        else:
            consonants.append(char)
        # Apply adjustment by subtracting twice the adjacency counts of the classified character
        adjustment = matrix[:, max_idx] * 2
        remaining_sums -= adjustment
        # Set the current character's sum to -infinity to exclude it from future consideration
        remaining_sums[max_idx] = -np.inf

    # Determine a threshold based on the minimum adjusted sum among vowels
    if vowels:
        vowel_indices = [idx for idx, c in index_to_char.items() if c in vowels]
        min_vowel_sum = adjusted_sums[vowel_indices].min()
    else:
        min_vowel_sum = 0

    # Calculate the threshold to reclassify consonants as vowels if necessary
    threshold = min_vowel_sum - abs(min_vowel_sum) * 1  # Adjust threshold as needed

    # Reclassify consonants with adjusted sums above the threshold as vowels
    reclassified_vowels = []
    new_consonants = []
    for char in consonants:
        # Get the index of the character
        idx = next(idx for idx, c in index_to_char.items() if c == char)
        # Check if the adjusted sum is above the threshold
        if adjusted_sums[idx] >= threshold:
            reclassified_vowels.append(char)
        else:
            new_consonants.append(char)

    # Update the vowels and consonants lists
    vowels.extend(reclassified_vowels)
    consonants = new_consonants
    return vowels, consonants

def suxotins_algorithm(text: str, preprocess: bool = True) -> Tuple[List[str], List[str]]:
    """
    Apply Suxotin's algorithm to classify characters into vowels and consonants.

    Parameters:
    - text: The input text string.
    - preprocess: Whether to preprocess the text.

    Returns:
    - vowels: List of characters classified as vowels.
    - consonants: List of characters classified as consonants.
    """
    # Optionally preprocess the text by removing non-alphabetic characters and converting to lowercase
    if preprocess:
        text = preprocess_text(text)
    # Generate the frequency matrix and character-to-index mapping from the text
    matrix, char_to_index = create_frequency_matrix(text)
    # Calculate the sum of adjacency counts for each character
    sums = matrix.sum(axis=1)
    # Reverse mapping from indices to characters for lookup
    index_to_char = {idx: char for char, idx in char_to_index.items()}
    # Classify characters into vowels and consonants using adjacency data
    vowels, consonants = classify_vowels(sums, matrix, index_to_char)
    return vowels, consonants

def run_analysis_for_language(language_code: str, language_name: str, test_folder: str, output_lines: List[str]):
    """
    Run Suxotin's algorithm for a given language and append results to output_lines.

    Parameters:
    - language_code: The code of the language.
    - language_name: The full name of the language.
    - test_folder: The folder where the language text files are located.
    - output_lines: The list where results will be appended.
    """
    # Add a header for the language
    output_lines.append(f"\nProcessing language: {language_name} ({language_code})")
    # Get the text data for the language
    text = get_language_data(language_code, test_folder)
    if not text:
        # If no text is returned, log an error and return
        output_lines.append(f"Error: No text data available for {language_name}")
        return
    # Record start time for performance measurement
    start_time = time.perf_counter()
    # Run Suxotin's algorithm to classify letters
    vowels, consonants = suxotins_algorithm(text, preprocess=True)
    # Filter out any whitespace and sort the results
    vowels = sorted([v for v in vowels if v.strip()])
    consonants = sorted([c for c in consonants if c.strip()])
    # Append the classification results to output_lines
    output_lines.append(f"Vowels in {language_name}: {', '.join(vowels)}")
    output_lines.append(f"Consonants in {language_name}: {', '.join(consonants)}")
    # Record end time and calculate execution time
    end_time = time.perf_counter()
    output_lines.append(f"Execution time: {end_time - start_time:.4f} seconds")

def main():
    """
    Main function to analyze languages and output results to a file.
    """
    # Read the lang_code.csv to get the list of languages
    lang_code_df = pd.read_csv('lang_code.csv')  # Ensure 'lang_code.csv' is in the same directory
    # Clean up column names and data by stripping whitespace
    lang_code_df.columns = lang_code_df.columns.str.strip()
    lang_code_df['code'] = lang_code_df['code'].astype(str).str.strip()
    lang_code_df['language'] = lang_code_df['language'].astype(str).str.strip()
    # Drop rows with missing 'code' or 'language' values
    lang_code_df = lang_code_df.dropna(subset=['code', 'language'])
    # Set the path to the Test folder containing the language data files
    test_folder = 'Test/data'
    # Get list of files in the Test folder
    files_in_test_folder = os.listdir(test_folder)
    # Ensure we only get files (not directories)
    files_in_test_folder = [f for f in files_in_test_folder if os.path.isfile(os.path.join(test_folder, f))]
    # The available language codes are the filenames themselves
    available_codes = files_in_test_folder
    # Keep only the rows where 'code' is in the list of available codes
    lang_code_df = lang_code_df[lang_code_df['code'].isin(available_codes)]
    # Open the output file for writing results
    output_filename = 'suxotin_algorithm_output.txt'
    with open(output_filename, 'w', encoding='utf-8') as output_file:
        output_lines = []
        # Run the algorithm for each language in the DataFrame
        for idx, row in lang_code_df.iterrows():
            language_code = row['code']
            language_name = row['language']
            # Call the function to process the language and collect results
            run_analysis_for_language(language_code, language_name, test_folder, output_lines)
        # Write all the collected output lines to the file
        output_file.write('\n'.join(output_lines))
    # Print a completion message
    print(f"Processing complete. Results saved to {output_filename}")

# Run the main function when the script is executed
if __name__ == '__main__':
    main()