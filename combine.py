import numpy as np
import pandas as pd
import os
import time
from typing import List, Tuple, Dict

# Existing functions: vowel_consonant_classification, algorithm1, preprocess_text, suxotins_algorithm
def vowel_consonant_classification(V, letters, most_freq_letter):
    """
    Classify letters into vowels and consonants based on the signs of the components
    of the second right singular vector from the Singular Value Decomposition (SVD).

    Parameters:
    - V: The matrix of right singular vectors (num_letters x num_letters).
    - letters: List of letters in the alphabet.
    - most_freq_letter: The most frequent letter in the corpus.

    Returns:
    - vowels: List of letters classified as vowels.
    - consonants: List of letters classified as consonants.
    """

    # Get the second right singular vector (V1) from matrix V
    # This vector captures the second most significant pattern in the data
    V1 = V[:, 1]

    # Split letters into two clusters based on the sign of the components in V1
    # Letters with positive components are in cluster1
    cluster1 = [letter for i, letter in enumerate(letters) if V1[i] > 0]
    # Letters with negative components are in cluster2
    cluster2 = [letter for i, letter in enumerate(letters) if V1[i] < 0]

    # Determine which cluster contains the most frequent letter
    # We assume the most frequent letter is a vowel (common in many languages)
    if most_freq_letter in cluster1:
        vowels = cluster1
        consonants = cluster2
    else:
        vowels = cluster2
        consonants = cluster1

    # Return the lists of vowels and consonants
    return vowels, consonants

def algorithm1(corpus, max_words):
    """
    Classify letters in the corpus into vowels and consonants using spectral decomposition.

    Parameters:
    - corpus: List of words (strings).
    - max_words: Maximum number of words to process.

    Returns:
    - vowels: List of letters classified as vowels.
    - consonants: List of letters classified as consonants.
    """

    # Limit the number of words processed to max_words
    corpus = corpus[:max_words]

    # Collect all unique letters present in the corpus
    # Join all words into a single string and convert to a set to get unique letters
    letters = sorted(set(''.join(corpus)))
    num_letters = len(letters)

    # Create a mapping from letters to their indices (positions in the letters list)
    letter_to_index = {letter: idx for idx, letter in enumerate(letters)}

    # Initialize an array to count the occurrences of each letter in the corpus
    letters_count = np.zeros(num_letters, dtype=int)

    # Initialize a dictionary to map p-frames to indices
    p_frame_indices = {}

    # Initialize a list to store entries for the matrix A
    A_entries = []

    # Iterate over each word in the corpus
    for word in corpus:
        # Pad the word with one space at the beginning and end to handle boundary conditions
        padded_word = ' ' + word + ' '

        # Iterate over each character in the padded word, excluding the first and last character (spaces)
        for i in range(1, len(padded_word) - 1):
            # Create a p-frame by taking the immediate preceding and succeeding letters with '*' in the middle
            p_frame = (padded_word[i - 1], '*', padded_word[i + 1])

            # Map the p-frame to a unique index
            if p_frame not in p_frame_indices:
                # Assign a new index if the p-frame is encountered for the first time
                p_frame_indices[p_frame] = len(p_frame_indices)
            p_frame_idx = p_frame_indices[p_frame]

            # Get the current letter (the one in the middle of the p-frame)
            letter = padded_word[i]
            # Get the index of the letter from the letter_to_index mapping
            letter_idx = letter_to_index.get(letter)
            if letter_idx is None:
                # If the character is not a letter (e.g., space), skip it
                continue

            # Increment the count for this letter
            letters_count[letter_idx] += 1

            # Record the (p-frame index, letter index) pair for constructing matrix A
            A_entries.append((p_frame_idx, letter_idx))

    # Number of unique p-frames encountered
    num_p_frames = len(p_frame_indices)

    # If there are no entries in A_entries, return empty lists
    if not A_entries:
        return [], []

    # Build the binary matrix A of shape (num_p_frames x num_letters)
    # Each row corresponds to a p-frame, and each column corresponds to a letter
    # A[i, j] = 1 if letter j occurs in p-frame i
    # Initialize the matrix with zeros
    A = np.zeros((num_p_frames, num_letters), dtype=float)

    # Unzip the A_entries list into separate lists of row indices and column indices
    row_indices, col_indices = zip(*A_entries)
    data = np.ones(len(A_entries), dtype=int)  # All entries are 1 (binary occurrence)

    # Use advanced indexing to assign the data to the appropriate positions in matrix A
    A[row_indices, col_indices] = data

    # Perform Singular Value Decomposition on matrix A
    # Decompose A into U, s, and Vt such that A = U * s * Vt
    U, s, Vt = np.linalg.svd(A, full_matrices=False)

    # Transpose Vt to get V (matrix of right singular vectors)
    V = Vt.T  # Shape of V is (num_letters x num_letters)

    # Find the index of the most frequent letter in the corpus
    most_freq_letter_idx = np.argmax(letters_count)
    # Get the most frequent letter using the index
    most_freq_letter = letters[most_freq_letter_idx]

    # Use the vowel_consonant_classification function to classify letters
    vowels, consonants = vowel_consonant_classification(V, letters, most_freq_letter)

    # Return the lists of vowels and consonants
    return vowels, consonants

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


def combined_classification(corpus, max_words, text, letters_count):
    """
    Combine the classifications from both Sukhotin's algorithm and the spectral decomposition approach.

    Parameters:
    - corpus: List of words from the preprocessed text.
    - max_words: Maximum number of words to process.
    - text: The original preprocessed text as a string.
    - letters_count: A dictionary with letters as keys and their counts as values.

    Returns:
    - vowels_combined: List of letters classified as vowels.
    - consonants_combined: List of letters classified as consonants.
    """
    # Run spectral decomposition algorithm (Algorithm 1)
    vowels_spectral, consonants_spectral = algorithm1(corpus, max_words)
    
    # Run Sukhotin's algorithm
    vowels_sukhotin, consonants_sukhotin = suxotins_algorithm(text, preprocess=False)
    
    # Combine the results
    vowels_combined = []
    consonants_combined = []
    all_letters = set(vowels_spectral + consonants_spectral + vowels_sukhotin + consonants_sukhotin)
    
    for letter in all_letters:
        in_vowels_spectral = letter in vowels_spectral
        in_vowels_sukhotin = letter in vowels_sukhotin
        
        if in_vowels_spectral and in_vowels_sukhotin:
            # Both algorithms agree the letter is a vowel
            vowels_combined.append(letter)
        elif not in_vowels_spectral and not in_vowels_sukhotin:
            # Both algorithms agree the letter is a consonant
            consonants_combined.append(letter)
        else:
            # Tie-breaker: Use frequency (you can adjust the threshold or tie-breaker strategy)
            frequency = letters_count.get(letter, 0)
            if frequency >= np.median(list(letters_count.values())):
                # If frequency is above the median, classify as vowel
                vowels_combined.append(letter)
            else:
                consonants_combined.append(letter)
    
    return sorted(vowels_combined), sorted(consonants_combined)

def run_combined_algorithm_for_language(language_code, language_name, max_words, test_folder, output_file):
    """
    Run the combined algorithm for a specific language and write the results to the output file.

    Parameters:
    - language_code: The code of the language (e.g., 'eng' for English).
    - language_name: The full name of the language.
    - max_words: Maximum number of words to process.
    - test_folder: The folder where the language text files are located.
    - output_file: The file object where results will be written.
    """
    output_lines = []
    output_lines.append(f"\nProcessing language: {language_name} ({language_code})")

    # Construct the filename for the language text file
    filename = os.path.join(test_folder, f"{language_code}")
    if not os.path.isfile(filename):
        output_lines.append(f"Error: File not found for language {language_name} at {filename}")
        output_file.write('\n'.join(output_lines))
        return

    # Read the text data from the file
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(filename, 'r', encoding='latin-1') as f:
            text = f.read()
    except Exception as e:
        output_lines.append(f"Error reading file {filename}: {e}")
        output_file.write('\n'.join(output_lines))
        return

    # Preprocess the text to clean and tokenize it
    corpus = preprocess_text(text)
    if not corpus:
        output_lines.append(f"No valid words found in {language_name}. Skipping this language.")
        output_file.write('\n'.join(output_lines))
        return

    # Prepare data for combined classification
    words_list = corpus.split()
    letters = ''.join(words_list)
    letters_count = {char: letters.count(char) for char in set(letters)}

    # Run the combined classification algorithm
    vowels, consonants = combined_classification(words_list, max_words, corpus, letters_count)

    if not vowels and not consonants:
        output_lines.append(f"No vowels or consonants identified in {language_name}.")
        output_file.write('\n'.join(output_lines))
        return

    # Write the results
    output_lines.append(f"Vowels in {language_name}: {', '.join(vowels)}")
    output_lines.append(f"Consonants in {language_name}: {', '.join(consonants)}")

    output_file.write('\n'.join(output_lines))
    output_file.write('\n')  # Add an extra newline for readability

# Update the main execution to use the combined algorithm
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

# Get list of files in the Test folder (filenames without extension)
files_in_test_folder = os.listdir(test_folder)
# Ensure we only get files (not directories)
files_in_test_folder = [f for f in files_in_test_folder if os.path.isfile(os.path.join(test_folder, f))]

# The available language codes are the filenames themselves
available_codes = files_in_test_folder

# Keep only the rows where 'code' is in the list of available codes
lang_code_df = lang_code_df[lang_code_df['code'].isin(available_codes)]

# Set the maximum number of words to process for each language
max_words = None  # Adjust this value as needed

# Open the output file for writing results
output_filename = 'combined_algorithm_output.txt'
with open(output_filename, 'w', encoding='utf-8') as output_file:
    # Run the combined algorithm for each language in the DataFrame
    for idx, row in lang_code_df.iterrows():
        language_code = row['code']
        language_name = row['language']
        # Call the function to process the language and write results
        run_combined_algorithm_for_language(language_code, language_name, max_words, test_folder, output_file)

print(f"Processing complete. Results saved to {output_filename}")