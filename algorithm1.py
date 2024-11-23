import numpy as np
import pandas as pd
import os

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

    # Get the second right singular vector from matrix V
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

def preprocess_text(text):
    """
    Preprocess the text by lowercasing and removing non-alphabetic characters.
    """
    words = text.lower().split()
    corpus = [
        ''.join(char for char in word if char.isalpha())
        for word in words
        if any(char.isalpha() for char in word)
    ]
    return corpus

def run_algorithm_for_language(language_code, language_name, max_words, test_folder):
    print(f"\nProcessing language: {language_name} ({language_code})")
    # Construct the filename without extension
    filename = os.path.join(test_folder, f"{language_code}")
    if not os.path.isfile(filename):
        print(f"Error: File not found for language {language_name} at {filename}")
        return
    # Read the text data
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()
    except UnicodeDecodeError:
        # Try a different encoding if utf-8 fails
        with open(filename, 'r', encoding='latin-1') as f:
            text = f.read()
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return
    # Preprocess the text
    corpus = preprocess_text(text)
    if not corpus:
        print(f"No valid words found in {language_name}. Skipping this language.")
        return
    # Run Algorithm 1
    vowels, consonants = algorithm1(corpus, max_words)
    # Sort and print the results
    vowels = sorted(set(vowels))
    consonants = sorted(set(consonants))
    print(f"Vowels in {language_name}: {', '.join(vowels)}")
    print(f"Consonants in {language_name}: {', '.join(consonants)}")

# Read the lang_code.csv to get the list of languages
lang_code_df = pd.read_csv('lang_code.csv')

# Clean up column names and data
lang_code_df.columns = lang_code_df.columns.str.strip()
lang_code_df['code'] = lang_code_df['code'].astype(str).str.strip()
lang_code_df['language'] = lang_code_df['language'].astype(str).str.strip()

# Drop rows with missing 'code' or 'language'
lang_code_df = lang_code_df.dropna(subset=['code', 'language'])

# Set the path to the Test folder
test_folder = 'Test'

# Get list of files in the Test folder (filenames without extension)
files_in_test_folder = os.listdir(test_folder)
# Ensure we only get files (not directories)
files_in_test_folder = [f for f in files_in_test_folder if os.path.isfile(os.path.join(test_folder, f))]

# The available language codes are the filenames themselves
available_codes = files_in_test_folder

# Keep only the rows where 'code' is in the list of available codes
lang_code_df = lang_code_df[lang_code_df['code'].isin(available_codes)]

# Set the maximum number of words to process
max_words = 100000  # Adjust this value as needed

# Run the algorithm for each language
for idx, row in lang_code_df.iterrows():
    language_code = row['code']
    language_name = row['language']
    run_algorithm_for_language(language_code, language_name, max_words, test_folder)