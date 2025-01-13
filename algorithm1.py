import numpy as np
import pandas as pd
import os

def vowel_consonant_classification(V, letters, most_freq_letter):
    """
    Classify letters into vowels and consonants based on the signs of the components
    of the second right singular vector (V1) from the Singular Value Decomposition (SVD).

    Parameters
    ----------
    V : np.ndarray
        The matrix of right singular vectors (num_letters x num_letters).
    letters : list of str
        List of letters in the alphabet.
    most_freq_letter : str
        The most frequent letter in the corpus.

    Returns
    -------
    vowels : list of str
        List of letters classified as vowels.
    consonants : list of str
        List of letters classified as consonants.

    References
    ----------
    - Golub, G. H., & Kahan, W. (1965). Calculating the singular values and pseudo-inverse 
      of a matrix. Journal of the Society for Industrial and Applied Mathematics, 
      Series B: Numerical Analysis, 2(2), 205-224.
    """
    # The second right singular vector
    V1 = V[:, 1]

    # Cluster letters by sign of V1
    cluster_pos = [letter for idx, letter in enumerate(letters) if V1[idx] > 0]
    cluster_neg = [letter for idx, letter in enumerate(letters) if V1[idx] < 0]

    # Determine which cluster contains the most frequent letter (assuming it is vowel)
    if most_freq_letter in cluster_pos:
        vowels, consonants = cluster_pos, cluster_neg
    else:
        vowels, consonants = cluster_neg, cluster_pos

    return vowels, consonants

def algorithm1(corpus, max_words):
    """
    Classify letters in the corpus into vowels and consonants using a spectral decomposition.

    Parameters
    ----------
    corpus : list of str
        List of words (strings).
    max_words : int or None
        Maximum number of words to process. If None, process all.

    Returns
    -------
    vowels : list of str
        List of letters classified as vowels.
    consonants : list of str
        List of letters classified as consonants.

    References
    ----------
    - Deerwester, S., Dumais, S. T., Furnas, G. W., Landauer, T. K., & Harshman, R. (1990).
      Indexing by latent semantic analysis. Journal of the American society for 
      information science, 41(6), 391-407.
    - Manning, C. D., & Schütze, H. (1999). Foundations of Statistical Natural Language
      Processing. MIT press.
    """

    # Limit the corpus size
    corpus = corpus if max_words is None else corpus[:max_words]

    # Collect sorted unique letters from all words
    letters = sorted(set(''.join(corpus)))
    num_letters = len(letters)

    # Map letters to indices
    letter_to_index = {letter: idx for idx, letter in enumerate(letters)}

    # Array to count occurrences of each letter
    letters_count = np.zeros(num_letters, dtype=int)

    # Dictionary to track unique p-frames
    p_frame_indices = {}

    # Collect matrix entries for building A
    A_entries = []

    for word in corpus:
        # Pad with spaces for boundary
        padded_word = f" {word} "
        # Capture letter in the middle of a triple (p_frame)
        for i in range(1, len(padded_word) - 1):
            p_frame = (padded_word[i - 1], '*', padded_word[i + 1])
            # Assign an index if new
            if p_frame not in p_frame_indices:
                p_frame_indices[p_frame] = len(p_frame_indices)

            letter = padded_word[i]
            letter_idx = letter_to_index.get(letter, None)
            if letter_idx is not None:
                letters_count[letter_idx] += 1
                A_entries.append((p_frame_indices[p_frame], letter_idx))

    # Check if there are any valid entries
    if not A_entries:
        return [], []

    num_p_frames = len(p_frame_indices)
    # Build binary matrix A
    A = np.zeros((num_p_frames, num_letters), dtype=float)
    row_indices, col_indices = zip(*A_entries)
    A[row_indices, col_indices] = 1

    # Perform SVD: A = U * s * Vt
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T  # Right singular vectors

    # Find most frequent letter
    most_freq_letter_idx = np.argmax(letters_count)
    most_freq_letter = letters[most_freq_letter_idx]

    # Classify into vowels/consonants
    vowels, consonants = vowel_consonant_classification(V, letters, most_freq_letter)

    return vowels, consonants

def preprocess_text(text):
    """
    Preprocess the text by lowercasing and removing non-alphabetic characters.

    Parameters
    ----------
    text : str
        The original text string.

    Returns
    -------
    corpus : list of str
        List of cleaned words.

    References
    ----------
    - Bird, S., Klein, E., & Loper, E. (2009). Natural Language Processing with Python: 
      Analyzing Text with the Natural Language Toolkit. O'Reilly Media, Inc.
    """
    # Lowercase and split
    words = text.lower().split()
    # Filter + remove non-alphabetic chars
    corpus = [
        ''.join(char for char in word if char.isalpha())
        for word in words
        if any(char.isalpha() for char in word)
    ]
    return corpus

def run_algorithm_for_language(language_code, language_name, max_words, test_folder, output_file):
    """
    Run the algorithm for a specific language and write the results to the output file.

    Parameters
    ----------
    language_code : str
        The code of the language (e.g., 'eng' for English).
    language_name : str
        The full name of the language.
    max_words : int or None
        Maximum number of words to process.
    test_folder : str
        The folder where the language text files are located.
    output_file : file-like object
        The file object where results will be written.
    """

    output_lines = [f"\nProcessing language: {language_name} ({language_code})"]

    filename = os.path.join(test_folder, f"{language_code}")
    if not os.path.isfile(filename):
        output_lines.append(f"Error: File not found for language {language_name} at {filename}")
        output_file.write('\n'.join(output_lines))
        return

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()
    except UnicodeDecodeError:
        # Fallback to 'latin-1'
        with open(filename, 'r', encoding='latin-1') as f:
            text = f.read()
    except Exception as e:
        output_lines.append(f"Error reading file {filename}: {e}")
        output_file.write('\n'.join(output_lines))
        return

    # Preprocess text
    corpus = preprocess_text(text)
    if not corpus:
        output_lines.append(f"No valid words found in {language_name}. Skipping this language.")
        output_file.write('\n'.join(output_lines))
        return

    # Run classification
    vowels, consonants = algorithm1(corpus, max_words)
    if not vowels and not consonants:
        output_lines.append(f"No vowels or consonants identified in {language_name}.")
        output_file.write('\n'.join(output_lines))
        return

    # Prepare results
    vowels = sorted(set(vowels))
    consonants = sorted(set(consonants))
    output_lines.append(f"Vowels in {language_name}: {', '.join(vowels)}")
    output_lines.append(f"Consonants in {language_name}: {', '.join(consonants)}")

    # Write results
    output_file.write('\n'.join(output_lines))
    output_file.write('\n')

# Main driver code
if __name__ == "__main__":
    # Read language codes
    lang_code_df = pd.read_csv('lang_code.csv')  # Make sure 'lang_code.csv' is in the same directory
    
    # Clean up column names and data
    lang_code_df.columns = lang_code_df.columns.str.strip()
    lang_code_df['code'] = lang_code_df['code'].astype(str).str.strip()
    lang_code_df['language'] = lang_code_df['language'].astype(str).str.strip()

    # Drop rows with missing codes or language names
    lang_code_df.dropna(subset=['code', 'language'], inplace=True)

    # Path to the folder with text files
    test_folder = 'Test/data'
    # Filter out directories in the test folder
    files_in_test_folder = [
        f for f in os.listdir(test_folder)
        if os.path.isfile(os.path.join(test_folder, f))
    ]

    # Keep only rows whose language code has a matching file
    lang_code_df = lang_code_df[lang_code_df['code'].isin(files_in_test_folder)]

    # Maximum words to process per language (None = no limit)
    max_words = None

    # Output file
    output_filename = 'algorithm1_output.txt'
    with open(output_filename, 'w', encoding='utf-8') as output_file:
        for _, row in lang_code_df.iterrows():
            run_algorithm_for_language(
                language_code=row['code'],
                language_name=row['language'],
                max_words=max_words,
                test_folder=test_folder,
                output_file=output_file
            )

    print(f"Processing complete. Results saved to {output_filename}")
