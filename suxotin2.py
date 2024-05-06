import numpy as np
import string

def preprocess_text(text):
    """
    Convert the input text to lowercase and remove all characters except alphabets and spaces.

    Args:
    text (str): The text to preprocess.

    Returns:
    str: The preprocessed text with only lowercase letters and spaces.
    """
    # Iterate through text, convert to lower case if alphabetic or space, otherwise remove
    return ''.join(char.lower() if char.isalpha() or char.isspace() else '' for char in text)

def create_frequency_matrix(text):
    """
    Create a frequency matrix of adjacent characters in the text based on unique characters.

    Args:
    text (str): The text from which to create the frequency matrix.

    Returns:
    tuple: A tuple containing the matrix as a NumPy array and a dictionary mapping characters to indices.
    """
    # Identify unique characters in the text
    unique_chars = set(text)
    # Create a mapping from characters to indices
    char_to_index = {char: idx for idx, char in enumerate(unique_chars)}
    # Initialize a square matrix of size based on the number of unique characters
    size = len(unique_chars)
    matrix = np.zeros((size, size), dtype=int)

    # Generate pairs of adjacent characters and populate the matrix
    pairs = zip(text, text[1:] + text[:1])
    for left, right in pairs:
        l_idx, r_idx = char_to_index[left], char_to_index[right]
        matrix[l_idx][r_idx] += 1
        matrix[r_idx][l_idx] += 1

    # Ensure the diagonal is zeroed out as self-pairing is not relevant
    np.fill_diagonal(matrix, 0)
    return matrix, char_to_index

def sum_rows(matrix):
    """
    Compute the sum of each row in the matrix to get total adjacency counts for each character.

    Args:
    matrix (np.array): The frequency matrix of character adjacencies.

    Returns:
    np.array: A NumPy array containing the sums of adjacencies for each character.
    """
    # Sum across rows to get total adjacency counts for each character
    return matrix.sum(axis=1)

def classify_vowels(sums, matrix, char_to_index):
    """
    Classify characters as vowels based on their adjacency counts using the provided matrix and sums.

    Args:
    sums (np.array): Array of sums for each character's adjacencies.
    matrix (np.array): The frequency matrix of character adjacencies.
    char_to_index (dict): Mapping of characters to their respective indices in the matrix.

    Returns:
    set: A set of characters classified as vowels.
    """
    classified_vowels = set()
    index_to_char = {idx: char for char, idx in char_to_index.items()}

    # Continue classifying vowels while there are positive sums
    while np.any(sums > 0):
        vowel_idx = np.argmax(sums)
        classified_vowels.add(index_to_char[vowel_idx])
        # Adjust sums based on the newly classified vowel
        sums -= 2 * matrix[:, vowel_idx]
        sums[vowel_idx] = 0
    return classified_vowels

def suxotins_algorithm(text, preprocess=True):
    """
    Apply Suxotin's algorithm to classify vowels in the text based on adjacency frequencies.

    Args:
    text (str): The text to process.
    preprocess (bool): Whether to preprocess the text (default True).

    Returns:
    set: A set of characters classified as vowels.
    """
    if preprocess:
        text = preprocess_text(text)
    matrix, char_to_index = create_frequency_matrix(text)
    sums = sum_rows(matrix)
    return classify_vowels(sums, matrix, char_to_index)

def main():
    """
    Main execution function to process a text file and classify vowels using Suxotin's algorithm.
    Filters and prints only sorted, printable vowels excluding spaces and newlines.
    """
    file_path = 'sherlock_holmes.txt'
    preprocess = input("Do you want to preprocess the text? (yes/no): ").lower() == 'yes'
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            vowels = suxotins_algorithm(text, preprocess)
            # Filter vowels to exclude spaces and newlines
            printable_vowels = sorted(filter(lambda x: x not in ' \n', vowels))
            print("Classified vowels:", printable_vowels)
    except FileNotFoundError:
        print("File not found. Ensure the file is in the correct directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    main()