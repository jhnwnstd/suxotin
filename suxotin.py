import numpy as np
from pathlib import Path
import nltk
from nltk.corpus import brown

nltk.download('brown')

def preprocess_text(text)->str:
    """
    Convert the input text to lowercase and remove all characters except alphabets and spaces.

    Args:
    text (str): The text to preprocess.

    Returns:
    str: The preprocessed text with only lowercase letters and spaces.
    """
    # Iterate through text, convert to lower case if alphabetic or space, otherwise remove
    return ''.join(char.lower() if char.isalpha() or char.isspace() else '' for char in text)

def create_frequency_matrix(text)->tuple:
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

def sum_rows(matrix)->np.array:
    """
    Compute the sum of each row in the matrix to get total adjacency counts for each character.

    Args:
    matrix (np.array): The frequency matrix of character adjacencies.

    Returns:
    np.array: A NumPy array containing the sums of adjacencies for each character.
    """
    # Sum across rows to get total adjacency counts for each character
    return matrix.sum(axis=1)

def classify_vowels(sums, matrix, char_to_index)->tuple:
    """
    Classify characters as vowels based on their adjacency counts using the provided matrix and sums,
    and classify remaining alphabetic characters as consonants, ensuring vowels and consonants are mutually exclusive.

    Args:
    sums (np.array): Array of sums for each character's adjacencies.
    matrix (np.array): The frequency matrix of character adjacencies.
    char_to_index (dict): Mapping of characters to their respective indices in the matrix.

    Returns:
    tuple: A tuple containing sets of characters classified as vowels and consonants.
    """
    classified_vowels = set()
    classified_consonants = set()
    index_to_char = {idx: char for char, idx in char_to_index.items()}
    used_indices = set()  # Keep track of indices classified as vowels

    # Continue classifying vowels while there are positive sums
    while np.any(sums > 0):
        vowel_idx = np.argmax(sums)
        if sums[vowel_idx] > 0:  # Ensure we are processing a positive sum
            classified_vowels.add(index_to_char[vowel_idx])
            used_indices.add(vowel_idx)
            # Adjust sums based on the newly classified vowel
            sums -= 2 * matrix[:, vowel_idx]
            sums[vowel_idx] = 0

    # Classify remaining alphabetic characters as consonants
    for idx, sum_value in enumerate(sums):
        char = index_to_char[idx]
        if char.isalpha() and idx not in used_indices:
            classified_consonants.add(char)

    return classified_vowels, classified_consonants

def suxotins_algorithm(text, preprocess=True)->set:
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

def process_brown_corpus(preprocess):
    """
    Process text from the Brown Corpus using Suxotin's algorithm.

    Args:
    preprocess (bool): Whether to preprocess the text.

    Returns:
    tuple: Sets of classified vowels and consonants.
    """
    # Concatenate all items in the Brown Corpus into a single string
    text = ' '.join(brown.words())
    return suxotins_algorithm(text, preprocess)

def main():
    """
    Main execution function to process a text file or Brown Corpus and classify characters using Suxotin's algorithm.
    Filters and prints sorted, printable vowels and consonants, excluding spaces and newlines.
    Enhances user input flexibility and organizes output for better readability.
    """
    # Enhanced input handling: choose data source
    data_source = input("Choose the data source - 'local' for local file, 'nltk' for Brown Corpus: ").lower()
    
    # Enhanced input handling: accept 'y', 'ye', or 'yes' for preprocessing confirmation
    preprocess_input = input("Do you want to preprocess the text? (yes/no): ").lower()
    preprocess = preprocess_input in ['y', 'ye', 'yes']
    
    try:
        if data_source == 'local':
            file_path = Path('sherlock_holmes.txt')
            with file_path.open('r', encoding='utf-8') as file:
                text = file.read()
        elif data_source == 'nltk':
            text = ' '.join(brown.words())
        else:
            raise ValueError("Invalid data source selected.")

        vowels, consonants = suxotins_algorithm(text, preprocess)

        # Using list comprehension to filter out non-printable characters
        printable_vowels = sorted([v for v in vowels if v not in ' \n\t'])
        printable_consonants = sorted([c for c in consonants if c not in ' \n\t'])

        # Improved output formatting for clarity and aesthetics
        print("\nClassified Characters:")
        print("Vowels:     ", ', '.join(printable_vowels))
        print("Consonants: ", ', '.join(printable_consonants))
            
    except FileNotFoundError:
        print("File not found. Ensure the file is in the correct directory.")
    except ValueError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    main()