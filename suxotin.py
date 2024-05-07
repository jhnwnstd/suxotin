import time
import numpy as np
from pathlib import Path
import nltk
from nltk.corpus import gutenberg

nltk.download('gutenberg')

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
    This matrix is symmetric and counts the adjacencies for each pair of characters in the text.
    The matrix and character index mapping are essential for analyzing character adjacency relationships.

    Args:
    text (str): The text from which to create the frequency matrix.

    Returns:
    tuple: A tuple containing the matrix as a NumPy array and a dictionary mapping characters to indices,
           which serves as the legend for understanding the matrix indices.
    """
    # Extract all unique characters from the text to define the matrix dimension
    unique_chars = set(text)
    # Map each unique character to a unique index using dictionary comprehension
    char_to_index = {char: idx for idx, char in enumerate(unique_chars)}
    # Determine the size of the matrix based on the number of unique characters
    size = len(unique_chars)
    # Initialize a square matrix of zeros with dimensions [size x size]
    matrix = np.zeros((size, size), dtype=int)

    # Loop through the text, character by character, excluding the last character to prevent index error
    for i in range(len(text) - 1):
        # Retrieve indices from the mapping for current and next character
        l_idx, r_idx = char_to_index[text[i]], char_to_index[text[i+1]]
        # Increment the position in the matrix for both (l_idx, r_idx) and (r_idx, l_idx) to maintain symmetry
        matrix[l_idx][r_idx] += 1
        matrix[r_idx][l_idx] += 1  # Ensures the matrix is symmetric

    # Ensure diagonal entries are zero to ignore self-pairings, as a character adjacent to itself is not considered
    np.fill_diagonal(matrix, 0)
    return matrix, char_to_index

def classify_vowels(sums, matrix, char_to_index)->tuple[set, set]:
    """
    Classify characters as vowels or consonants based on their adjacency counts within a given text.
    Characters with higher adjacency counts are initially considered for classification as vowels,
    as they typically appear more frequently next to a variety of other characters. The classification
    process iteratively selects the character with the highest adjacency count, adjusts the counts,
    and then classifies the remaining characters as consonants.

    Args:
    sums (np.array): Array of sums for each character's adjacencies, where each sum is the total count
                     of adjacencies involving the corresponding character.
    matrix (np.array): The frequency matrix of character adjacencies, where the element at [i, j]
                       indicates the adjacency count between characters i and j.
    char_to_index (dict): Mapping of characters to their respective indices in the matrix. This mapping
                          helps identify characters from indices in the matrix.

    Returns:
    tuple: A tuple containing two sets, one of characters classified as vowels and the other as consonants.
    """
    classified_vowels = set()  # Set to store characters classified as vowels
    classified_consonants = set()  # Initially empty set for consonants
    index_to_char = {idx: char for char, idx in char_to_index.items()}  # Reverse mapping from indices to characters
    remaining_sums = sums.copy()  # Copy of the sums array to avoid altering the original during classification

    # Iterate over sums to classify characters with the highest adjacency counts as vowels
    while np.any(remaining_sums > 0):
        vowel_idx = np.argmax(remaining_sums)  # Find index with the highest adjacency count
        if remaining_sums[vowel_idx] > 0:  # Check if the highest count is positive
            char = index_to_char[vowel_idx]
            classified_vowels.add(char)  # Classify character as a vowel
            remaining_sums -= 2 * matrix[:, vowel_idx]  # Reduce sums by counts associated with the newly classified vowel
            remaining_sums[vowel_idx] = 0  # Zero out the sum for the classified vowel to prevent reclassification

    # Use set comprehension to classify the remaining characters as consonants
    classified_consonants = {
        index_to_char[idx] for idx in range(len(sums))
        if index_to_char[idx].isalpha() and index_to_char[idx] not in classified_vowels  # Ensure character is alphabetic and not already classified as a vowel
    }

    return classified_vowels, classified_consonants

def suxotins_algorithm(text: str, preprocess: bool = True) -> tuple[set, set]:
    """
    Apply Suxotin's algorithm to classify characters in a given text into vowels and consonants based on their adjacency frequencies.
    This method involves processing the text through a frequency matrix that counts adjacencies between characters.
    Characters with higher adjacency counts are more likely to be classified as vowels, a decision based on the assumption
    that vowels generally appear more frequently and adjacent to a variety of other characters.

    Args:
    text (str): The text to be analyzed and processed.
    preprocess (bool): Indicates whether the text should be preprocessed to normalize it (e.g., converting to lowercase and removing non-alphabetic characters).
                       The default value is True, which applies preprocessing.

    Returns:
    tuple: A tuple containing two sets:
           - The first set includes characters classified as vowels.
           - The second set includes characters classified as consonants.
    """
    # Conditionally preprocess the text to remove non-alphabetic characters and convert to lowercase for uniformity
    if preprocess:
        text = preprocess_text(text)

    # Generate a frequency matrix and a character-to-index mapping from the processed text
    matrix, char_to_index = create_frequency_matrix(text)
    
    # Define a lambda function to compute the sum of adjacencies for each character in the matrix
    # The lambda function takes a matrix 'm' and computes the sum across its rows (axis=1)
    sum_rows = lambda m: m.sum(axis=1)
    
    # Calculate the adjacency sums which will be used to classify characters
    sums = sum_rows(matrix)
    
    # Classify characters into vowels and consonants based on the calculated sums and the frequency matrix
    vowels, consonants = classify_vowels(sums, matrix, char_to_index)

    # Return the classified sets of vowels and consonants
    return vowels, consonants

def process_gutenberg_corpus(preprocess):
    """
    Process text from the gutenberg Corpus using Suxotin's algorithm.

    Args:
    preprocess (bool): Whether to preprocess the text.

    Returns:
    tuple: Sets of classified vowels and consonants.
    """
    # Concatenate all items in the gutenberg Corpus into a single string
    text = ' '.join(gutenberg.words())
    return suxotins_algorithm(text, preprocess)

def get_preprocess_confirmation():
    """
    Prompt the user to confirm if they want to preprocess the text.
    Accepts any input starting with 'y' or 'n' as a valid response and is case insensitive.

    Returns:
    bool: True if the user confirms preprocessing, False otherwise.
    """
    while True:
        preprocess_input = input("Do you want to preprocess the text? (yes/no): ").strip().lower()
        if preprocess_input.startswith('y'):
            return True
        elif preprocess_input.startswith('n'):
            return False
        else:
            print("Invalid input. Please answer with 'yes' or 'no'.")

def main():
    """
    Main execution function to process a text file or Gutenberg Corpus and classify characters using Suxotin's algorithm.
    Filters and prints sorted, printable vowels and consonants, excluding spaces and newlines.
    Enhances user input flexibility and organizes output for better readability.
    """
    # Gather user inputs first
    data_source = input("Choose the data source - 'local' for local file, 'nltk' for Gutenberg Corpus: ").lower()
    preprocess = get_preprocess_confirmation()

    # Start timing after user input
    start_time = time.perf_counter()

    try:
        if data_source == 'local':
            file_path = Path('sherlock_holmes.txt')
            with file_path.open('r', encoding='utf-8') as file:
                text = file.read()
        elif data_source == 'nltk':
            text = ' '.join(gutenberg.words())
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
        print(str(e))
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

    end_time = time.perf_counter()  # End timing
    print(f"Execution time: {end_time - start_time:.4f} seconds")  # Print execution time

if __name__ == '__main__':
    main()
