import sys
from pathlib import Path
import numpy as np

def load_text(filename):
    """Reads and processes text from a specified file, returning only lowercase alphabetical characters."""
    file_path = Path(filename)
    try:
        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError(f"Error: The file {filename} does not exist.")
        with file_path.open('r', encoding='ISO-8859-1') as file:
            return ''.join(filter(str.isalpha, file.read().lower()))
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return None
    except IOError as e:
        print(f"File read error: {e}", file=sys.stderr)
        return None

def preprocess_text(text):
    """
    Preprocess the input text by converting it to lowercase and removing non-alphabetic characters.
    
    Returns:
    str: The preprocessed text containing only lowercase alphabetic characters and spaces.
    """
    # Convert the text to lowercase first to standardize it
    text = text.lower()
    # Use a generator expression to filter out only alphabetic characters and spaces
    return ''.join(c if c.isalpha() or c.isspace() else '' for c in text)

def create_frequency_matrix(text):
    """
    Create a frequency matrix from a given text which counts each adjacent pair of characters.

    Returns:
    tuple: A tuple containing the frequency matrix (as a defaultdict of Counters) and a sorted list of unique letters.
    """
    # Split the input text into individual words
    words = text.split()
    
    # Create a sorted list of unique characters from the text, excluding spaces
    unique_chars = sorted(set(text.replace(" ", "")))
    
    # Create a dictionary that maps each character to a unique index based on sorted order
    char_index = {char: idx for idx, char in enumerate(unique_chars)}
    
    # Initialize a square matrix of zeros with dimensions based on the number of unique characters
    matrix = np.zeros((len(unique_chars), len(unique_chars)), dtype=int)

    # Iterate over each word to fill the frequency matrix with transition counts
    for word in words:
        prev_char = None  # Initialize the previous character variable
        for char in word:  # Iterate over each character in the word
            if prev_char is not None:  # Ensure there is a previous character to compare
                # Increment the matrix cell that corresponds to the transition from prev_char to char
                matrix[char_index[prev_char], char_index[char]] += 1
            prev_char = char  # Update prev_char to the current character for the next iteration

    # Return the list of unique characters and the filled frequency matrix
    return unique_chars, matrix

def find_vowels_and_consonants(unique_chars, freq_matrix):
    """
    Classify characters as vowels or consonants based on their adjacency frequencies in the frequency matrix.

    Args:
    unique_chars (list): List of unique characters.
    freq_matrix (numpy.ndarray): Frequency matrix where each element represents the count of adjacencies between characters.

    Returns:
    dict: A dictionary with characters as keys and 'vowel' or 'consonant' as values.
    """
    sums = np.sum(freq_matrix, axis=1)
    status = np.array(['consonant'] * len(unique_chars))
    
    while np.any(sums > 0):
        vowel_index = np.argmax(sums)
        status[vowel_index] = 'vowel'
        
        # Update sums
        sums -= 2 * freq_matrix[:, vowel_index]
        
        # Disable further modifications from the identified vowel
        freq_matrix[:, vowel_index] = 0
        freq_matrix[vowel_index, :] = 0
        
        # Update sums directly
        sums[vowel_index] = 0

    return dict(zip(unique_chars, status))

def main():
    filename = 'sherlock_holmes.txt'
    raw_text = load_text(filename)
    if raw_text:
        preprocessed_text = preprocess_text(raw_text)  # Use the correct function to preprocess text
        unique_chars, freq_matrix = create_frequency_matrix(preprocessed_text)  # Pass the cleaned, continuous text
        status = find_vowels_and_consonants(unique_chars, freq_matrix)  # Correctly using two parameters here
        vowels = [char for char, st in status.items() if st == 'vowel']
        consonants = [char for char, st in status.items() if st == 'consonant']
        print("Vowels:", vowels)
        print("Consonants:", consonants)
    else:
        print("Failed to process text due to an error or empty file.", file=sys.stderr)

if __name__ == "__main__":
    main()
