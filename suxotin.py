import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from nltk.corpus import gutenberg
from nltk import download

# Ensure the necessary NLTK data is downloaded silently without output
download('gutenberg', quiet=True)

def preprocess_text(text: str) -> str:
    """
    Convert the input text to lowercase and remove all non-alphabetic characters except spaces.

    Args:
        text (str): The text to preprocess.

    Returns:
        str: The preprocessed text with only lowercase letters and spaces.
    """
    # Use a generator expression to iterate over each character in the text
    # Convert each character to lowercase if it's an alphabet or space; otherwise, remove it
    return ''.join(
        char.lower() if char.isalpha() or char.isspace() else ''  # Keep lowercase letters and spaces
        for char in text
    )

def create_frequency_matrix(text: str) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Create a symmetric frequency matrix of adjacent characters in the text.

    Args:
        text (str): The text from which to create the frequency matrix.

    Returns:
        Tuple[np.ndarray, Dict[str, int]]: The frequency matrix and character-to-index mapping.
    """
    # Get a sorted list of unique characters to maintain consistent ordering
    unique_chars = sorted(set(text))
    # Create a mapping from character to a unique index
    char_to_index = {char: idx for idx, char in enumerate(unique_chars)}
    # Determine the size of the frequency matrix
    size = len(unique_chars)

    # Map text characters to their corresponding indices using NumPy
    text_indices = np.fromiter(
        (char_to_index[char] for char in text),  # Generator expression over characters
        dtype=np.int32,  # Specify data type for efficiency
        count=len(text)   # Total number of characters
    )

    # Get pairs of adjacent character indices
    left_indices = text_indices[:-1]   # All indices except the last
    right_indices = text_indices[1:]   # All indices except the first

    # Create a boolean mask to exclude self-adjacency (where left and right indices are the same)
    mask = left_indices != right_indices
    # Apply the mask to filter out self-adjacency pairs
    left_indices = left_indices[mask]
    right_indices = right_indices[mask]

    # Initialize the adjacency matrix with zeros
    matrix = np.zeros((size, size), dtype=np.int32)
    # Use NumPy's advanced indexing to increment counts for each pair of indices
    np.add.at(matrix, (left_indices, right_indices), 1)  # Increment for (left, right) pairs
    np.add.at(matrix, (right_indices, left_indices), 1)  # Increment for (right, left) pairs to keep matrix symmetric

    return matrix, char_to_index  # Return the adjacency matrix and character-to-index mapping

def classify_vowels(
    sums: np.ndarray, matrix: np.ndarray, index_to_char: Dict[int, str]
) -> Tuple[List[str], List[str]]:
    """
    Classify characters as vowels or consonants based on adjacency counts,
    using a threshold to capture misclassified vowels.

    Args:
        sums (np.ndarray): Sum of adjacency counts for each character.
        matrix (np.ndarray): The frequency matrix of character adjacencies.
        index_to_char (Dict[int, str]): Mapping from indices to characters.

    Returns:
        Tuple[List[str], List[str]]: Lists of vowels and consonants in the order they were classified.
    """
    num_chars = len(sums)
    remaining_sums = sums.astype(np.float64)
    vowels = []
    consonants = []
    adjusted_sums = np.zeros(num_chars, dtype=np.float64)  # To store final adjusted sums

    for _ in range(num_chars):
        max_idx = np.argmax(remaining_sums)
        char = index_to_char[max_idx]
        adjusted_sums[max_idx] = remaining_sums[max_idx]  # Store the adjusted sum
        # Classify based on remaining sum
        if remaining_sums[max_idx] > 0:
            vowels.append(char)
        else:
            consonants.append(char)
        adjustment = matrix[:, max_idx] * 2
        remaining_sums -= adjustment
        remaining_sums[max_idx] = -np.inf

    # Determine threshold based on minimum adjusted sum among vowels
    if vowels:
        vowel_indices = [idx for idx, c in index_to_char.items() if c in vowels]
        min_vowel_sum = adjusted_sums[vowel_indices].min()
    else:
        min_vowel_sum = 0

    # Set a threshold slightly below min_vowel_sum
    threshold = min_vowel_sum - (abs(min_vowel_sum) * 2)  # Adjust the factor as needed

    # Reclassify consonants with adjusted sums above the threshold as vowels
    reclassified_vowels = []
    new_consonants = []
    for char in consonants:
        idx = next(idx for idx, c in index_to_char.items() if c == char)
        if adjusted_sums[idx] >= threshold:
            reclassified_vowels.append(char)
        else:
            new_consonants.append(char)

    # Combine the original vowels with reclassified vowels
    vowels.extend(reclassified_vowels)

    # The consonants list now consists of new_consonants
    consonants = new_consonants

    return vowels, consonants  # Return the classified lists

def suxotins_algorithm(text: str, preprocess: bool = True) -> Tuple[List[str], List[str]]:
    """
    Apply Suxotin's algorithm to classify characters into vowels and consonants.

    Args:
        text (str): The text to analyze.
        preprocess (bool): Whether to preprocess the text.

    Returns:
        Tuple[List[str], List[str]]: Lists of vowels and consonants.
    """
    if preprocess:
        # Preprocess the text to remove non-alphabetic characters and convert to lowercase
        text = preprocess_text(text)

    # Create the frequency matrix and character-to-index mapping from the text
    matrix, char_to_index = create_frequency_matrix(text)
    # Calculate the sum of adjacency counts for each character
    sums = matrix.sum(axis=1)
    # Create a reverse mapping from indices to characters for easy lookup
    index_to_char = {idx: char for char, idx in char_to_index.items()}
    # Classify characters into vowels and consonants
    vowels, consonants = classify_vowels(sums, matrix, index_to_char)

    return vowels, consonants  # Return the classified lists

def get_preprocess_confirmation() -> bool:
    """
    Prompt the user to confirm if they want to preprocess the text.

    Returns:
        bool: True if the user confirms preprocessing, False otherwise.
    """
    while True:
        # Prompt the user for preprocessing confirmation
        response = input("Do you want to preprocess the text? (yes/no): ").strip().lower()
        if response in {'yes', 'y'}:
            return True   # User confirms preprocessing
        elif response in {'no', 'n'}:
            return False  # User declines preprocessing
        else:
            # Invalid input; prompt again
            print("Invalid input. Please answer with 'yes' or 'no'.")

def select_data_source() -> str:
    """
    Prompt the user to select the data source.

    Returns:
        str: The text to analyze.
    """
    while True:
        # Present options for data source selection
        choice = input(
            "Choose the data source:\n"
            "1 - Local file ('sherlock_holmes.txt')\n"
            "2 - NLTK Gutenberg Corpus\n"
            "Enter 1 or 2: "
        ).strip()
        if choice == '1':
            # User selects local file
            file_path = Path('sherlock_holmes.txt')
            if not file_path.exists():
                # Notify user if file does not exist
                print(f"File '{file_path}' not found.")
                continue  # Prompt again
            # Read and return the content of the file
            with file_path.open('r', encoding='utf-8') as file:
                return file.read()
        elif choice == '2':
            # User selects NLTK Gutenberg Corpus
            return ' '.join(gutenberg.words())  # Return the corpus as a single string
        else:
            # Invalid input; prompt again
            print("Invalid choice. Please enter 1 or 2.")

def main():
    """
    Main function to execute the classification algorithm and display results.
    """
    # Get the text to analyze based on user's choice
    text = select_data_source()
    # Ask user whether to preprocess the text
    preprocess = get_preprocess_confirmation()
    # Record the start time for performance measurement
    start_time = time.perf_counter()

    # Apply Suxotin's algorithm to classify vowels and consonants
    vowels, consonants = suxotins_algorithm(text, preprocess)

    # Remove non-printable characters (e.g., spaces, tabs, newlines) from the results
    vowels = [v for v in vowels if v not in {' ', '\n', '\t'}]
    consonants = [c for c in consonants if c not in {' ', '\n', '\t'}]

    # Display the classified characters
    print("\nClassified Characters:")
    print("Vowels:     ", ', '.join(vowels))
    print("Consonants: ", ', '.join(consonants))

    # Record the end time and calculate execution duration
    end_time = time.perf_counter()
    print(f"\nExecution time: {end_time - start_time:.4f} seconds")

# Entry point of the script; execute main() if the script is run directly
if __name__ == '__main__':
    main()