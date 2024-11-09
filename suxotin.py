import time
from typing import Dict, List, Tuple
import nltk
import numpy as np

# Ensure necessary NLTK corpora are downloaded
def setup_resources():
    """Download and setup required NLTK language resources."""
    try:
        # Attempt to import the Europarl corpus for use
        from nltk.corpus import europarl_raw
    except LookupError:
        # If the corpus isn't found, download it
        nltk.download('europarl_raw')
        # Import it again after downloading
        from nltk.corpus import europarl_raw

# Call the function to ensure resources are available before proceeding
setup_resources()

# Now we can safely import the europarl_raw corpus
from nltk.corpus import europarl_raw

def get_language_data(language: str) -> str:
    """Get text data for the specified language from the Europarl corpus."""
    # Dictionary mapping each language to its specific corpus within europarl_raw
    corpus_map = {
        'German': europarl_raw.german,
        'French': europarl_raw.french,
        'Spanish': europarl_raw.spanish,
        'Italian': europarl_raw.italian,
        'Dutch': europarl_raw.dutch,
        'Greek': europarl_raw.greek,
        'English': europarl_raw.english,
        'Swedish': europarl_raw.swedish,
        'Portuguese': europarl_raw.portuguese,
        'Finnish': europarl_raw.finnish,
    }
    # Retrieve the corpus based on the language selected
    corpus = corpus_map.get(language)
    if corpus:
        try:
            # Retrieve the first 5000 words and join them into a single string
            text = ' '.join(corpus.words()[:5000])
            return text
        except Exception as e:
            # Handle any issues accessing the corpus
            print(f"Error accessing {language} corpus: {e}")
            return ""
    else:
        # Message if the specified language does not have a corpus
        print(f"No corpus available for language: {language}")
        return ""

def preprocess_text(text: str) -> str:
    """
    Convert the input text to lowercase and remove all non-alphabetic characters except spaces.
    """
    # Use a generator expression to filter text
    return ''.join(
        char.lower() if char.isalpha() or char.isspace() else ''
        for char in text
    )

def create_frequency_matrix(text: str) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Create a symmetric frequency matrix of adjacent characters in the text.
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
    """
    num_chars = len(sums)
    # Copy sums to be adjusted during the classification
    remaining_sums = sums.astype(np.float64)
    # Initialize an array to store the final adjusted sums for thresholding
    adjusted_sums = np.zeros(num_chars, dtype=np.float64)
    vowels = []  # List to hold classified vowels
    consonants = []  # List to hold classified consonants

    # Classify each character one at a time based on its remaining sum
    for _ in range(num_chars):
        max_idx = np.argmax(remaining_sums)  # Index of the max sum in remaining_sums
        char = index_to_char[max_idx]  # Character corresponding to max_idx
        adjusted_sums[max_idx] = remaining_sums[max_idx]  # Store the adjusted sum
        # Classify the character based on whether the sum is positive
        if remaining_sums[max_idx] > 0:
            vowels.append(char)
        else:
            consonants.append(char)
        # Apply adjustment by subtracting twice the adjacency counts of the classified character
        adjustment = matrix[:, max_idx] * 2
        remaining_sums -= adjustment
        remaining_sums[max_idx] = -np.inf  # Set to -inf to exclude from future consideration

    # Determine a threshold based on the minimum adjusted sum among vowels
    if vowels:
        vowel_indices = [idx for idx, c in index_to_char.items() if c in vowels]
        min_vowel_sum = adjusted_sums[vowel_indices].min()
    else:
        min_vowel_sum = 0

    # Calculate the threshold to reclassify consonants as vowels if necessary
    threshold = min_vowel_sum - abs(min_vowel_sum) * 2 # Adjust threshold as needed

    # Reclassify consonants with adjusted sums above the threshold as vowels
    reclassified_vowels = []
    new_consonants = []
    for char in consonants:
        idx = next(idx for idx, c in index_to_char.items() if c == char)
        if adjusted_sums[idx] >= threshold:
            reclassified_vowels.append(char)
        else:
            new_consonants.append(char)

    vowels.extend(reclassified_vowels)  # Add reclassified vowels to the main list
    consonants = new_consonants  # Update consonants list with remaining consonants
    return vowels, consonants

def suxotins_algorithm(text: str, preprocess: bool = True) -> Tuple[List[str], List[str]]:
    """
    Apply Suxotin's algorithm to classify characters into vowels and consonants.
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

def run_analysis_for_language(language: str, preprocess: bool):
    """Run Suxotin's algorithm for a given language and display results."""
    print(f"\nGetting {language} text data...")
    text = get_language_data(language)
    if not text:
        print(f"Error: No text data available for {language}")
        return
    # Record start time
    start_time = time.perf_counter()
    # Run classification
    vowels, consonants = suxotins_algorithm(text, preprocess)
    # Filter and sort results
    vowels = sorted([v for v in vowels if v.strip()])
    consonants = sorted([c for c in consonants if c.strip()])
    # Display results
    print(f"\n{language} Classification Results:")
    print("Vowels:     ", ', '.join(vowels))
    print("Consonants: ", ', '.join(consonants))
    # Show execution time
    end_time = time.perf_counter()
    print(f"\nExecution time: {end_time - start_time:.4f} seconds")

def main():
    """
    Main function with multilingual support for analyzing all or selected languages.
    """
    languages = ['German', 'French', 'Spanish', 'Italian', 'Dutch', 'Greek', 'English', 'Swedish', 'Portuguese', 'Finnish']
    print("\nAvailable options:")
    print("0 - Run analysis for all languages")
    for idx, lang in enumerate(languages, 1):
        print(f"{idx} - {lang}")

    choice = input("\nSelect a language by number or choose 0 to analyze all languages: ").strip()
    preprocess = input("Preprocess text? (yes/no): ").strip().lower().startswith('y')

    if choice == '0':
        # Run analysis for all languages
        for language in languages:
            run_analysis_for_language(language, preprocess)
    elif choice.isdigit() and 1 <= int(choice) <= len(languages):
        # Run analysis for a specific language
        language = languages[int(choice) - 1]
        run_analysis_for_language(language, preprocess)
    else:
        print("Invalid choice. Please restart and enter a valid number.")

# Run the main function when the script is executed
if __name__ == '__main__':
    main()