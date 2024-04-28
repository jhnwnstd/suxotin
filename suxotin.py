import re
import nltk
import numpy as np
from pathlib import Path
from nltk.corpus.util import LazyCorpusLoader

def preprocess_text(text):
    # Lowercase the text and remove non-alphabetic characters except spaces
    return re.sub(r'[^a-zA-Z\s]', '', text.lower())

def create_frequency_matrix(text):
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

def separate_vowels_consonants(text):
    # Generate the frequency matrix and retrieve the list of unique characters
    chars, freq_matrix = create_frequency_matrix(text)
    
    # Initialize an array to mark identified vowels (True if vowel, False otherwise)
    is_vowel = np.zeros(len(chars), dtype=bool)
    
    # List to store the identified vowels
    vowels = []
    
    # Compute the sum of entries in each row of the frequency matrix
    row_sums = np.sum(freq_matrix, axis=1)

    # Loop until there are no more significant transitions left to process
    while np.any(row_sums > 0):
        # Find the index of the character with the maximum row sum
        max_idx = np.argmax(row_sums)
        
        # Classify the character at max_idx as a vowel
        is_vowel[max_idx] = True
        
        # Append the identified vowel to the list
        vowels.append(chars[max_idx])
        
        # Adjust row sums for the remaining characters by subtracting twice the transition counts
        # involving the newly identified vowel
        row_sums -= 2 * freq_matrix[:, max_idx]
        
        # Remove the row and column corresponding to the identified vowel from the matrix
        freq_matrix = np.delete(freq_matrix, max_idx, axis=0)
        freq_matrix = np.delete(freq_matrix, max_idx, axis=1)
        
        # Update the row sums and the character list to exclude the identified vowel
        row_sums = np.delete(row_sums, max_idx)
        chars.pop(max_idx)
        is_vowel = np.delete(is_vowel, max_idx)

    # List comprehension to collect the remaining characters classified as consonants
    consonants = [char for i, char in enumerate(chars) if not is_vowel[i]]

    # Return the lists of identified vowels and consonants
    return vowels, consonants

def load_text(source):
    # Check if the provided source is a path that exists on the file system
    if Path(source).exists():
        file_path = Path(source)
        try:
            # Open the file with a specified encoding (ISO-8859-1, commonly used for Western European languages)
            with file_path.open('r', encoding='ISO-8859-1') as file:
                # Read the entire content of the file into a string
                text = file.read()
        except FileNotFoundError:
            # This exception is raised if the file does not exist at the given path
            print(f"Error: The file {source} was not found.")
            return None
        except Exception as e:
            # Catch any other exceptions that may occur during file opening or reading
            print(f"An error occurred: {e}")
            return None
    else:
        try:
            # Ensure the NLTK corpus is available, download it if it's not already downloaded
            nltk.download(source, quiet=True)
            # Initialize a LazyCorpusLoader with the specified corpus using a plaintext reader.
            # This setup tries to read the text with UTF-8 encoding initially.
            nltk_corpus = LazyCorpusLoader(source, nltk.corpus.reader.PlaintextCorpusReader, r'.*\.txt', encoding='utf-8')
            try:
                # Attempt to fetch the raw content of the corpus
                text = nltk_corpus.raw()
            except UnicodeDecodeError:
                # If a UnicodeDecodeError occurs, it suggests the text isn't compatible with UTF-8 encoding.
                # The block retries loading the corpus with ISO-8859-1 encoding, which can handle different character sets.
                nltk_corpus = LazyCorpusLoader(source, nltk.corpus.reader.PlaintextCorpusReader, r'.*\.txt', encoding='iso-8859-1')
                text = nltk_corpus.raw()
        except (LookupError, AttributeError, UnicodeDecodeError) as e:
            # Handle errors related to corpus access or encoding issues
            print(f"Failed to load NLTK corpus '{source}': {e}")
            return None

    # Normalize the text by replacing non-alphanumeric characters and digits with spaces
    # This helps in cleaning the text from any punctuation or unwanted symbols
    text = re.sub(r"[\W\d]+", " ", text)
    # Return the cleaned text with leading and trailing spaces removed
    return text.strip()

# Example usage for file
filename = "sherlock_holmes.txt"
text = load_text(filename)
if text:
    text = preprocess_text(text)
    vowels, consonants = separate_vowels_consonants(text)
    print("Vowels:", vowels)
    print("Consonants:", consonants)

# Example usage for NLTK corpus
nltk.download('gutenberg')
corpus_name = 'gutenberg'
nltk_text = load_text(corpus_name)
if nltk_text:
    nltk_text = preprocess_text(nltk_text)
    vowels, consonants = separate_vowels_consonants(nltk_text)
    print("Vowels from NLTK Corpus:", vowels)
    print("Consonants from NLTK Corpus:", consonants)
