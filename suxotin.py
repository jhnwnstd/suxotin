import nltk
import re
import numpy as np
from pathlib import Path

def preprocess_text(text):
    # Lowercase the text and remove non-alphabetic characters except spaces
    return re.sub(r'[^a-zA-Z\s]', '', text.lower())

def create_frequency_matrix(text):
    words = text.split()
    unique_chars = sorted(set(text.replace(" ", "")))
    char_index = {char: idx for idx, char in enumerate(unique_chars)}
    matrix = np.zeros((len(unique_chars), len(unique_chars)), dtype=int)

    for word in words:
        prev_char = None
        for char in word:
            if prev_char is not None:
                matrix[char_index[prev_char], char_index[char]] += 1
            prev_char = char

    return unique_chars, matrix

def separate_vowels_consonants(text):
    chars, freq_matrix = create_frequency_matrix(text)
    is_vowel = np.zeros(len(chars), dtype=bool)
    vowels = []
    row_sums = np.sum(freq_matrix, axis=1)

    while np.any(row_sums > 0):
        max_idx = np.argmax(row_sums)
        is_vowel[max_idx] = True
        row_sums -= 2 * freq_matrix[:, max_idx]
        vowels.append(chars[max_idx])
        freq_matrix = np.delete(freq_matrix, max_idx, axis=0)
        freq_matrix = np.delete(freq_matrix, max_idx, axis=1)
        row_sums = np.delete(row_sums, max_idx)
        chars.pop(max_idx)
        is_vowel = np.delete(is_vowel, max_idx)

    consonants = [char for i, char in enumerate(chars) if not is_vowel[i]]

    return vowels, consonants

def load_text(source):
    if Path(source).exists():
        # Assume source is a file path
        file_path = Path(source)
        try:
            with file_path.open('r', encoding='ISO-8859-1') as file:
                text = file.read()
        except FileNotFoundError:
            print(f"Error: The file {source} was not found.")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
    else:
        # Assume source is an NLTK corpus name
        try:
            text = nltk.corpus.__getattr__(source).raw()
        except AttributeError:
            print(f"Error: NLTK corpus '{source}' not found.")
            return None
    text = re.sub(r"[\W\d]+", " ", text)  # Normalize text
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
