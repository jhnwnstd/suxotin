import re
import numpy as np
from pathlib import Path

def preprocess_text(text):
    # Lowercase the text and remove non-alphabetic characters except spaces
    return re.sub(r'[^a-zA-Z\s]', '', text.lower())

def create_frequency_matrix(text):
    words = text.split()  # Split the text into words
    unique_chars = sorted(set(text.replace(" ", "")))  # Get unique characters, excluding spaces
    n = len(unique_chars)
    matrix = np.zeros((n, n), dtype=int)

    # Process each word individually
    for word in words:
        prev_char = None
        for char in word:
            if prev_char is not None:
                row = unique_chars.index(prev_char)
                col = unique_chars.index(char)
                matrix[row, col] += 1
            prev_char = char

    return unique_chars, matrix

def separate_vowels_consonants(text):
    chars, freq_matrix = create_frequency_matrix(text)
    n = len(chars)
    is_vowel = np.zeros(n, dtype=bool)
    vowels = []
    row_sums = np.sum(freq_matrix, axis=1)

    while np.any(row_sums > 0):
        max_idx = np.argmax(row_sums)
        is_vowel[max_idx] = True

        for i in range(len(is_vowel)):
            if not is_vowel[i]:
                row_sums[i] -= 2 * freq_matrix[i, max_idx]

        freq_matrix = np.delete(freq_matrix, max_idx, axis=0)
        freq_matrix = np.delete(freq_matrix, max_idx, axis=1)
        row_sums = np.delete(row_sums, max_idx)
        vowels.append(chars[max_idx])
        chars.pop(max_idx)
        is_vowel = np.delete(is_vowel, max_idx)

    consonants = [char for i, char in enumerate(chars) if not is_vowel[i]]

    return vowels, consonants

def load_text_file(filename):
    file_path = Path.cwd() / filename
    try:
        with file_path.open('r', encoding='ISO-8859-1') as file:
            text = file.read()
            text = re.sub(r"\W+", " ", text)
            text = re.sub(r"\d+", "", text)
            text = re.sub(r"\s+", " ", text)
            return text
    except FileNotFoundError:
        print(f"Error: The file {filename} was not found in {file_path.parent}.")
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

# Example usage
filename = "sherlock_holmes.txt"
text = load_text_file(filename)
if text is not None:
    text = preprocess_text(text)
    vowels, consonants = separate_vowels_consonants(text)
    print("Vowels:", vowels)
    print("Consonants:", consonants)
