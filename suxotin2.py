from collections import defaultdict, Counter

def read_text(filename):
    """
    Reads and processes text from a specified file.

    Returns:
    str: The processed text containing only lowercase alphabetical characters.

    Raises:
    IOError: An error occurred accessing the file.
    """
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            # Read the entire file and filter out non-alphabetical characters
            return ''.join(c for c in file.read().lower() if c.isalpha())
    except FileNotFoundError:
        print(f"Error: The file {filename} does not exist.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def create_frequency_matrix(text):
    """
    Create a frequency matrix from a given text which counts each adjacent pair of characters.
    
    Returns:
    tuple: A tuple containing the frequency matrix (as a defaultdict of Counters) and a sorted list of unique letters.
    """
    if len(text) < 2:
        raise ValueError("Text must contain at least two characters to form pairs.")

    freq_matrix = defaultdict(Counter)
    for i in range(len(text) - 1):
        # Increment the count for forward pair
        freq_matrix[text[i]][text[i+1]] += 1
        # Increment the count for reverse pair to maintain symmetry
        freq_matrix[text[i+1]][text[i]] += 1

    # Extract and sort the unique letters involved in the pairs
    letters = sorted(freq_matrix.keys())
    return freq_matrix, letters

def classify_letters(freq_matrix, letters):
    """
    Classify letters as vowels or consonants based on their connection frequencies.
    
    Returns:
    tuple: A tuple containing two lists (vowels, consonants).
    """
    sum_connections = {letter: sum(freq_matrix[letter].values()) for letter in letters}
    vowels, consonants = set(), set(letters)

    while sum_connections:
        # Identify the letter with the highest sum of connections
        vowel = max(sum_connections, key=sum_connections.get)
        vowels.add(vowel)
        consonants.remove(vowel)
        
        # Prepare updates for sum_connections to reflect the new classification
        updates = {letter: 2 * freq_matrix[letter][vowel] for letter in consonants if vowel in freq_matrix[letter]}
        
        # Apply the updates safely
        for letter, deduction in updates.items():
            sum_connections[letter] = sum_connections.get(letter, 0) - deduction

        # Clean up: remove processed vowel and non-positive sums
        sum_connections.pop(vowel, None)
        sum_connections = {k: v for k, v in sum_connections.items() if v > 0}

    return sorted(vowels), sorted(consonants)

def main():
    """
    Main function to execute the vowel and consonant separation algorithm.
    """
    filename = 'sherlock_holmes.txt'
    text = read_text(filename)
    if text:
        freq_matrix, letters = create_frequency_matrix(text)
        vowels, consonants = classify_letters(freq_matrix, letters)
        print("Vowels:", vowels)
        print("Consonants:", consonants)
    else:
        print("Failed to process text due to an error or empty file.")

if __name__ == "__main__":
    main()