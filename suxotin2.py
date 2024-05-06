from collections import defaultdict

def preprocess_text(text):
    """
    Preprocess the input text by converting it to lowercase and removing non-alphabetic characters.
    """
    return ''.join(char.lower() if char.isalpha() or char.isspace() else '' for char in text)

def create_frequency_matrix(text):
    """
    Create a frequency matrix from a given text which counts each adjacent pair of characters.
    """
    # Create a frequency matrix using a defaultdict of defaultdicts, initializing missing values to int (0)
    matrix = defaultdict(lambda: defaultdict(int))
    # Generate pairs of adjacent characters, considering the text as circular
    pairs = zip(text, text[1:] + text[:1])
    # Populate the matrix with counts of each pair occurrence
    for left, right in pairs:
        matrix[left][right] += 1
        matrix[right][left] += 1
    # Zero out the diagonal (self-pairings) as they are not relevant for this analysis
    for letter in matrix:
        matrix[letter][letter] = 0
    return matrix

def sum_rows(matrix):
    """
    Sum up the values in each row of the matrix to get the total counts of adjacencies for each character.
    """
    # Sum up the values in each row of the matrix to get the total counts of adjacencies for each character
    return {letter: sum(connections.values()) for letter, connections in matrix.items()}

def classify_vowels(sums, matrix):
    """
    Classify vowels based on the adjacency sums of each character.
    """
    # Initialize a set to keep track of classified vowels
    classified_vowels = set()
    # Loop until all entries are either processed or sums are zero or negative
    while any(value > 0 for value in sums.values()):
        # Identify the character with the maximum adjacency sum, considering it a vowel
        vowel = max(sums, key=sums.get)
        classified_vowels.add(vowel)
        # Subtract twice the adjacency count from all other characters' sums
        for letter in list(sums):
            sums[letter] -= 2 * matrix[letter][vowel]
        # Reset the sum of the newly classified vowel to zero to prevent reclassification
        sums[vowel] = 0
    return classified_vowels

def suxotins_algorithm(text, preprocess=True):
    """
    Apply Suxotin's algorithm to classify vowels in a given text.
    """
    # Conditionally preprocess text based on user choice
    if preprocess:
        text = preprocess_text(text)
    # Generate a frequency matrix from the text
    matrix = create_frequency_matrix(text)
    # Calculate the sum of connections for each letter
    sums = sum_rows(matrix)
    # Classify letters as vowels based on their adjacency sums
    return classify_vowels(sums, matrix)

def main():
    """
    Main function to run Suxotin's algorithm on the Sherlock Holmes text.
    """
    # Path to the Sherlock Holmes text file in your working directory
    file_path = 'sherlock_holmes.txt'
    # Ask the user whether to preprocess the text or not
    preprocess = input("Do you want to preprocess the text? (yes/no): ").lower() == 'yes'
    try:
        # Open and read the file
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            # Run the Suxotin's algorithm and print classified vowels
            vowels = suxotins_algorithm(text, preprocess)
            print("Classified vowels:", vowels)
    except FileNotFoundError:
        # Handle the case where the file is not found
        print("File not found. Ensure the file is in the correct directory.")
    except Exception as e:
        # Handle other exceptions that may occur
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    main()