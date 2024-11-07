# Suxotin's Vowel Identification Algorithm

## Overview

This repository contains an implementation of Suxotin's algorithm, designed to identify vowels in texts by analyzing character adjacency patterns. The algorithm utilizes statistical analysis of letter combinations to classify letters based on their adjacency counts, making it a powerful tool for linguists, cryptanalysts, and language enthusiasts.

The algorithm operates independently of any specific language, making it applicable to a wide range of texts. It is particularly effective in deciphering unknown languages or scripts by leveraging the statistical properties of letters in natural languages.

## Features

- **Language Agnostic**: Processes texts in any language and accurately identifies vowel patterns regardless of the language's orthography or phonetics.
- **Efficient Computation**: Utilizes NumPy for efficient matrix operations, allowing for the analysis of large texts with high performance.
- **Improved Accuracy**: Implements an adjustable threshold factor to refine the classification of vowels and consonants, ensuring high accuracy across different datasets.
- **Alphabetical Output**: The final classified vowels and consonants are sorted alphabetically for easy interpretation.

## Why the Algorithm Works

Suxotin's algorithm is effective because it leverages fundamental statistical properties observed in natural languages:

1. **Alternation of Vowels and Consonants**: In most languages, vowels and consonants alternate frequently within words. This means that vowels are often adjacent to consonants, and vice versa. By analyzing the adjacency patterns between letters, the algorithm can detect which letters behave like vowels based on their frequent transitions with consonants.

2. **Frequency of Vowels**: Vowels tend to occur more frequently than individual consonants due to their essential role in word formation and pronunciation. They often appear in multiple contexts and combinations, leading to higher overall adjacency counts.

## How the Algorithm Works

### 1. Preprocessing the Text

- The text is converted to lowercase, and all non-alphabetic characters are removed (except for spaces, if desired). This standardizes the input and focuses the analysis on the letters themselves.

### 2. Creating the Frequency Matrix

- A frequency matrix is constructed where each cell `(i, j)` represents the number of times character `i` appears adjacent to character `j` in the text. This matrix is symmetric since adjacency is bidirectional.

### 3. Calculating the Adjacency Sums

- The sum of each row (or column) in the frequency matrix is calculated, representing the total number of adjacencies for each character. Characters with higher adjacency sums are more connected to other characters in the text.

### 4. Iterative Classification of Vowels and Consonants

- The algorithm iteratively selects the character with the highest remaining adjacency sum as a vowel.
- After selecting a vowel, the adjacency sums of all other characters are adjusted by subtracting twice the adjacency counts involving the newly classified vowel. This reduces the influence of already classified vowels on the remaining characters.
- This process continues until all characters have been classified.

### 5. Threshold-Based Reclassification

- To capture any vowels that might have been misclassified due to slight variations in their adjacency sums, a threshold is applied.
- The threshold is calculated based on the minimum adjusted sum among the initially classified vowels, adjusted by a factor (set to **2** in this implementation).
- Consonants with adjusted sums above this threshold are reclassified as vowels.
- This step ensures that vowels with slightly lower adjacency sums are not overlooked.

**Why Factor 2?**

- **Capturing Misclassified Vowels**: Some vowels may have adjusted sums that are slightly negative due to the adjustments made during the iterative classification. By setting the threshold factor to **2**, the threshold is lowered enough to include these vowels in the final classification.
- **Avoiding Overclassification**: Higher threshold factors can lead to consonants with low adjacency sums being misclassified as vowels. By selecting a factor of **2**, we strike a balance between capturing all vowels and avoiding the inclusion of consonants.

### 6. Sorting the Results

- The final lists of vowels and consonants are sorted alphabetically. This makes the results easier to read and compare, especially when analyzing outputs across different texts.

## Example Results

### Using the NLTK Gutenberg Corpus

```
Classified Characters:
Vowels:      a, e, i, o, u, æ, è, é, î
Consonants:  b, c, d, f, g, h, j, k, l, m, n, p, q, r, s, t, v, w, x, y, z

Execution time: 2.8717 seconds
```

### Using `sherlock_holmes.txt`

```
Classified Characters:
Vowels:      a, e, i, o, u, à, â, æ, è, é, œ
Consonants:  b, c, d, f, g, h, j, k, l, m, n, p, q, r, s, t, v, w, x, y, z

Execution time: 0.1365 seconds
```

In both examples, the algorithm successfully identified the vowels, including accented characters such as 'à', 'â', 'æ', 'è', 'é', and 'œ'. The consonants were also correctly classified, demonstrating the algorithm's effectiveness across different texts.

## Repository Contents

- **`suxotin.py`**: The main Python script implementing Suxotin's algorithm with the improvements discussed.
- **`sherlock_holmes.txt`**: A sample text file used for testing the algorithm.
- **Documentation Files**:
  - **`Decipherment_Models_1973.pdf`**: Details the original version of Suxotin's algorithm.
  - **`guySuxotin.pdf`**: Provides an alternative version of the algorithm.
  - **`vowel-algorithm.txt`**: A text version of the algorithm for further testing and exploration.
- **`README.md`**: This file, providing an overview of the algorithm and instructions for use.

## Installation

### Prerequisites

- Python 3.11 or higher
- Pip (Python package installer)

### Steps

1. **Clone the Repository**

   Using HTTPS:

   ```bash
   git clone https://github.com/jhnwnstd/suxotin.git
   ```

   Or using GitHub CLI:

   ```bash
   gh repo clone jhnwnstd/suxotin
   ```

2. **Navigate to the Repository**

   ```bash
   cd suxotin
   ```

3. **Install the Required Dependencies**

   ```bash
   pip install numpy nltk
   ```

4. **Download NLTK Data**

   In a Python shell or script, download the Gutenberg corpus:

   ```python
   import nltk
   nltk.download('gutenberg')
   ```

## Usage

To run the algorithm:

1. **Execute the Script**

   ```bash
   python suxotin.py
   ```

2. **Select the Data Source**

   - **Option 1**: Use the local file `sherlock_holmes.txt`.
   - **Option 2**: Use the NLTK Gutenberg Corpus.

3. **Preprocess the Text**

   - When prompted, it's recommended to choose 'yes' to preprocess the text. This will remove non-alphabetic characters and convert all letters to lowercase, improving the accuracy of the algorithm.

4. **View the Results**

   - The script will display the classified vowels and consonants, sorted alphabetically.