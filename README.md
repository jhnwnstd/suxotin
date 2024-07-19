# Suxotin's Vowel Identification Algorithm

## Overview
This repository contains an implementation of Suxotin's algorithm, designed to identify vowels in texts by analyzing their formal properties. Using statistical analysis, the algorithm examines letter combinations to classify letters based on their adjacency patterns. This algorithm is especially useful for linguists and decipherment enthusiasts.

Although the underlying principles of the algorithm are not extensively explained in the literature, its efficacy is evident. For example, in the `sherlock_holmes.txt` file, the algorithm accurately identified all relevant vowels with minimal errors:

```
Classified vowels: ['a', 'e', 'i', 'o', 'u', 'â', 'æ', 'è', 'é']
Classified consonants: ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'y', 'z', 'à', 'œ']
```

Testing on the NLTK Gutenberg corpus yielded similar accuracy, confirming its robustness:

```
Classified Vowels: ['a', 'e', 'i', 'o', 'u', 'æ', 'è']
Classified Consonants: ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'y', 'z', 'é', 'î']
```

The algorithm occasionally misclassified low-frequency vowels like 'à', 'œ', 'é' and 'î' as consonants.

## Features
- **Language Agnostic**: The algorithm can process texts in multiple languages and accurately identify vowel patterns regardless of the language.
- **Fast and Efficient**: By leveraging NumPy for matrix operations, the algorithm achieves high computational speed and efficiency.

## Why the Algorithm Works
Suxotin's algorithm is effective because it leverages two key linguistic and statistical principles observed in natural languages:

1. **Alternation of Vowels and Consonants**:
    - Vowels and consonants often alternate in texts. This means that vowels are frequently adjacent to consonants and vice versa. Clusters of vowels or consonants do occur but are less common.
    - This alternation results in high adjacency counts between vowels and consonants, which is captured in the frequency matrix.

2. **Prevalence of Vowels**:
    - Vowels typically occur more frequently in texts than individual consonants because they are essential for the phonetic structure of words.
    - Therefore, in any sufficiently long text, vowels will have higher overall counts and higher adjacency counts due to their frequent alternation with consonants.

### Algorithm Steps
1. **Frequency Matrix Creation**:
    - Create a matrix where each cell (i, j) represents the number of times character i is adjacent to character j.
    - High values in this matrix indicate frequent adjacency.

2. **Row Sums**:
    - The sum of each row in the frequency matrix provides a measure of how frequently each character appears next to others.
    - Vowels, due to their frequent alternation with consonants, will generally have higher row sums compared to consonants.

3. **Iterative Classification**:
    - Classify the character with the highest adjacency sum as a vowel.
    - Adjust the sums of the remaining characters by subtracting twice the adjacency counts involving the newly classified vowel, reducing the likelihood of adjacent characters being misclassified.
    - Repeat until no positive sums are left.

### Mathematical Foundation
- **Markov Chains**: The algorithm can be viewed through the lens of Markov chains, where state transitions (i.e., moving from one character to another) are influenced by the classification of previous characters.
- **Maximum Likelihood Estimation (MLE)**: The algorithm maximizes the likelihood that the classified vowels and consonants fit the observed adjacency patterns in the text.

These principles, combined with the iterative refinement process, make Suxotin's algorithm effective in separating vowels from consonants based on the statistical behavior of characters in natural language texts.

## Repository Contents
- `suxotin.py`: The primary Python script implementing Suxotin's algorithm with NumPy for efficient matrix operations.
- `sherlock_holmes.txt`: A sample text file used to test the algorithm's performance.
- `suxotin`: A directory containing documentation of two versions of the Suxotin algorithm and a text version of the algorithm for further testing purposes.
   - `Decipherment_Models_1973.pdf`: Details the original version of the Suxotin algorithm.
   - `guySuxotin.pdf`: Details an alternate version of the Suxotin algorithm.
   - `vowel-algorithm.txt`: A text version of the Suxotin algorithm for additional testing purposes.
- `README.md`: Provides an overview and guidance on using the repository's resources.

## Installation
1. **Clone this repository to your local machine:**

   - Using `git`:
     ```bash
     git clone https://github.com/jhnwnstd/suxotin
     ```

   - Using GitHub CLI:
     ```bash
     gh repo clone jhnwnstd/suxotin
     ```

2. **Install the required dependencies:**
   - You need Python 3.11 or higher. Check your Python version by running `python --version` in your terminal.
   - Install NumPy and NLTK, which are essential for running the algorithm:
     ```bash
     pip install numpy nltk
     ```

3. **Download the necessary NLTK data:**
   - After installing NLTK, download the Gutenberg corpus, which is used if you choose the NLTK data source:
     ```python
     import nltk
     nltk.download('gutenberg')
     ```

## Usage
To run the script `suxotin.py` in this repository:
1. **Ensure that Python and the required packages (NumPy and NLTK) are installed on your system.**
2. **Navigate to the directory containing the scripts.** 
   - Open a terminal or command prompt.
   - Change the directory to where `suxotin.py` is located, e.g., `cd path/to/suxotin`.
3. **Execute the script:**
   ```bash
   python suxotin.py
   ```
   - Follow the on-screen prompts to choose between processing a local text file or using the NLTK Gutenberg corpus.
   - You will also be asked whether to preprocess the text. Respond with 'yes' or 'no' based on your preference. It is recommended 'yes' to preprocess the text to remove punctuation and convert all characters to lowercase.

## Contributing
I appreciate contributions to enhance this project and extend the capabilities of the algorithm. To contribute, please fork the repository, make changes, and submit a pull request.