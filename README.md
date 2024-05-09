# Suxotin's Vowel Identification Algorithm

## Overview
This repository contains an implementation of Suxotin's algorithm, designed to identify vowels in texts by analyzing their formal properties. Using statistical analysis, the algorithm examines letter combinations to classify letters based on their adjacency patterns. This algorithm is especially useful for linguists and decipherment enthusiasts.

Although the underlying principles of the algorithm are not extensively explained in the literature, its efficacy is evident. For example, in the `sherlock_holmes.txt` file, the algorithm accurately identified all relevant vowels with minimal errors.

`Classified vowels: ['a', 'e', 'i', 'o', 'u', 'â', 'æ', 'è', 'é']`
`Classified consonants: ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'y', 'z', 'à', 'œ']`

Testing on the NLTK Gutenberg corpus yielded similar accuracy, confirming its robustness.

`Classified Vowels: [a, e, i, o, u, æ, è]`
`Classified Consonants: [b, c, d, f, g, h, j, k, l, m, n, p, q, r, s, t, v, w, x, y, z, é, î]`

The algorithm occasionally misclassified low-frequency diacritical vowels like 'à', 'œ', 'é' and 'î' as consonants.

## Features
- **Language Agnostic**: The algorithm can process texts in multiple languages and accurately identify vowel patterns regardless of the language.
- **Fast and Efficient**: By leveraging NumPy for matrix operations, the algorithm achieves high computational speed and efficiency.

## Repository Contents
- `suxotin.py`: The primary Python script implementing Suxotin's algorithm with NumPy for efficient matrix operations.
- `sherlock_holmes.txt`: A sample text file used to test the algorithm's performance.
- `suxotin`: A directory containing documentation of two versions of the Suxotin algorithm and a text version of the algorithm for furhter testing purposes.
   - `Decipherment_Models_1973.pdf`: Details the original version of the Suxotin algorithm.
   - `guySuxotin.pdf`: Details an alternate version of the Suxotin algorithm.
   - `vowel-algorithm.txt`: A text version of the Suxotin algorithm for additional testing purposes.
- `README.md`: Provides an overview and guidance on using the repository's resources.

## Installation
1. **Clone this repository to your local machine:**
   ```bash
   git clone <repository-url>
   ```
   Replace `<repository-url>` with the actual URL of the repository.

2. **Install the required dependencies:**
   - You need Python 3.6 or higher. Check your Python version by running `python --version` in your terminal.
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