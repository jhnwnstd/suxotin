# Suxotin's Vowel Identification Algorithm

## Overview
This repository hosts an implementation of Suxotin's algorithm, designed to distinguish vowels from consonants by analyzing their formal properties in texts. It is intended for use by both linguistics researchers and decipherment hobbyists. The algorithm utilizes statistical properties of letter combinations to classify characters based on their typical alternation patterns in various languages.

The exact reasons for the algorithm's effectiveness are not detailed in existing literature, underscoring the importance of further study.

Currently, the implementation effectively identifies consonants, but it mistakenly classifies 'n' and 't' as vowels on occasion. Enhancements are needed to improve its robustness and to expand its usability with different text corpora.

## Background
Suxotin's algorithm was extensively tested in computer experiments across multiple languages including Russian, English, French, German, and Spanish, demonstrating robustness with minimal errors. Detailed descriptions of the algorithm are available in the `vowel-algorithm.sty` file.

## Features
- **Text Preprocessing**: Converts text to lowercase and removes non-alphabetic characters, preparing it for analysis.
- **Frequency Matrix Generation**: Creates a matrix to count transitions between each pair of characters within words.
- **Vowel and Consonant Separation**: Uses the generated matrix to determine the most likely vowels and consonants based on the frequency of their transitions.

## Repository Contents
- `suxotin.py`: Main Python script implementing Suxotin's algorithm.
- `vowel-algorithm.sty`: Contains two prose descriptions of Suxotin's algorithm.
- `README.md`: This file, providing an overview and usage instructions.

## Usage
The script can handle text input directly from files within the working directory or from predefined NLTK corpora. Here are the steps to run the script:
1. Ensure you have Python and NLTK installed, and the required NLTK corpora are downloaded.
2. Execute `suxotin.py` with a specified text source:
   - For local files: `python suxotin.py sherlock_holmes.txt`
   - For NLTK corpora: `python suxotin.py gutenberg`

## Installation
1. Clone this repository.
2. Install the necessary Python packages:
   ```
   pip install numpy nltk
   ```
3. Ensure NLTK corpora are available:
   ```
   python -m nltk.downloader gutenberg
   ```
