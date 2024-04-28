# Suxotin's Algorithm Implementation

## Overview
This repository contains an implementation of Suxotin's algorithm, designed to distinguish vowels from consonants based on their formal properties within a text. The code is ideal for decipherment hobbyists who are interested in exploring text properties and linguistic patterns.

Suxotin's algorithm utilizes statistical properties of letter combinations within a text to classify letters as either vowels or consonants. This method is based on the hypothesis that vowels and consonants typically alternate in most languages, with vowels being more likely to precede and follow consonants rather than other vowels, and vice versa for consonants.

## Features
- **Text Preprocessing**: Converts text to a lower case and removes non-alphabetic characters, preparing it for analysis.
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

## Background
Suxotin's algorithm was extensively tested in computer experiments across multiple languages including Russian, English, French, German, and Spanish, demonstrating robustness with minimal errors. Detailed descriptions and theoretical justifications of the algorithm are available in the `vowel-algorithm.txt` file.
