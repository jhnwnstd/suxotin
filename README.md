# Suxotin's Vowel Identification Algorithm

## Overview
This repository hosts two implementations of Suxotin's algorithm, which is designed to distinguish vowels from consonants by analyzing their formal properties in texts. It is ideally suited for linguists and decipherment hobbyists. The algorithm employs statistical analysis of letter combinations to classify characters based on their adjacency patterns across various languages.

Although the exact mechanisms behind the algorithm's effectiveness are not extensively detailed in the literature, its intriguing results underscore the need for further empirical study and discussion.

While both implementations identify consonants, occasional misclassifications of 'n' and 't' as vowels highlight areas for potential refinement.

## Background
Suxotin's algorithm has been tested in computer experiments across multiple languages, including Russian, English, French, German, and Spanish. These tests have shown the algorithm to be highly robust, with minimal classification errors. The algorithm's theoretical and operational principles are more comprehensively explained in the `vowel-algorithm.sty` file.

## Features
- **Text Preprocessing**: Standardizes input text by converting to lowercase and filtering out non-alphabetic characters, preparing it for analysis.
- **Frequency Matrix Generation**: Constructs a matrix that counts transitions between each pair of characters within words, serving as the foundation for analysis.
- **Vowel and Consonant Classification**: Determines probable vowels and consonants using the transition frequency matrix, leveraging statistical patterns.

## Repository Contents
- `suxotin.py`: Primary Python script implementing Suxotin's algorithm using NumPy for efficient matrix operations.
- `suxotin2.py`: An alternative implementation.
- `sherlock_holmes.txt`: A sample text file used for testing the algorithm's efficacy.
- `vowel-algorithm.sty`: Provides a detailed narrative description of the algorithm's theoretical underpinnings.
- `README.md`: This document, which offers an overview and guidance on how to utilize the resources in this repository.

## Usage
To use the scripts in this repository, follow these instructions:
1. Ensure Python is installed on your system.
2. Navigate to the directory containing the scripts and run them with a specified text file:
   - For `suxotin.py` : `python suxotin.py sherlock_holmes.txt`
   - For `suxotin2.py`: `python suxotin2.py sherlock_holmes.txt`

## Installation
1. Clone this repository to your local machine.
2. Install the required dependencies:
   ```
   pip install numpy
   ```
   Note: Installing NumPy is essential for running `suxotin.py`, as it depends on matrix operations optimized by this library.