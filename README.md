# Suxotin's Vowel Identification Algorithm

## Overview
This repository hosts an implementation of Suxotin's algorithm, designed to identify vowels by analyzing only their formal properties in texts. Ideal for linguists and decipherment enthusiasts, the algorithm employs statistical analysis of letter combinations to classify characters based on their adjacency patterns across various words.

The explaination for why the algorithm works is not detailed in literature, but it has shown promising results, highlighting the potential for further empirical study and discussion. In the sherlock_holmes text file, the algorithm identified all the vowels in the sample text with no incorrect classifications.

`Classified vowels: ['a', 'e', 'i', 'o', 'u', 'â', 'æ', 'è', 'é']`

## Features
- **Multi-Language Support**: The algorithm is capable of processing text in multiple languages, recognizing vowel patterns effectively across linguistic boundaries.
- **Efficient Processing**: Utilizes NumPy for matrix operations, enhancing the efficiency and speed of the computational process.
- **Extensible Framework**: Designed to be adaptable to additional languages and dialects with minimal adjustments.

## Repository Contents
- `suxotin.py`: The primary Python script implementing Suxotin's algorithm using NumPy for efficient matrix operations.
- `sherlock_holmes.txt`: A sample text file used for testing the algorithm's efficacy.
- `vowel-algorithm.sty`: Provides a detailed description of the algorithm in both theoretical and operational forms.
- `README.md`: This document, offering an overview and guidance on utilizing the resources in this repository.

## Usage
To use the scripts in this repository, follow these instructions:
1. Ensure Python and NumPy are installed on your system.
2. Navigate to the directory containing the scripts.
3. Run the script with a specified text file:
   ```bash
   python suxotin.py
   ```
   Follow the on-screen prompts to preprocess the text and view results.

## Installation
1. Clone this repository to your local machine using:
   ```bash
   git clone <repository-url>
   ```
2. Install the required dependencies:
   ```bash
   pip install numpy
   ```
   NumPy is essential for running `suxotin.py` as it relies on matrix operations optimized by this library.

## Contributing
Contributions to this project are welcome! To contribute, please fork the repository, make your changes, and submit a pull request. We appreciate your input in enhancing the algorithm and expanding its capabilities.