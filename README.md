# Suxotin's Vowel Identification Algorithm

## Overview

This repository contains an implementation of Suxotin's algorithm, designed to identify vowels in texts by analyzing character adjacency patterns. By leveraging the statistical properties of letters in natural languages, the algorithm can classify letters as vowels or consonants, making it a valuable tool for linguists, cryptanalysts, and language enthusiasts.

The algorithm operates independently of any specific language, making it applicable to a wide range of texts. Its versatility allows it to identify vowel patterns across different languages with high precision.

## Features

- **Language Agnostic**: Processes texts in any language and identifies vowel patterns, regardless of the language's orthography.
- **Efficient Computation**: Utilizes NumPy for matrix operations, enabling high performance for large texts.
- **Precision-Focused Classification**: Refines vowel and consonant classification through an adjustable threshold, achieving high accuracy across various datasets.
- **Alphabetical Output**: The results are sorted alphabetically for easy interpretation.

## Why the Algorithm Works

Suxotin's algorithm relies on fundamental statistical properties observed in natural languages:

1. **Alternation of Vowels and Consonants**: Vowels and consonants frequently alternate in words, leading to characteristic adjacency patterns. This behavior helps the algorithm detect letters that function like vowels.
  
2. **Frequency of Vowels**: Vowels tend to appear frequently and in multiple combinations, creating higher adjacency counts, which the algorithm leverages for classification.

## Algorithm Workflow

### 1. Preprocessing the Text

- Text is converted to lowercase, and non-alphabetic characters are removed, standardizing the input for analysis.

### 2. Creating the Frequency Matrix

- A symmetric matrix is created, where each cell `(i, j)` represents the number of times character `i` is adjacent to character `j` in the text.

### 3. Calculating Adjacency Sums

- Each character’s adjacency count is summed, highlighting characters with high connectivity to others, often vowels.

### 4. Iterative Classification of Vowels and Consonants

- Characters with the highest remaining adjacency sums are iteratively classified as vowels, with adjustments made to minimize the influence of previously classified vowels.

### 5. Threshold-Based Reclassification

- A threshold, set to **2** in this implementation, ensures that vowels with slightly lower adjacency sums are not misclassified. This balance helps capture all vowels while avoiding overclassification.

### 6. Sorting the Results

- Classified vowels and consonants are sorted alphabetically for clarity.

## Performance and Evaluation

The algorithm was tested across multiple languages with distinct vowel and consonant sets. Here are the results, including precision, recall, and F1 scores:

| **Language** | **True Positives (TP)** | **False Positives (FP)** | **False Negatives (FN)** | **Precision** | **Recall** | **F1 Score** |
|--------------|-------------------------|--------------------------|--------------------------|---------------|------------|--------------|
| **German**   | 9                       | 1                        | 0                        | 0.9000        | 1.0000     | 0.9474       |
| **French**   | 12                      | 0                        | 2                        | 1.0000        | 0.8571     | 0.9231       |
| **Spanish**  | 9                       | 1                        | 1                        | 0.9000        | 0.9000     | 0.9000       |
| **Italian**  | 7                       | 0                        | 4                        | 1.0000        | 0.6364     | 0.7778       |
| **Dutch**    | 10                      | 0                        | 1                        | 1.0000        | 0.9091     | 0.9524       |
| **Greek**    | 14                      | 0                        | 1                        | 1.0000        | 0.9333     | 0.9655       |
| **English**  | 5                       | 0                        | 1                        | 1.0000        | 0.8333     | 0.9091       |

### Interpretation

- **High Precision**: Precision is perfect (1.0) for most languages, indicating that the algorithm correctly identifies vowels without misclassifying consonants.
- **Balanced F1 Scores**: F1 scores are high across most languages, with the algorithm performing exceptionally well in Greek, Dutch, and German.
- **Recall Considerations**: Languages like Italian and French have lower recall due to missed vowels, suggesting that minor adjustments could improve accuracy.

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

## Repository Contents

- **`suxotin.py`**: The main Python script implementing Suxotin's algorithm.
- **`sherlock_holmes.txt`**: Sample text used for testing.
- **Documentation Files**:
  - **`Decipherment_Models_1973.pdf`**: Original version of Suxotin's algorithm.
  - **`guySuxotin.pdf`**: An alternative version of the algorithm.
  - **`vowel-algorithm.txt`**: Text version of the algorithm for further exploration.
- **`README.md`**: This file, summarizing the algorithm and instructions.

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

   In a Python shell or script, download the required data:

   ```python
   import nltk
   nltk.download('europarl_raw')
   ```

## Usage

To run the algorithm:

1. **Execute the Script**

   ```bash
   python suxotin.py
   ```

2. **Select Languages**

   - Choose a specific language or select "0" to analyze all languages.

3. **Preprocess the Text**

   - When prompted, choosing 'yes' for preprocessing is recommended, as this removes non-alphabetic characters and standardizes input.

4. **View the Results**

   - The classified vowels and consonants are displayed alphabetically, alongside precision, recall, and F1 scores for each language.