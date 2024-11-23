Understood! Here's the updated README reflecting that there is only one Sukhotin's Algorithm, which is the modified one:

---

# Vowel Identification Algorithms

## Overview

This repository contains implementations of algorithms designed to identify vowels in texts by analyzing character adjacency patterns and co-occurrence statistics. By leveraging the statistical properties of letters in natural languages, these algorithms can classify letters as vowels or consonants, making them valuable tools for linguists, cryptanalysts, and language enthusiasts.

The repository includes:

1. **Sukhotin's Algorithm**: An implementation based on the work of B.V. Sukhotin, which iteratively classifies letters as vowels or consonants based on adjacency sums. This script has been modified to process the same dataset as Algorithm 1 and outputs its results to a separate file. It includes detailed in-line comments explaining the logic of the algorithm, making it easier to understand and modify.

2. **Spectral Decomposition Algorithm (Algorithm 1)**: An implementation of Algorithm 1 from the paper:

   **"Vowel and Consonant Classification through Spectral Decomposition"**  
   *Patricia Thaine and Gerald Penn*  
   Proceedings of the First Workshop on Subword and Character Level Models in NLP, pages 82–91, Copenhagen, Denmark, September 7, 2017.

Both algorithms operate independently of any specific language, making them applicable to a wide range of texts and scripts.

## Features

- **Language-Agnostic**: Process texts in any language and identify vowel patterns, regardless of orthography.
- **Unsupervised Learning**: No training data or labeled examples are required; the algorithms discover patterns inherent in the text data itself.
- **Efficient Computation**: Utilize NumPy for matrix operations and efficient algorithms optimized for performance.
- **Detailed Explanations**: Sukhotin's algorithm script includes comprehensive in-line comments, aiding in understanding the implementation.
- **File Output**: Both algorithms output results to their own files for easy analysis and record-keeping.
- **High Accuracy**: Demonstrated strong performance in correctly classifying vowels and consonants across multiple languages.
- **Alphabetical Output**: Results are sorted alphabetically for easy interpretation.

## Why the Algorithms Work

### Sukhotin's Algorithm

Sukhotin's algorithm relies on fundamental statistical properties observed in natural languages:

1. **Alternation of Vowels and Consonants**: Vowels and consonants frequently alternate in words, leading to characteristic adjacency patterns. This behavior helps the algorithm detect letters that function as vowels.

2. **Frequency of Vowels**: Vowels tend to appear frequently and in multiple combinations, creating higher adjacency counts, which the algorithm leverages for classification.

### Spectral Decomposition Algorithm (Algorithm 1)

The spectral decomposition algorithm is based on the observation that vowels and consonants exhibit different co-occurrence patterns in text:

1. **Character Contexts (P-Frames)**: By examining the immediate preceding and succeeding characters around each letter (p-frames), the algorithm captures the local context in which letters appear.

2. **Spectral Decomposition**: Applying Singular Value Decomposition (SVD) to the matrix constructed from these contexts reveals underlying structures in the data. The second right singular vector, in particular, tends to separate vowels and consonants due to their distinct distributional properties.

3. **Most Frequent Letter Assumption**: The algorithm assumes that the most frequent letter in the text is a vowel—a reasonable assumption for many languages (e.g., 'e' in English).

## Algorithm Workflows

### Sukhotin's Algorithm Workflow

#### 1. Reading the Dataset

- The script reads from the dataset, ensuring consistency in data processing with Algorithm 1.
- Language text files are read from the `Test/data/` directory, named according to their language codes.

#### 2. Preprocessing the Text

- Convert text to lowercase and remove non-alphabetic characters, standardizing the input for analysis.

#### 3. Creating the Frequency Matrix

- Create a symmetric matrix where each cell `(i, j)` represents the number of times character `i` is adjacent to character `j` in the text.
- Exclude self-adjacency (e.g., repeated characters like 'aa') to focus on interactions between different characters.

#### 4. Calculating Adjacency Sums

- Sum each character’s adjacency counts, highlighting characters with high connectivity to others, often vowels.

#### 5. Iterative Classification of Vowels and Consonants

- Iteratively classify characters with the highest remaining adjacency sums as vowels, adjusting the counts to minimize the influence of previously classified vowels.
- Adjustments involve subtracting twice the adjacency counts of the classified character from the remaining sums.

#### 6. Threshold-Based Reclassification

- Use an adjustable threshold to ensure that vowels with slightly lower adjacency sums are not misclassified.
- Reclassify consonants with adjusted sums above the threshold as vowels.

#### 7. Sorting the Results

- Sort classified vowels and consonants alphabetically for clarity.

#### 8. Outputting Results to File

- The script writes the classification results to its own output file (`suxotin_algorithm_output.txt`), including execution time and any errors encountered.

### Spectral Decomposition Algorithm Workflow (Algorithm 1)

#### 1. Preprocessing the Text

- Convert text to lowercase and remove non-alphabetic characters.
- Tokenize the text into words and letters for analysis.

#### 2. Constructing the P-Frame Matrix

- For each letter in the text, create a p-frame consisting of its immediate preceding and succeeding characters with the letter itself replaced by a placeholder (e.g., `(prev_char, '*', next_char)`).
- Build a binary matrix `A` where each row represents a unique p-frame, and each column represents a letter. An entry `A[i, j] = 1` indicates that letter `j` occurs in p-frame `i`.

#### 3. Singular Value Decomposition (SVD)

- Perform SVD on matrix `A` to obtain matrices `U`, `Σ`, and `V^T`.
- Extract the right singular vectors in `V`, focusing on the second vector `V[:, 1]`.

#### 4. Classifying Letters

- Split letters into two clusters based on the sign of their corresponding component in the second right singular vector.
- Assign labels: The cluster containing the most frequent letter is labeled as vowels; the other as consonants.

#### 5. Output Results

- Present the classified vowels and consonants in alphabetical order.
- Results are written to `algorithm1_output.txt`.

## Performance and Evaluation

### Sukhotin's Algorithm

The algorithm was tested across multiple languages with distinct vowel and consonant sets. Here are the results, including precision, recall, and F1 scores:

| Language    | True Positives (TP) | False Positives (FP) | False Negatives (FN) | Precision | Recall | F1 Score |
|-------------|---------------------|----------------------|----------------------|-----------|--------|----------|
| German      | 12                  | 1                    | 0                    | 0.923     | 1.000  | 0.960    |
| French      | 17                  | 1                    | 1                    | 0.944     | 0.944  | 0.944    |
| Spanish     | 11                  | 2                    | 0                    | 0.846     | 1.000  | 0.917    |
| Italian     | 12                  | 2                    | 4                    | 0.857     | 0.750  | 0.800    |
| Dutch       | 12                  | 0                    | 1                    | 1.000     | 0.923  | 0.960    |
| Greek       | 20                  | 3                    | 2                    | 0.870     | 0.909  | 0.889    |
| English     | 9                   | 0                    | 0                    | 1.000     | 1.000  | 1.000    |
| Swedish     | 12                  | 2                    | 0                    | 0.860     | 1.000  | 0.920    |
| Portuguese  | 15                  | 3                    | 3                    | 0.830     | 0.830  | 0.830    |
| Finnish     | 10                  | 2                    | 0                    | 0.830     | 1.000  | 0.910    |

#### Performance Summary

- **Strong Precision**: The algorithm achieved good precision across most languages, with English showing perfect precision (1.0). Most other languages maintained precision above 0.84.
- **Varied Recall**: Recall scores ranged from perfect (1.0) to lower values, demonstrating good but not perfect detection of vowel patterns, including accented variants.
- **F1 Scores**: Strong overall performance, with English achieving a perfect score and other languages showing solid results.

### Spectral Decomposition Algorithm (Algorithm 1)

The algorithm was tested using the same dataset as Sukhotin's algorithm and demonstrated high accuracy in classifying vowels and consonants across multiple languages.

#### Example Results

```python
Processing language: English (eng)
Vowels in English: a, e, i, o, u
Consonants in English: b, c, d, f, g, h, j, k, l, m, n, p, q, r, s, t, v, w, x, y, z
```

- **Accuracy**: The algorithm correctly identified all standard English vowels and classified consonants accurately.
- **Handling Ambiguous Letters**: The letter `'y'` was classified based on its predominant usage in the corpus.

## Repository Contents

- **`suxotin_algorithm.py`**: Python script implementing Sukhotin's algorithm, modified to read from the same dataset as Algorithm 1 and output results to its own file. Includes detailed in-line comments.
- **`algorithm1.py`**: Python script implementing Algorithm 1 from the paper.
- **`lang_code.csv`**: CSV file containing language codes and names used for processing.
- **`Test/data/`**: Directory containing language text files named according to their language codes.
- **Documentation Files**:
  - **`Decipherment_Models_1973.pdf`**: Original version of Sukhotin's algorithm.
  - **`guySuxotin.pdf`**: An alternative version of the algorithm.
  - **`vowel-algorithm.txt`**: Text version of the algorithm for further exploration.
- **`Vowel and Consonant Classification through Spectral Decomposition.pdf`**: Paper detailing Algorithm 1.
- **`README.md`**: This file, summarizing the algorithms and instructions.

## Installation

### Prerequisites

- **Python 3.6** or higher
- **Pip** (Python package installer)

### Steps

1. **Clone the Repository**

   Using HTTPS:

   ```bash
   git clone https://github.com/your_username/your_repository.git
   ```

   Or using GitHub CLI:

   ```bash
   gh repo clone your_username/your_repository
   ```

2. **Navigate to the Repository**

   ```bash
   cd your_repository
   ```

3. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
   ```

4. **Install the Required Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   **Note**: If `requirements.txt` is not provided, install the necessary packages manually:

   ```bash
   pip install numpy pandas
   ```

5. **Prepare the Data**

   - Place your language text files in the `Test/data/` directory.
   - Ensure that each file is named exactly as the language code specified in `lang_code.csv`, without any file extension.
   - Example:
     ```
     Test/data/eng
     Test/data/deu
     Test/data/fra
     ```

6. **Prepare the `lang_code.csv` File**

   - Ensure that `lang_code.csv` is in the same directory as the scripts.
   - The CSV should have at least two columns: `code` and `language`.

## Usage

### Running Sukhotin's Algorithm

1. **Execute the Script**

   ```bash
   python suxotin_algorithm.py
   ```

2. **View the Results**

   - Results will be saved to `suxotin_algorithm_output.txt`.
   - The file will contain the classified vowels and consonants for each language, including execution times and any errors.

### Running the Spectral Decomposition Algorithm (Algorithm 1)

1. **Execute the Script**

   ```bash
   python algorithm1.py
   ```

2. **View the Results**

   - Results will be saved to `algorithm1_output.txt`.
   - The file will contain the classified vowels and consonants for each language.

## Understanding the Code

### Sukhotin's Algorithm (`suxotin_algorithm.py`)

- **Reading the Dataset**: The script reads language text files from the `Test/data/` directory, ensuring it processes the same dataset as Algorithm 1.
- **Detailed In-Line Comments**: The script includes comprehensive comments explaining each step, making it easier to understand and modify.
- **Functions**:
  - `get_language_data`: Reads text data for a specified language code.
  - `preprocess_text`: Cleans the text by converting to lowercase and removing non-alphabetic characters.
  - `create_frequency_matrix`: Constructs a symmetric adjacency matrix of character pairs.
  - `classify_vowels`: Implements the core of Sukhotin's algorithm, classifying characters based on adjacency sums.
  - `run_analysis_for_language`: Orchestrates the processing for each language and collects results.
- **Output**: Results are written to `suxotin_algorithm_output.txt`, including execution times for performance insights.

### Spectral Decomposition Algorithm (`algorithm1.py`)

- **Functions**:
  - `vowel_consonant_classification`: Classifies letters based on the signs of the components in the second right singular vector obtained from SVD.
  - `algorithm1`: Prepares the corpus, constructs the p-frame matrix, performs SVD, and calls the classification function.
  - `run_algorithm_for_language`: Processes each language and writes results to an output file.
- **Key Libraries**:
  - **NumPy**: Used for numerical computations and performing SVD.
  - **Pandas**: For handling CSV data and data manipulation.
- **File Output**: Results are saved to `algorithm1_output.txt`.

## Extending the Algorithms

- **Different Corpora**: Modify the scripts to use other corpora or your own text data by adjusting the `test_folder` variable and ensuring your text files are properly formatted.
- **Language Adaptation**: Apply the algorithms to texts in other languages. Validate assumptions (e.g., most frequent letter being a vowel) for each language.
- **Threshold Adjustment** (Sukhotin's Algorithm): Experiment with different threshold values in the `classify_vowels` function to optimize performance for specific languages or datasets.
- **Detailed Analysis**: Use the execution times and logging to analyze performance bottlenecks or to compare algorithm efficiency across languages.

## References

- **Sukhotin's Algorithm**:
  - Sukhotin, B.V. (1962). *Eksperimental’noe vydelenie klassov bukv s pomoshch’ju elektronnoj vychislitel’noj mashiny*.
- **Spectral Decomposition Algorithm**:
  - Thaine, P., & Penn, G. (2017). *Vowel and Consonant Classification through Spectral Decomposition*. In Proceedings of the First Workshop on Subword and Character Level Models in NLP (pp. 82–91). Copenhagen, Denmark: Association for Computational Linguistics. [Link to Paper](https://aclanthology.org/W17-4109/)

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **B.V. Sukhotin**: For the foundational vowel identification algorithm.
- **Patricia Thaine and Gerald Penn**: For their work on vowel and consonant classification through spectral decomposition.
- **NLTK Team**: For providing a comprehensive toolkit for natural language processing in Python.
- **OpenAI**: For offering guidance on best practices in AI and code generation.

---

*This README was updated to include a modified implementation of Sukhotin's algorithm that reads from the same dataset as Algorithm 1 and outputs its results to a separate file. The script includes detailed in-line comments explaining the logic of the algorithm, enhancing the project's educational value and usability.*