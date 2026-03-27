# Vowel Identification Algorithms

Language-agnostic algorithms that classify letters as vowels or consonants from raw text, using only statistical properties — no linguistic knowledge required. Tested across 500+ languages.

## Algorithms

### Combined Classifier (`classify.py`) — Recommended

Ensembles both approaches below with Unicode accent propagation. Achieves **F1 = 0.965** (micro-averaged across 32 languages with known ground truth).

1. **SVD** provides high-precision primary classification.
2. **Sukhotin's** confirms borderline cases and filters false positives.
3. **Accent propagation** catches diacritical vowel variants (e.g. if 'a' is a vowel, so are 'á', 'à', 'â', 'ã', 'ä', 'å').

### Spectral Decomposition (`algorithm1.py`)

Based on [Thaine & Penn (2017)](https://aclanthology.org/W17-4109/). F1 = 0.942.

1. Captures each letter's trigram context as a p-frame `(prev, *, next)`.
2. Builds a binary letter-by-p-frame matrix and performs SVD.
3. Clusters letters by the sign of the second right singular vector.

### Sukhotin's Algorithm (`suxotin.py`)

Based on Sukhotin (1962). F1 = 0.919.

Vowels and consonants alternate in natural language, giving vowels high adjacency counts. The algorithm:

1. Builds a symmetric character adjacency matrix from **within-word bigrams only** (excluding whitespace — a key improvement over naive implementations).
2. Iteratively selects the character with the highest adjacency sum as a vowel, subtracting its contributions from remaining sums.
3. Stops when no character has a positive remaining sum.

## Performance

Micro-averaged scores across 32 well-studied languages (only characters present in the test data are evaluated):

| Algorithm | Precision | Recall | F1 |
|---|---|---|---|
| Combined (`classify.py`) | **0.939** | **0.992** | **0.965** |
| SVD (`algorithm1.py`) | 0.936 | 0.947 | 0.942 |
| Sukhotin (`suxotin.py`) | 0.893 | 0.947 | 0.919 |

Perfect scores (F1 = 1.0) on: English, German, Spanish, Finnish, Dutch, Romanian, Latin, Indonesian, Bosnian, Lithuanian, Estonian, Samoan, Fijian, Tok Pisin, Bislama, Ilocano, Cebuano, Tagalog, Hiligaynon, and others.

## Repository Structure

| Path | Description |
|---|---|
| `classify.py` | Combined classifier (recommended) |
| `algorithm1.py` | SVD spectral decomposition |
| `suxotin.py` | Sukhotin's adjacency algorithm |
| `evaluate.py` | Evaluation harness with ground truth for 32+ languages |
| `utils.py` | Shared utilities (file I/O, language loading, preprocessing) |
| `lang_code.csv` | Language code-to-name mapping |
| `Test/data/` | Text files per language (named by language code, no extension) |
| `visualizations/` | Generated classification score bar charts |
| `Literature/` | Reference PDFs |

## Installation

Requires Python 3.8+.

```bash
git clone git@github.com:jhnwnstd/suxotin.git
cd suxotin
pip install -r requirements.txt
```

## Usage

```bash
# Recommended: combined classifier
python classify.py

# Individual algorithms
python algorithm1.py
python suxotin.py

# Run evaluation
python -c "from classify import run; from evaluate import evaluate; evaluate(run, label='Combined')"
```

## Example Output

```
Processing language: English (eng)
Vowels in English: a, e, i, o, u
Consonants in English: b, c, d, f, g, h, j, k, l, m, n, p, q, r, s, t, v, w, x, y, z

Processing language: Finnish (fin)
Vowels in Finnish: a, e, i, o, u, y, ä, ö
Consonants in Finnish: b, d, f, g, h, j, k, l, m, n, p, r, s, t, v
```

## Key Findings

- **Spaces in adjacency matrix distort Sukhotin's algorithm.** Word-boundary consonants ('s', 'p', 't') get inflated adjacency sums from whitespace, causing systematic false positives. Using within-word bigrams only eliminates this.
- **'y' is statistically a vowel in most languages.** Adjacency analysis shows 'y' has 67–100% consonant neighbors across tested languages, consistent with vowel alternation behavior.
- **Accent propagation is critical for recall.** Many test texts contain accented vowel variants (é, ö, ü) that SVD alone misclassifies due to low frequency.
- **SVD and Sukhotin are complementary.** SVD has higher precision; Sukhotin has fewer blind spots on high-frequency characters. Ensembling reduces errors by 40% vs SVD alone.

## References

- Sukhotin, B.V. (1962). *Eksperimental'noe vydelenie klassov bukv s pomoshch'ju elektronnoj vychislitel'noj mashiny*.
- Thaine, P., & Penn, G. (2017). [Vowel and Consonant Classification through Spectral Decomposition](https://aclanthology.org/W17-4109/). In Proceedings of the First Workshop on Subword and Character Level Models in NLP (pp. 82–91).

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
