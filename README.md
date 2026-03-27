# Vowel Identification Algorithms

Unsupervised, language-agnostic classification of letters as vowels or consonants from raw text. No linguistic priors — purely statistical. Tested on 502 languages.

## Algorithms

| Script | Method | F1 |
|---|---|---|
| `classify.py` | Ensemble of SVD + Sukhotin + accent propagation | **0.965** |
| `algorithm1.py` | SVD spectral decomposition ([Thaine & Penn 2017](https://aclanthology.org/W17-4109/)) | 0.942 |
| `suxotin.py` | Sukhotin's adjacency algorithm (1962) | 0.919 |

**`classify.py`** is the recommended entry point. It uses SVD as the primary classifier, Sukhotin to detect and correct SVD label flips, and Unicode decomposition to propagate vowel status to accented variants.

## Results

Across 502 languages, the combined classifier:
- Correctly identifies all 5 basic vowels in **85%** of languages
- Corrects **20** SVD label-flip errors using Sukhotin's orientation
- Adds accent-propagated vowels in **140** languages
- Filters out Suxotin false positives in **270** languages
- Achieves perfect F1 on English, German, Spanish, Finnish, Portuguese, Czech, Polish, Indonesian, Latin, Estonian, and 10+ others

## Usage

```bash
pip install -r requirements.txt

python classify.py            # recommended — writes classification_output.txt
python algorithm1.py          # SVD only — writes algorithm1_output.txt
python suxotin.py             # Sukhotin only — writes suxotin_algorithm_output.txt
```

Evaluate against ground truth (32 languages):

```bash
python -c "from classify import run; from evaluate import evaluate; evaluate(run)"
```

## How It Works

**SVD** builds a binary matrix of letters vs. trigram contexts (p-frames), then uses the second right singular vector to split letters into two clusters. The cluster with higher mean frequency is labeled vowels.

**Sukhotin's** builds a character adjacency matrix from within-word bigrams (excluding whitespace — critical for avoiding false positives). It iteratively selects the highest-sum character as a vowel and subtracts its adjacency contributions until no positive sums remain.

**Ensemble** runs both, uses Sukhotin to correct SVD when the two disagree on cluster orientation, then propagates vowel status to accented variants via Unicode NFKD decomposition.

## Repository Structure

```
classify.py          Combined classifier (recommended)
algorithm1.py        SVD spectral decomposition
suxotin.py           Sukhotin's adjacency algorithm
evaluate.py          Evaluation harness (32 languages with ground truth)
utils.py             Shared I/O and preprocessing
lang_code.csv        Language code → name mapping
Test/data/           Text corpus per language (502 files)
visualizations/      Per-language classification score charts
Literature/          Reference papers
```

## References

- Sukhotin, B.V. (1962). *Eksperimental'noe vydelenie klassov bukv s pomoshch'ju elektronnoj vychislitel'noj mashiny*.
- Thaine, P. & Penn, G. (2017). [Vowel and Consonant Classification through Spectral Decomposition](https://aclanthology.org/W17-4109/). Workshop on Subword and Character Level Models in NLP, pp. 82–91.

## License

GPL-3.0 — see [LICENSE](LICENSE).
