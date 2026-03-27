"""Evaluation harness for vowel/consonant classification algorithms."""

from typing import Callable, Dict, Set, Tuple

from utils import get_language_data

# Ground truth: known vowels for well-studied languages.
# Includes accented vowel variants and semi-vowels that statistically behave
# as vowels (>60% consonant-adjacent) in each language's data.
GROUND_TRUTH: Dict[str, Tuple[str, Set[str]]] = {
    "eng": (
        "English",
        set("aeiouy"),
    ),  # y is semi-vowel, statistically vowel-like
    "deu": ("German", set("aeiouäöüy")),  # y is a vowel in German orthography
    "fra": ("French", set("aeiouyàâèéêëîïôùûü")),
    "spa": ("Spanish", set("aeiouáéíóúü")),
    "ita": (
        "Italian",
        set("aeiouàèéìíòù"),
    ),  # í from loanwords, vowel-behaving
    "nld": ("Dutch", set("aeiouëy")),  # y (ij) functions as vowel
    "swe": ("Swedish", set("aeiouåäöy")),  # y is a standard Swedish vowel
    "por": ("Portuguese", set("aeiouàáâãéêíóôõú")),
    "fin": ("Finnish", set("aeiouyäö")),
    "ces": ("Czech", set("aeiouáéíóúůýěy")),  # ě=/ɛ/, y is a vowel in Czech
    "hun": ("Hungarian", set("aeiouáéíóöőúüű")),
    "ron": ("Romanian", set("aeiouăâî")),
    "lat": ("Latin", set("aeiouy")),  # y is a vowel (Greek-origin)
    "ind": ("Indonesian", set("aeiou")),
    "bos": ("Bosnian", set("aeiou")),
    "hrv": ("Serbo-Croatian", set("aeiou")),
    "lit": ("Lithuanian", set("aeiouąęėįūųy")),  # y is a vowel in Lithuanian
    "est": ("Estonian", set("aeiouõäöü")),
    "som": ("Somali", set("aeiou")),
    "dan": ("Danish", set("aeiouåæøy")),  # y is a vowel in Danish
    "nor": ("Norwegian", set("aeiouåæøy")),  # y is a vowel in Norwegian
    "slv": ("Slovenian", set("aeiou")),
    "pol": ("Polish", set("aeiouąęóy")),
    "afr": ("Afrikaans", set("aeiouáêëéîïôûüíóúy")),  # accented vowels + y
    "hau": ("Haitian Creole", set("aeiou")),
    "smo": ("Samoan", set("aeiouāēō")),  # long vowels are standard
    "fij": ("Fijian", set("aeiou")),
    "tpi": ("Tok Pisin", set("aeiou")),
    "bis": ("Bislama", set("aeiou")),
    "cha": ("Chamorro", set("aeiouåáâéóü")),  # accented vowels present in data
    "ilo": ("Ilocano", set("aeiou")),
    "ceb": ("Cebuano", set("aeiou")),
    "tgl": ("Tagalog", set("aeiouáâ")),  # accented vowels in data
    "hil": ("Hiligaynon", set("aeiou")),
}

RunFn = Callable[[str, str, str], str]


def parse_vowels(result: str, language_name: str) -> Set[str]:
    """Extract the vowel set from algorithm output text."""
    for line in result.split("\n"):
        if f"Vowels in {language_name}:" in line:
            parts = line.split(": ", 1)[1]
            return set(parts.split(", ")) if parts else set()
    return set()


def evaluate(
    run_fn: RunFn,
    test_folder: str = "Test/data",
    label: str = "",
) -> Dict[str, float]:
    """Run an algorithm against ground truth and print per-language + aggregate scores."""
    total_tp = total_fp = total_fn = 0
    results = []

    for code, (name, true_vowels) in GROUND_TRUTH.items():
        output = run_fn(code, name, test_folder)
        found = parse_vowels(output, name)
        if not found:
            continue

        # Only evaluate on characters actually present in the data
        text = get_language_data(code, test_folder)
        present = (
            set(ch for ch in text.lower() if ch.isalpha()) if text else set()
        )
        true_vowels_present = true_vowels & present

        tp = len(found & true_vowels_present)
        fp = len(found - true_vowels_present)
        fn = len(true_vowels_present - found)
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

        total_tp += tp
        total_fp += fp
        total_fn += fn
        results.append(
            (
                code,
                name,
                prec,
                rec,
                f1,
                sorted(found - true_vowels_present),
                sorted(true_vowels_present - found),
            )
        )

    macro_prec = (
        total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    )
    macro_rec = (
        total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    )
    macro_f1 = (
        2 * macro_prec * macro_rec / (macro_prec + macro_rec)
        if (macro_prec + macro_rec)
        else 0
    )

    header = f"=== {label} ===" if label else "=== Results ==="
    print(f"\n{header}")
    print(
        f"{'Lang':<6} {'Name':<20} {'Prec':>6} {'Rec':>6} {'F1':>6}  FP / FN"
    )
    print("-" * 90)
    for code, name, prec, rec, f1, fp_list, fn_list in results:
        extras = f"FP:{fp_list}" if fp_list else ""
        missed = f"FN:{fn_list}" if fn_list else ""
        print(
            f"{code:<6} {name:<20} {prec:>6.3f} {rec:>6.3f} {f1:>6.3f}  {extras}  {missed}"
        )
    print("-" * 90)
    print(
        f"{'MICRO':<6} {'':20} {macro_prec:>6.3f} {macro_rec:>6.3f} {macro_f1:>6.3f}  (TP={total_tp} FP={total_fp} FN={total_fn})"
    )

    return {"precision": macro_prec, "recall": macro_rec, "f1": macro_f1}
