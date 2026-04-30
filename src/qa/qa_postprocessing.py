"""
Post-process raw QA dataset outputs into cleaned, batched QA pairs.

Takes the raw approved pairs saved by app.py (list of {"input": ..., "output": ...})
and splits each output blob into a batch of {"question": ..., "answer": ...} dicts
with cleaned text. Mirrors the processing in legacy/Dataset.ipynb (Cells 24-30)
while preserving per-prompt batches and without requiring external metadata.

Can be used as a library or run directly:
    python src/qa/qa_postprocessing.py src/app_data/qa_dataset_xyz.json
"""

import json
import re
import string
import sys
from pathlib import Path
from typing import List, Dict

LETTERS = set(string.ascii_letters)
CHAT_META_RE = re.compile(
    r"If you want, I can|Let me know if you want|Would you like me to",
    re.IGNORECASE,
)
JUNK_LINE_RE = re.compile(r"^[\s{}*_`]+$")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def split_qas_markdown(text: str) -> List[Dict[str, str]]:
    """
    Extract individual QA pairs from raw LLM markdown output.

    Matches patterns like **Q6:** ... **A6:** ... supporting optional bold
    markers, various separators (: . ) -), and multi-line content.

    Returns:
        List of {"n": int, "question": str, "answer": str} dicts.
    """
    pattern = re.compile(
        r"""
        (?:^|\n)\s*\*{0,2}\s*Q(?P<num>\d+)\s*[:.)\-]?\s*\*{0,2}\s*  # Qn header
        (?P<q>.*?)                                                     # question body
        (?:^|\n)\s*\*{0,2}\s*A(?P=num)\s*[:.)\-]?\s*\*{0,2}\s*        # An with same n
        (?P<a>.*?)                                                     # answer body
        (?=(?:^|\n)\s*\*{0,2}\s*Q\d+\s*[:.)\-]?\s*\*{0,2}|\Z)         # next Q or end
        """,
        re.DOTALL | re.VERBOSE | re.IGNORECASE | re.MULTILINE,
    )

    pairs = []
    for m in pattern.finditer(text):
        q = _clean_block(m.group("q"))
        a = _clean_block(m.group("a"))
        if q and a:
            pairs.append({"n": int(m.group("num")), "question": q, "answer": a})
    return pairs


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def _clean_block(text: str) -> str:
    """Remove markdown dividers and normalise whitespace."""
    text = text.strip()
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\*{1,2}\s*", "", text)
    text = re.sub(r"\s*\*{1,2}$", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _clean_qa_text(text: str) -> str:
    """
    Clean a question or answer string:
    - Strip \\textbf markers
    - Remove leaked Q-labels (e.g. leading "Q6")
    - Trim non-alphabetic tokens from start/end
    """
    text = text.replace("\\textbf", "")

    # Remove wrapper-only lines leaked from object-like or markdown output.
    lines = text.splitlines()
    while lines and JUNK_LINE_RE.fullmatch(lines[0] or ""):
        lines = lines[1:]
    while lines and JUNK_LINE_RE.fullmatch(lines[-1] or ""):
        lines = lines[:-1]
    text = "\n".join(lines).strip()

    # Remove leaked leading QA labels like "Q6", "Q6:", "A6", "A6:".
    text = re.sub(r"^[QA]\d+\s*[:.)\-]?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(?:[{}*_`]+\s*)+", "", text)
    text = re.sub(r"(?:\s*[{}*_`]+)+$", "", text)

    tokens = text.split(" ")
    # Trim leading tokens that contain no letters
    while tokens and not LETTERS.intersection(tokens[0]):
        tokens = tokens[1:]
    # Trim trailing tokens that contain no letters
    while tokens and not LETTERS.intersection(tokens[-1]):
        tokens = tokens[:-1]

    return " ".join(tokens)


def _normalise_pair_key(pair: Dict[str, str]) -> tuple[str, str]:
    """Build a stable key for deduplicating QA pairs while preserving order."""
    return (
        re.sub(r"\s+", " ", pair["question"]).strip(),
        re.sub(r"\s+", " ", pair["answer"]).strip(),
    )


def _has_balanced_latex_delimiters(text: str) -> bool:
    """Check basic balance for common LaTeX math delimiters."""
    unescaped_text = re.sub(r"\\\$", "", text)
    display_dollars = unescaped_text.count("$$")
    inline_dollars = unescaped_text.count("$") - (2 * display_dollars)
    return (
        text.count(r"\[") == text.count(r"\]")
        and text.count(r"\(") == text.count(r"\)")
        and display_dollars % 2 == 0
        and inline_dollars % 2 == 0
    )


def _is_valid_pair(pair: Dict[str, str]) -> bool:
    """Reject obviously truncated or assistant-meta QA pairs."""
    question = pair["question"].strip()
    answer = pair["answer"].strip()
    if not question or not answer:
        return False
    if CHAT_META_RE.search(question) or CHAT_META_RE.search(answer):
        return False
    if not _has_balanced_latex_delimiters(question):
        return False
    if not _has_balanced_latex_delimiters(answer):
        return False
    return True


def _unique_pairs(parsed: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Drop duplicate QA pairs while preserving first occurrence order."""
    unique = []
    seen = set()
    for pair in parsed:
        key = _normalise_pair_key(pair)
        if key in seen:
            continue
        seen.add(key)
        unique.append(pair)
    return unique


def _infer_batch_size(input_text: str, parsed: List[Dict[str, str]]) -> int:
    """
    Infer the generated batch size dynamically.

    Prefer the explicit target QA range in the prompt, e.g. ``Q6-Q10``.
    If that is unavailable, fall back to the number of unique parsed QA pairs.
    """
    range_match = re.search(r"Q(\d+)\s*-\s*Q(\d+)", input_text or "", flags=re.IGNORECASE)
    if range_match:
        start = int(range_match.group(1))
        end = int(range_match.group(2))
        if end >= start:
            return end - start + 1

    return len(_unique_pairs(parsed))


# ---------------------------------------------------------------------------
# Main post-processing
# ---------------------------------------------------------------------------

def postprocess_raw_dataset(
    raw_pairs: List[Dict[str, str]],
) -> List[List[Dict[str, str]]]:
    """
    Post-process raw approved pairs from app.py into cleaned QA batches.

    Args:
        raw_pairs: List of {"input": str, "output": str} from the app.

    Returns:
        List of batches, one per raw model output. Each batch is a list of
        {"question": str, "answer": str} dicts.
    """
    dataset = []
    for pair in raw_pairs:
        input_text = pair.get("input", "")
        output = pair.get("output", "")
        parsed = split_qas_markdown(output)
        batch_size = _infer_batch_size(input_text, parsed)
        unique_parsed = _unique_pairs(parsed)

        # Keep the trailing block when the model echoes earlier examples before
        # producing the requested batch.
        if batch_size and len(unique_parsed) > batch_size:
            unique_parsed = unique_parsed[-batch_size:]

        batch = []
        for p in unique_parsed:
            q = _clean_qa_text(p["question"])
            a = _clean_qa_text(p["answer"])
            cleaned_pair = {"question": q, "answer": a}
            if _is_valid_pair(cleaned_pair):
                batch.append(cleaned_pair)

        if batch:
            dataset.append(batch)

    return dataset


def postprocess_file(
    input_path: str | Path,
    output_path: str | Path | None = None,
) -> List[List[Dict[str, str]]]:
    """
    Read a raw dataset JSON file, post-process it, and write the result.

    Args:
        input_path: Path to the raw JSON (list of {"input", "output"}).
        output_path: Where to write the cleaned JSON. Defaults to
            ``<input_stem>_cleaned.json`` in the same directory.
    Returns:
        The cleaned dataset.
    """
    input_path = Path(input_path)
    with open(input_path, "r", encoding="utf-8") as f:
        raw_pairs = json.load(f)

    dataset = postprocess_raw_dataset(raw_pairs)

    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_cleaned.json")
    output_path = Path(output_path)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    return dataset


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <input.json> [output.json]")
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    result = postprocess_file(in_path, out_path)
    print(f"Wrote {len(result)} cleaned QA pairs.")
