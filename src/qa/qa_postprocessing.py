"""
Post-process raw QA dataset outputs into cleaned, individual QA pairs.

Takes the raw approved pairs saved by app.py (list of {"input": ..., "output": ...})
and splits each output blob into individual {"question": ..., "answer": ...} dicts
with cleaned text. Mirrors the processing in legacy/Dataset.ipynb (Cells 24-30)
but without requiring any external metadata.

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


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def split_qas_markdown(text: str) -> List[Dict[str, str]]:
    """
    Extract individual QA pairs from raw LLM markdown output.

    Matches patterns like **Q6:** ... **A6:** ... supporting optional bold
    markers, various separators (: . ) -), and multi-line content.

    Returns:
        List of {"question": str, "answer": str} dicts.
    """
    pattern = re.compile(
        r"""
        \*{0,2}\s*Q(?P<num>\d+)\s*[:.)\-]?\s*\*{0,2}\s*   # Qn header
        (?P<q>.*?)                                          # question body
        \b\*{0,2}\s*A(?P=num)\s*[:.)\-]?\s*\*{0,2}\s*      # An with same n
        (?P<a>.*?)                                          # answer body
        (?=\s*\*{0,2}\s*Q\d+\s*[:.)\-]?\s*\*{0,2}|$)       # next Q or end
        """,
        re.DOTALL | re.VERBOSE | re.IGNORECASE,
    )

    pairs = []
    for m in pattern.finditer(text):
        q = _clean_block(m.group("q"))
        a = _clean_block(m.group("a"))
        if q and a:
            pairs.append({"question": q, "answer": a})
    return pairs


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def _clean_block(text: str) -> str:
    """Remove markdown dividers and normalise whitespace."""
    text = text.strip()
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
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

    # Remove leaked leading Q-labels like "Q6" or "Q6:"
    text = re.sub(r"^Q\d+\s*[:.)\-]?\s*", "", text, flags=re.IGNORECASE)

    tokens = text.split(" ")
    # Trim leading tokens that contain no letters
    while tokens and not LETTERS.intersection(tokens[0]):
        tokens = tokens[1:]
    # Trim trailing tokens that contain no letters
    while tokens and not LETTERS.intersection(tokens[-1]):
        tokens = tokens[:-1]

    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Main post-processing
# ---------------------------------------------------------------------------

def postprocess_raw_dataset(
    raw_pairs: List[Dict[str, str]],
    num_few_shot: int = 5,
) -> List[Dict[str, str]]:
    """
    Post-process raw approved pairs from app.py into cleaned individual QA pairs.

    Args:
        raw_pairs: List of {"input": str, "output": str} from the app.
        num_few_shot: Number of few-shot examples in the prompt. Parsed QA
            pairs whose index falls within the few-shot range are dropped
            (they are echoed examples, not generated content).

    Returns:
        Flat list of {"question": str, "answer": str} dicts.
    """
    dataset = []
    for pair in raw_pairs:
        output = pair.get("output", "")
        parsed = split_qas_markdown(output)

        # If the LLM echoed few-shot examples back, they appear first.
        # The few-shot Qs are numbered 1..num_few_shot; generated ones start
        # at num_few_shot+1. We can detect this by checking if we got more
        # pairs than expected (i.e. > batch size) or if early pair numbers
        # are low. Safest heuristic: drop any extras beyond the expected
        # generated count. In the default pipeline each prompt generates ~5
        # pairs, so anything beyond that is likely echoed few-shot.
        if len(parsed) > num_few_shot:
            parsed = parsed[len(parsed) - num_few_shot:]

        for p in parsed:
            q = _clean_qa_text(p["question"])
            a = _clean_qa_text(p["answer"])
            if q and a:
                dataset.append({"question": q, "answer": a})

    return dataset


def postprocess_file(
    input_path: str | Path,
    output_path: str | Path | None = None,
    num_few_shot: int = 5,
) -> List[Dict[str, str]]:
    """
    Read a raw dataset JSON file, post-process it, and write the result.

    Args:
        input_path: Path to the raw JSON (list of {"input", "output"}).
        output_path: Where to write the cleaned JSON. Defaults to
            ``<input_stem>_cleaned.json`` in the same directory.
        num_few_shot: Passed through to ``postprocess_raw_dataset``.

    Returns:
        The cleaned dataset.
    """
    input_path = Path(input_path)
    with open(input_path, "r", encoding="utf-8") as f:
        raw_pairs = json.load(f)

    dataset = postprocess_raw_dataset(raw_pairs, num_few_shot=num_few_shot)

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
