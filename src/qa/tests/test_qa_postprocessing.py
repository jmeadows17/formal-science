import sys
from pathlib import Path


QA_DIR = Path(__file__).resolve().parents[1]
if str(QA_DIR) not in sys.path:
    sys.path.insert(0, str(QA_DIR))

from qa_postprocessing import postprocess_raw_dataset, split_qas_markdown


def test_split_qas_markdown_parses_markdown_pairs_with_bold_headers():
    output = """
**Q6:** First question text.

**A6:** First answer text.

**Q7:** Second question text.

**A7:** Second answer text.
"""

    assert split_qas_markdown(output) == [
        {"n": 6, "question": "First question text.", "answer": "First answer text."},
        {"n": 7, "question": "Second question text.", "answer": "Second answer text."},
    ]


def test_postprocess_keeps_requested_trailing_batch_when_model_echoes_examples():
    raw_pairs = [{
        "input": "Please generate Question (Q6-Q7) and Answer (A6-A7).",
        "output": """
**Q1:** Echoed example.
**A1:** Echoed answer.

**Q6:** Keep this question.
**A6:** Keep this answer.

**Q7:** And this one too.
**A7:** And this answer too.
""",
    }]

    assert postprocess_raw_dataset(raw_pairs) == [[
        {"question": "Keep this question.", "answer": "Keep this answer."},
        {"question": "And this one too.", "answer": "And this answer too."},
    ]]


def test_postprocess_deduplicates_identical_pairs_within_a_batch():
    raw_pairs = [{
        "input": "Generate Question (Q6-Q7) and Answer (A6-A7).",
        "output": """
Q6: Duplicate question.
A6: Duplicate answer.

Q7: Duplicate question.
A7: Duplicate answer.
""",
    }]

    assert postprocess_raw_dataset(raw_pairs) == [[
        {"question": "Duplicate question.", "answer": "Duplicate answer."},
    ]]


def test_postprocess_strips_wrapper_braces_and_stray_markdown_markers():
    raw_pairs = [{
        "input": "Generate Question (Q6-Q7) and Answer (A6-A7).",
        "output": """
**Q6:** In the planar arc-length functional, compute the derivative.

{
**A6:** }
Define the quantity and differentiate carefully.

**Q7:** Clean control example.
**A7:** Clean control answer.
""",
    }]

    assert postprocess_raw_dataset(raw_pairs) == [[
        {
            "question": "In the planar arc-length functional, compute the derivative.",
            "answer": "Define the quantity and differentiate carefully.",
        },
        {"question": "Clean control example.", "answer": "Clean control answer."},
    ]]


def test_postprocess_rejects_pair_with_unbalanced_display_math_in_question():
    raw_pairs = [{
        "input": "Generate Question (Q6-Q6) and Answer (A6-A6).",
        "output": r"""
Q6: Show that
\[
\nabla^2 \phi
A6: Because the Laplacian vanishes, the result follows.
""",
    }]

    assert postprocess_raw_dataset(raw_pairs) == []


def test_postprocess_rejects_pair_with_unbalanced_inline_math_in_question():
    raw_pairs = [{
        "input": "Generate Question (Q6-Q6) and Answer (A6-A6).",
        "output": r"""
Q6: Show that the energy density is defined for \(E \ge
A6: The intended domain is nonnegative energy.
""",
    }]

    assert postprocess_raw_dataset(raw_pairs) == []


def test_postprocess_rejects_pair_with_unbalanced_display_math_in_answer():
    raw_pairs = [{
        "input": "Generate Question (Q6-Q6) and Answer (A6-A6).",
        "output": r"""
Q6: Show that the field is divergence free.
A6: We compute
\[
\nabla \cdot \mathbf{B}
""",
    }]

    assert postprocess_raw_dataset(raw_pairs) == []


def test_postprocess_rejects_pair_with_unbalanced_dollar_math_in_question():
    raw_pairs = [{
        "input": "Generate Question (Q6-Q6) and Answer (A6-A6).",
        "output": """
Q6: Show that the radial derivative satisfies $\\frac{\\partial \\phi}{\\partial r}
A6: The derivative vanishes in the region.
""",
    }]

    assert postprocess_raw_dataset(raw_pairs) == []


def test_postprocess_rejects_answer_with_chatty_assistant_meta_text():
    raw_pairs = [{
        "input": "Generate Question (Q6-Q6) and Answer (A6-A6).",
        "output": """
Q6: Show the spherical surface integral form.
A6: Apply the divergence theorem and obtain the result.

If you want, I can also rewrite these in exactly the same LaTeX style as your notes.
""",
    }]

    assert postprocess_raw_dataset(raw_pairs) == []


def test_postprocess_removes_dangling_trailing_bold_line():
    raw_pairs = [{
        "input": "Generate Question (Q6-Q6) and Answer (A6-A6).",
        "output": """
Q6: Example with dangling bold line.

**
A6: Valid answer.
""",
    }]

    assert postprocess_raw_dataset(raw_pairs) == [[
        {"question": "Example with dangling bold line.", "answer": "Valid answer."},
    ]]


def test_postprocess_keeps_valid_pairs_when_one_pair_in_batch_is_invalid():
    raw_pairs = [{
        "input": "Generate Question (Q6-Q7) and Answer (A6-A7).",
        "output": r"""
Q6: Show that the energy density is defined for \(E \ge
A6: This pair is malformed and should be dropped.

Q7: Valid question.
A7: Valid answer.
""",
    }]

    assert postprocess_raw_dataset(raw_pairs) == [[
        {"question": "Valid question.", "answer": "Valid answer."},
    ]]


def test_postprocess_discards_mismatched_question_answer_numbering():
    raw_pairs = [{
        "input": "Generate Question (Q6-Q6) and Answer (A6-A6).",
        "output": """
Q6: This question never receives A6.
A7: Wrong answer header.
""",
    }]

    assert postprocess_raw_dataset(raw_pairs) == []
