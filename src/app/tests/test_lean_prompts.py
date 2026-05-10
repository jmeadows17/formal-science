import sys
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import lean_prompts


def test_build_lean_prompt_includes_rubric_path_and_qa_json():
    prompt = lean_prompts.build_lean_prompt(
        {"question": "What is Q1?", "answer": "A1."},
        qa_index=2,
        batch_size=3,
    )

    assert "Write Lean 4 + Mathlib code to `FSLean/proof.lean`" in prompt
    assert "`src/app/autoformalisation_rubric.md`" in prompt
    assert '"question": "What is Q1?"' in prompt
    assert '"answer": "A1."' in prompt
    assert "This prompt covers exactly one QA pair." in prompt
    assert "must be named `C2`" not in prompt
    assert "original batch of size 3" not in prompt
    assert "Produce exactly one aligned target theorem" in prompt


def test_build_lean_prompt_does_not_use_old_sectioned_template():
    prompt = lean_prompts.build_lean_prompt({"question": "Q", "answer": "A"})

    assert "# Requirements" not in prompt
    assert "# Deliverables" not in prompt
    assert "# Acceptance criteria" not in prompt
    assert "**Q1:**" not in prompt


def test_build_lean_prompt_dataset_flattens_batches_to_one_prompt_per_pair():
    prompts = lean_prompts.build_lean_prompt_dataset(
        [
            [
                {"question": "Q1", "answer": "A1"},
                {"question": "Q2", "answer": "A2"},
            ],
            [
                {"question": "Q3", "answer": "A3"},
            ],
        ]
    )

    assert len(prompts) == 3
    assert all("This prompt covers exactly one QA pair." in prompt for prompt in prompts)
    assert all("must be named `C" not in prompt for prompt in prompts)
