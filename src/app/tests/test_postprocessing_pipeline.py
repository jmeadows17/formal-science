import json
import sys
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from compile_lean import compile_lean
from postprocessing_pipeline import (
    build_boundary_prompt,
    build_extraction_prompt,
    build_fragment_assembled_answer,
    detect_target_theorems,
    evaluate_boundary_response,
    evaluate_extraction_response,
    reconcile_audit_with_formal_qa,
    parse_boundary_response,
    parse_extraction_response,
    referenced_helper_blocks,
    shared_preamble_line_count,
    source_line_count,
    validate_extracted_formal_answer,
    validate_llm_extracted_formal_answer,
    write_formal_qa_data_from_audit,
)


def _ok_compile(_code: str):
    return 0, "", ""


def test_parse_boundary_response_accepts_fenced_json():
    response = """
Here is the alignment.

```json
{"items":[{"qa_index":1,"target_label":"C1","target_theorem_name":"C1","start_line":1,"end_line":4}]}
```
"""

    assert parse_boundary_response(response) == [
        {
            "qa_index": 1,
            "target_label": "C1",
            "target_theorem_name": "C1",
            "start_line": 1,
            "end_line": 4,
            "confidence": "",
            "justification": "",
        }
    ]


def test_parse_extraction_response_accepts_fenced_json():
    response = """
```json
{"items":[{"qa_index":1,"target_label":"C1","target_theorem_name":"C1","formal_answer":"theorem C1 : True := by\\n  trivial\\n"}]}
```
"""

    assert parse_extraction_response(response) == [
        {
            "qa_index": 1,
            "target_label": "C1",
            "target_theorem_name": "C1",
            "formal_answer": "theorem C1 : True := by\n  trivial\n",
            "confidence": "",
            "justification": "",
        }
    ]


def test_build_fragment_assembled_answer_prepends_shared_preamble_without_prior_theorems():
    formal_proofs = (
        "import Mathlib.Tactic\n"
        "\n"
        "theorem C1 : 1 = 1 := by\n"
        "  rfl\n"
        "\n"
        "theorem C2 : 2 = 2 := by\n"
        "  rfl\n"
    )

    assembled, fragments = build_fragment_assembled_answer(formal_proofs, 6, 7)

    assert shared_preamble_line_count(formal_proofs) == 2
    assert assembled == "import Mathlib.Tactic\n\ntheorem C2 : 2 = 2 := by\n  rfl\n"
    assert fragments == [
        {"kind": "preamble", "start_line": 1, "end_line": 2},
        {"kind": "body", "start_line": 6, "end_line": 7},
    ]


def test_referenced_helper_blocks_include_non_c_dependencies_but_not_prior_c_theorems():
    formal_proofs = (
        "import Mathlib.Tactic\n"
        "\n"
        "theorem C1 : 1 = 1 := by\n"
        "  rfl\n"
        "\n"
        "def helper (n : Nat) : Nat := n\n"
        "\n"
        "lemma helper_id (n : Nat) : helper n = n := by\n"
        "  rfl\n"
        "\n"
        "theorem C2 : helper 2 = 2 := by\n"
        "  simpa using helper_id 2\n"
    )

    helper_blocks = referenced_helper_blocks(formal_proofs, 11, 12)

    assert [block["decl_name"] for block in helper_blocks] == ["helper", "helper_id"]
    assembled, _ = build_fragment_assembled_answer(formal_proofs, 11, 12, helper_blocks)
    assert "theorem C1" not in assembled
    assert "def helper" in assembled
    assert "lemma helper_id" in assembled
    assert "theorem C2" in assembled


def test_build_fragment_assembled_answer_prepends_environment_for_c1_with_attached_docstring():
    formal_proofs = (
        "import Mathlib.Tactic\n"
        "namespace AutoFormal\n"
        "open scoped BigOperators\n"
        "\n"
        "/-!\n"
        "C1 docs.\n"
        "-/\n"
        "theorem C1 : True := by\n"
        "  trivial\n"
    )

    assembled, fragments = build_fragment_assembled_answer(formal_proofs, 5, 9)

    assert assembled.startswith(
        "import Mathlib.Tactic\nnamespace AutoFormal\nopen scoped BigOperators\n\n/-!\nC1 docs.\n"
    )
    assert fragments == [
        {"kind": "preamble", "start_line": 1, "end_line": 4},
        {"kind": "body", "start_line": 5, "end_line": 9},
    ]


def test_build_fragment_assembled_answer_does_not_promote_pre_c1_helper_into_shared_preamble():
    formal_proofs = (
        "import Mathlib.Tactic\n"
        "namespace AutoFormal\n"
        "\n"
        "def helper : Nat := 1\n"
        "\n"
        "/-!\n"
        "C1 docs.\n"
        "-/\n"
        "theorem C1 : helper = 1 := by\n"
        "  rfl\n"
        "\n"
        "/-!\n"
        "C2 docs.\n"
        "-/\n"
        "theorem C2 : True := by\n"
        "  trivial\n"
    )

    assembled, _ = build_fragment_assembled_answer(formal_proofs, 12, 16)

    assert assembled.startswith("import Mathlib.Tactic\nnamespace AutoFormal\n\n")
    assert "def helper" not in assembled
    assert "theorem C1" not in assembled
    assert "theorem C2" in assembled


def test_evaluate_boundary_response_canonicalizes_prefix_style_spans_to_target_theorem_block():
    entry = {
        "lean_prompt": "prompt",
        "qa_batch": [
            {"question": "q1", "answer": "a1"},
            {"question": "q2", "answer": "a2"},
        ],
        "formal_proofs": (
            "import Mathlib.Tactic\n\n"
            "theorem C1 : 1 = 1 := by\n"
            "  rfl\n\n"
            "theorem C2 : 2 = 2 := by\n"
            "  rfl\n"
        ),
    }
    candidates = detect_target_theorems(entry["formal_proofs"])
    c2_line = candidates[1]["line_number"]

    result = evaluate_boundary_response(
        entry,
        batch_index=1,
        response_text=json.dumps(
            {
                "items": [
                    {
                        "qa_index": 1,
                        "target_label": "C1",
                        "target_theorem_name": "C1",
                        "start_line": 1,
                        "end_line": c2_line,
                    },
                    {
                        "qa_index": 2,
                        "target_label": "C2",
                        "target_theorem_name": "C2",
                        "start_line": 1,
                        "end_line": source_line_count(entry["formal_proofs"]),
                    },
                ]
            }
        ),
        compile_fn=_ok_compile,
    )

    assert result["status"] == "validated"
    assert result["extracted_items"][0]["requested_start_line"] == 1
    assert result["extracted_items"][0]["start_line"] == candidates[0]["line_number"]
    assert detect_target_theorems(result["extracted_items"][0]["formal_answer"])[0]["theorem_name"] == "C1"
    assert result["extracted_items"][1]["requested_start_line"] == 1
    assert result["extracted_items"][1]["start_line"] == candidates[1]["line_number"]
    assert detect_target_theorems(result["extracted_items"][1]["formal_answer"])[0]["theorem_name"] == "C2"


def test_write_formal_qa_data_from_approved_audit_only_includes_approved_batches(tmp_path):
    audit_entries = [
        {
            "batch_index": 1,
            "status": "approved",
            "flattened_records": [{"question": "q1", "answer": "a1", "formal_answer": "p1"}],
        },
        {
            "batch_index": 2,
            "status": "failed",
            "flattened_records": [{"question": "q2", "answer": "a2", "formal_answer": "p2"}],
        },
    ]
    output_path = tmp_path / "formal_qa_data.json"

    flattened = write_formal_qa_data_from_audit(audit_entries, output_path)

    assert flattened == [{"question": "q1", "answer": "a1", "formal_answer": "p1"}]
    assert json.loads(output_path.read_text(encoding="utf-8")) == flattened


def test_evaluate_boundary_response_compiles_exact_real_chunks():
    entry = {
        "lean_prompt": "prompt",
        "qa_batch": [
            {"question": "q1", "answer": "a1"},
            {"question": "q2", "answer": "a2"},
        ],
        "formal_proofs": (
            "import Mathlib.Tactic\n\n"
            "theorem C1 : 1 = 1 := by\n"
            "  rfl\n\n"
            "theorem C2 : 2 = 2 := by\n"
            "  rfl\n"
        ),
    }
    line_count = source_line_count(entry["formal_proofs"])
    response_text = json.dumps(
        {
            "items": [
                {
                    "qa_index": 1,
                    "target_label": "C1",
                    "target_theorem_name": "C1",
                    "start_line": 3,
                    "end_line": 4,
                },
                {
                    "qa_index": 2,
                    "target_label": "C2",
                    "target_theorem_name": "C2",
                    "start_line": 6,
                    "end_line": line_count,
                },
            ]
        }
    )

    result = evaluate_boundary_response(
        entry,
        batch_index=1,
        response_text=response_text,
        compile_fn=compile_lean,
    )

    assert result["status"] == "validated"
    assert all(item["compile_ok"] for item in result["extracted_items"])
    assert all(len(detect_target_theorems(item["formal_answer"])) == 1 for item in result["extracted_items"])


def test_evaluate_boundary_response_compiles_first_theorem_with_preamble_and_docstring():
    entry = {
        "lean_prompt": "prompt",
        "qa_batch": [{"question": "q1", "answer": "a1"}],
        "formal_proofs": (
            "import Mathlib.Tactic\n"
            "namespace AutoFormal\n"
            "open scoped BigOperators\n"
            "\n"
            "/-!\n"
            "C1 docs.\n"
            "-/\n"
            "theorem C1 : True := by\n"
            "  trivial\n"
        ),
    }
    response_text = json.dumps(
        {
            "items": [
                {
                    "qa_index": 1,
                    "target_label": "C1",
                    "target_theorem_name": "C1",
                    "start_line": 8,
                    "end_line": 9,
                }
            ]
        }
    )

    result = evaluate_boundary_response(
        entry,
        batch_index=1,
        response_text=response_text,
        compile_fn=compile_lean,
    )

    assert result["status"] == "validated"
    assert result["extracted_items"][0]["compile_ok"] is True
    assert result["extracted_items"][0]["question"] == "q1"
    assert result["extracted_items"][0]["answer"] == "a1"


def test_evaluate_boundary_response_compiles_with_stitched_helper_blocks():
    entry = {
        "lean_prompt": "prompt",
        "qa_batch": [
            {"question": "q1", "answer": "a1"},
            {"question": "q2", "answer": "a2"},
        ],
        "formal_proofs": (
            "import Mathlib.Tactic\n\n"
            "theorem C1 : 1 = 1 := by\n"
            "  rfl\n\n"
            "def helper (n : Nat) : Nat := n\n\n"
            "lemma helper_id (n : Nat) : helper n = n := by\n"
            "  rfl\n\n"
            "theorem C2 : helper 2 = 2 := by\n"
            "  simpa using helper_id 2\n"
        ),
    }
    response_text = json.dumps(
        {
            "items": [
                {
                    "qa_index": 1,
                    "target_label": "C1",
                    "target_theorem_name": "C1",
                    "start_line": 3,
                    "end_line": 4,
                },
                {
                    "qa_index": 2,
                    "target_label": "C2",
                    "target_theorem_name": "C2",
                    "start_line": 11,
                    "end_line": 12,
                },
            ]
        }
    )

    result = evaluate_boundary_response(
        entry,
        batch_index=1,
        response_text=response_text,
        compile_fn=compile_lean,
    )

    assert result["status"] == "validated"
    c2 = result["extracted_items"][1]
    assert c2["compile_ok"] is True
    assert c2["helper_decl_names"] == ["helper", "helper_id"]
    assert "theorem C1" not in c2["formal_answer"]


def test_validate_extracted_formal_answer_rejects_cross_contaminated_theorem_content():
    formal_answer = (
        "import Mathlib.Tactic\n\n"
        "theorem C1 : 1 = 1 := by\n"
        "  rfl\n\n"
        "theorem C2 : 2 = 2 := by\n"
        "  rfl\n"
    )

    errors = validate_extracted_formal_answer(
        formal_answer,
        qa_index=1,
        target_theorem_name="C1",
    )

    assert errors == ["QA item 1 assembled answer must contain exactly one C-theorem, found 2."]


def test_validate_llm_extracted_formal_answer_allows_prior_target_dependency():
    formal_answer = (
        "import Mathlib.Tactic\n\n"
        "theorem C1 : True := by\n"
        "  trivial\n\n"
        "theorem C2 : True := by\n"
        "  exact C1\n"
    )

    errors = validate_llm_extracted_formal_answer(
        formal_answer,
        qa_index=2,
        target_theorem_name="C2",
    )

    assert errors == []


def test_real_batch_three_prefix_style_spans_are_canonicalized_to_target_blocks():
    structured_path = Path(__file__).resolve().parents[2] / "app_data" / "structured_proofs.json"
    data = json.loads(structured_path.read_text(encoding="utf-8"))
    entry = data[2]
    response_text = json.dumps(
        {
            "items": [
                {
                    "qa_index": 1,
                    "target_label": "C1",
                    "target_theorem_name": "C1_radialDerivative_eq_zero_of_gradient_eq_zero",
                    "start_line": 1,
                    "end_line": 40,
                },
                {
                    "qa_index": 2,
                    "target_label": "C2",
                    "target_theorem_name": "C2_constant_of_zero_radial_derivative",
                    "start_line": 47,
                    "end_line": 54,
                },
                {
                    "qa_index": 3,
                    "target_label": "C3",
                    "target_theorem_name": "C3_potentialDifference_is_constant",
                    "start_line": 61,
                    "end_line": 69,
                },
                {
                    "qa_index": 4,
                    "target_label": "C4",
                    "target_theorem_name": "C4_poisson_of_gaussLaw_and_potential_definition",
                    "start_line": 1,
                    "end_line": 98,
                },
                {
                    "qa_index": 5,
                    "target_label": "C5",
                    "target_theorem_name": "C5_integrated_poisson",
                    "start_line": 1,
                    "end_line": 118,
                },
            ]
        }
    )

    result = evaluate_boundary_response(
        entry,
        batch_index=3,
        response_text=response_text,
        compile_fn=_ok_compile,
    )

    assert result["status"] == "validated"
    assert result["extracted_items"][3]["requested_start_line"] == 1
    assert result["extracted_items"][3]["start_line"] == 71
    assert result["extracted_items"][4]["requested_start_line"] == 1
    assert result["extracted_items"][4]["start_line"] == 100


def test_evaluate_extraction_response_accepts_dependency_theorem_when_target_is_last():
    formal_answer_q1 = "import Mathlib.Tactic\n\ntheorem C1 : True := by\n  trivial\n"
    formal_answer_q2 = (
        "import Mathlib.Tactic\n\n"
        "theorem C1 : True := by\n"
        "  trivial\n\n"
        "theorem C2 : True := by\n"
        "  exact C1\n"
    )
    entry = {
        "lean_prompt": "prompt",
        "qa_batch": [
            {"question": "q1", "answer": "a1"},
            {"question": "q2", "answer": "a2"},
        ],
        "formal_proofs": formal_answer_q2,
    }
    response_text = json.dumps(
        {
            "items": [
                {
                    "qa_index": 1,
                    "target_label": "C1",
                    "target_theorem_name": "C1",
                    "formal_answer": formal_answer_q1,
                },
                {
                    "qa_index": 2,
                    "target_label": "C2",
                    "target_theorem_name": "C2",
                    "formal_answer": formal_answer_q2,
                },
            ]
        }
    )

    result = evaluate_extraction_response(
        entry,
        batch_index=1,
        response_text=response_text,
        compile_fn=compile_lean,
    )

    assert result["status"] == "validated"
    assert result["extraction_items"][1]["target_theorem_name"] == "C2"
    assert result["extracted_items"][1]["compile_ok"] is True
    assert result["flattened_records"][1]["question"] == "q2"
    assert result["flattened_records"][1]["answer"] == "a2"


def test_build_boundary_prompt_includes_previous_compile_feedback():
    entry = {
        "lean_prompt": "prompt",
        "qa_batch": [{"question": "q1", "answer": "a1"}],
        "formal_proofs": "import Mathlib.Tactic\n\ntheorem C1 : 1 = 1 := by\n  rfl\n",
    }
    previous_attempt = {
        "validation_errors": ["QA item 1 exact chunk did not compile (exit code 1)."],
        "boundary_items": [{"qa_index": 1, "target_label": "C1", "start_line": 2, "end_line": 4}],
        "extracted_items": [
            {
                "qa_index": 1,
                "target_theorem_name": "C1",
                "start_line": 2,
                "end_line": 4,
                "compile_ok": False,
                "compile_summary": "STDERR:\nunknown identifier",
            }
        ],
    }

    prompt = build_boundary_prompt(entry, batch_index=1, previous_attempt=previous_attempt)

    assert "Compiler feedback from the previous exact-slice attempt" in prompt
    assert "unknown identifier" in prompt
    assert "Adjust the line boundaries only. Do not rewrite Lean code." in prompt


def test_build_boundary_prompt_includes_structural_retry_instruction():
    entry = {
        "lean_prompt": "prompt",
        "qa_batch": [{"question": "q1", "answer": "a1"}, {"question": "q2", "answer": "a2"}],
        "formal_proofs": "import Mathlib.Tactic\n\ntheorem C1 : 1 = 1 := by\n  rfl\n\ntheorem C2 : 2 = 2 := by\n  rfl\n",
    }
    previous_attempt = {
        "validation_errors": [
            "Model returned 1 boundary items for 2 QA pairs.",
            "Boundary item indices must be exactly [1, 2], got [2].",
        ],
        "boundary_items": [{"qa_index": 2, "target_label": "C2", "start_line": 1, "end_line": 7}],
        "extracted_items": [],
    }

    prompt = build_boundary_prompt(entry, batch_index=1, previous_attempt=previous_attempt)

    assert "The previous response was structurally invalid." in prompt
    assert "Return exactly one JSON item per QA pair" in prompt
    assert "Do not rewrite Lean code or include irrelevant C-theorems." in prompt


def test_build_extraction_prompt_includes_compile_feedback_and_dependency_instruction():
    entry = {
        "lean_prompt": "prompt",
        "qa_batch": [{"question": "q1", "answer": "a1"}],
        "formal_proofs": "import Mathlib.Tactic\n\ntheorem C1 : True := by\n  trivial\n",
    }
    previous_attempt = {
        "validation_errors": ["QA item 1 extracted answer did not compile."],
        "extraction_items": [
            {
                "qa_index": 1,
                "target_label": "C1",
                "target_theorem_name": "C1",
                "formal_answer": "theorem C1 : True := by\n  exact missing\n",
            }
        ],
        "extracted_items": [
            {
                "qa_index": 1,
                "target_theorem_name": "C1",
                "compile_ok": False,
                "compile_summary": "STDERR:\nunknown identifier `missing`",
            }
        ],
    }

    prompt = build_extraction_prompt(entry, batch_index=1, previous_attempt=previous_attempt)

    assert "Compiler feedback from the previous extraction attempt" in prompt
    assert "unknown identifier `missing`" in prompt
    assert "You may include additional extracted dependencies" in prompt


def test_reconcile_audit_with_formal_qa_uses_flattened_prefix_as_source_of_truth():
    structured_entries = [
        {
            "lean_prompt": "p1",
            "qa_batch": [{"question": "q1", "answer": "a1"}, {"question": "q2", "answer": "a2"}],
            "formal_proofs": "proof1",
        },
        {
            "lean_prompt": "p2",
            "qa_batch": [{"question": "q3", "answer": "a3"}, {"question": "q4", "answer": "a4"}],
            "formal_proofs": "proof2",
        },
    ]
    formal_qa_entries = [
        {"question": "q1", "answer": "a1", "formal_answer": "fa1"},
        {"question": "q2", "answer": "a2", "formal_answer": "fa2"},
    ]
    audit_entries = [
        {"batch_index": 1, "status": "approved", "flattened_records": formal_qa_entries},
        {"batch_index": 2, "status": "failed", "flattened_records": [{"question": "q3", "answer": "a3", "formal_answer": "bad"}]},
    ]

    reconciled = reconcile_audit_with_formal_qa(structured_entries, audit_entries, formal_qa_entries)

    assert len(reconciled) == 1
    assert reconciled[0]["batch_index"] == 1
    assert reconciled[0]["status"] == "approved"
    assert reconciled[0]["flattened_records"] == formal_qa_entries
