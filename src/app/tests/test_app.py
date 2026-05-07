import json
import sys
import types
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

sys.modules.setdefault("gradio", types.SimpleNamespace(update=lambda **kwargs: kwargs))

import app as qa_app


def _record_by_file_name(records, file_name):
    for record in records:
        if record["file_name"] == file_name:
            return record
    raise AssertionError(f"Record for {file_name} not found")


def test_parse_generated_qa_json_repairs_unescaped_latex_backslashes():
    raw = r"""
```json
[
  {
    "question": "Show that \(x \in X\) and \[x^2 \ge 0\].",
    "answer": "Use \alpha and \frac{1}{2}."
  }
]
```
"""

    batch, batch_json_text, message = qa_app._parse_generated_qa_json(raw, expected_count=1)

    assert message == "1 QA pairs validated"
    assert batch == [
        {
            "question": r"Show that \(x \in X\) and \[x^2 \ge 0\].",
            "answer": r"Use \alpha and \frac{1}{2}.",
        }
    ]
    assert json.loads(batch_json_text) == batch


def test_parse_generated_qa_json_preserves_valid_json_escapes():
    raw = r"""
[
  {
    "question": "Line 1\nLine 2",
    "answer": "Escaped quote: \"x\" and valid LaTeX \\frac{1}{2}."
  }
]
"""

    batch, _, message = qa_app._parse_generated_qa_json(raw, expected_count=1)

    assert message == "1 QA pairs validated"
    assert batch == [
        {
            "question": "Line 1\nLine 2",
            "answer": 'Escaped quote: "x" and valid LaTeX \\frac{1}{2}.',
        }
    ]


def test_parse_alignment_json_repairs_unescaped_latex_backslashes():
    raw = r"""
```json
[
  {
    "alignment_comment": "Matches \(x \in X\) and uses \frac{1}{2} correctly.",
    "likert_score": 4.7
  }
]
```
"""

    entries, json_text, message = qa_app._parse_alignment_json(raw, expected_count=1)

    assert message == "1 alignment entries validated"
    assert entries == [
        {
            "alignment_comment": r"Matches \(x \in X\) and uses \frac{1}{2} correctly.",
            "likert_score": 4.7,
        }
    ]
    assert json.loads(json_text) == entries


def test_build_batch_generation_prompt_requires_json_escaped_latex():
    prompt = qa_app._build_batch_generation_prompt(
        [{"item_id": "reasoning_1", "reasoning_text": "Some derivation."}]
    )

    assert "escape every backslash inside string values" in prompt
    assert r"write `\\(` not `\(`" in prompt


def test_extraction_prompts_do_not_reference_few_shot_examples():
    assert "few-shot" not in qa_app._EXTRACTION_PROMPT_RANGED
    assert "few-shot" not in qa_app._EXTRACTION_PROMPT_GENERIC


def test_eval_preamble_requests_1dp_likert_scores():
    assert "5-point Likert score to 1dp" in qa_app._EVAL_PREAMBLE
    assert "Use any score in 0.1 increments from 1.0 to 5.0 when needed" in qa_app._EVAL_PREAMBLE
    assert "escape every backslash inside string values" in qa_app._EVAL_PREAMBLE
    assert '"alignment_comment"' in qa_app._EVAL_PREAMBLE
    assert '"likert_score"' in qa_app._EVAL_PREAMBLE
    assert "Do not include any extra keys, markdown, or overall summary entry." in qa_app._EVAL_PREAMBLE


def test_alignment_prompt_includes_rubric_source_reasoning_panel_text_and_current_qa():
    source_reasoning_text = "### Core Input\n\n#### Example 1: `core_reasoning_1`\n\nSome derivation."
    current_output_json = '[{"question":"Q","answer":"A"}]'

    prompt = qa_app._build_alignment_prompt(source_reasoning_text, current_output_json)

    rubric_text = qa_app._load_qa_rubric_text()

    assert "QA RUBRIC (qa_rubric.md):" in prompt
    assert rubric_text in prompt
    assert "SOURCE REASONING PANEL TEXT:" in prompt
    assert source_reasoning_text in prompt
    assert "CURRENT TEMP QA JSON:" in prompt
    assert current_output_json in prompt


def test_alignment_checkbox_update_requires_all_pair_scores_above_4_5():
    checked = qa_app._alignment_checkbox_update(
        [
            {"alignment_comment": "Strong match.", "likert_score": 4.6},
            {"alignment_comment": "Also strong.", "likert_score": 4.7},
        ]
    )
    unchecked = qa_app._alignment_checkbox_update(
        [
            {"alignment_comment": "Strong match.", "likert_score": 4.6},
            {"alignment_comment": "Borderline.", "likert_score": 4.4},
        ]
    )

    assert checked == {"value": True}
    assert unchecked == {"value": False}


def test_parse_alignment_json_accepts_valid_entries():
    raw = """
```json
[
  {"alignment_comment": "Strong alignment.", "likert_score": 4.6},
  {"alignment_comment": "Very strong alignment.", "likert_score": 4.8}
]
```
"""

    entries, json_text, message = qa_app._parse_alignment_json(raw, expected_count=2)

    assert message == "2 alignment entries validated"
    assert entries == [
        {"alignment_comment": "Strong alignment.", "likert_score": 4.6},
        {"alignment_comment": "Very strong alignment.", "likert_score": 4.8},
    ]
    assert json.loads(json_text) == entries


def test_render_alignment_markdown_formats_saved_entries():
    rendered = qa_app._render_alignment_markdown(
        [{"alignment_comment": "Strong alignment.", "likert_score": 4.6, "change_summary": "None"}]
    )

    assert "### Change Summary" in rendered
    assert "None" in rendered
    assert "### QA Pair 1" in rendered
    assert "4.6/5.0" in rendered
    assert "Strong alignment." in rendered


def test_attach_change_summary_applies_same_value_to_each_entry():
    entries = qa_app._attach_change_summary(
        [
            {"alignment_comment": "Strong alignment.", "likert_score": 4.6},
            {"alignment_comment": "Very strong alignment.", "likert_score": 4.8},
        ],
        "None",
    )

    assert entries == [
        {"alignment_comment": "Strong alignment.", "likert_score": 4.6, "change_summary": "None"},
        {"alignment_comment": "Very strong alignment.", "likert_score": 4.8, "change_summary": "None"},
    ]


def test_build_change_summary_summarizes_pair_edits_without_literal_diff():
    previous = json.dumps(
        [
            {"question": "Q1", "answer": "A1"},
            {"question": "Q2", "answer": "A2"},
        ]
    )
    current = json.dumps(
        [
            {"question": "Q1 revised", "answer": "A1"},
            {"question": "Q2", "answer": "A2 revised"},
        ]
    )

    summary = qa_app._build_change_summary(previous, current)

    assert "Revised 2 QA pair(s)." in summary
    assert "QA pair 1: revised question." in summary
    assert "QA pair 2: revised answer." in summary
    assert "@@" not in summary


def test_resume_record_index_uses_saved_pipeline_progress_per_mode(tmp_path, monkeypatch):
    app_data_dir = tmp_path / "app_data"
    app_data_dir.mkdir()
    (app_data_dir / "pipeline_progress.json").write_text(
        json.dumps({"Core": 1, "Deeper": 2}),
        encoding="utf-8",
    )

    monkeypatch.setattr(qa_app, "_SRC", tmp_path)
    monkeypatch.setattr(
        qa_app,
        "_PIPELINE_RECORDS",
        {"Core": [{}, {}], "Deeper": [{}, {}, {}]},
    )

    assert qa_app._resume_record_index("Core", approved_pairs=[]) == 1
    assert qa_app._resume_record_index("Deeper", approved_pairs=[]) == 2


def test_resume_record_index_falls_back_to_saved_batch_count_without_progress_file(tmp_path, monkeypatch):
    (tmp_path / "app_data").mkdir()

    monkeypatch.setattr(qa_app, "_SRC", tmp_path)
    monkeypatch.setattr(
        qa_app,
        "_PIPELINE_RECORDS",
        {"Core": [{}, {}], "Deeper": [{}]},
    )

    approved_pairs = [{"batch": [{"question": "Q1", "answer": "A1"}]} for _ in range(3)]

    assert qa_app._resume_record_index("Core", approved_pairs=approved_pairs) == 2
    assert qa_app._resume_record_index("Deeper", approved_pairs=approved_pairs) == 1


def test_core_and_deeper_pipeline_records_share_the_same_record_shape():
    datasets = qa_app._load_pipeline_records()
    expected_record_keys = {
        "mode",
        "file_number",
        "file_position",
        "dataset_file_total",
        "file_name",
        "file_path",
        "paper_title",
        "example_count",
        "examples",
        "prompt_text",
        "display_text",
    }

    for mode in ("Core", "Deeper"):
        for record in datasets[mode]:
            assert set(record) == expected_record_keys
            assert record["example_count"] == len(record["examples"])
            assert record["example_count"] > 0
            assert record["prompt_text"].startswith(
                f"Convert the following {record['example_count']} reasoning examples into exactly "
                f"{record['example_count']} question-answer pairs."
            )
            assert f"### {record['mode']} Input" in record["display_text"]
            assert f"**Reasoning Examples:** {record['example_count']}" in record["display_text"]
            for example in record["examples"]:
                assert set(example) == {"item_id", "reasoning_text"}
                assert example["item_id"].strip()
                assert example["reasoning_text"].strip()


def test_deeper_examples_are_extracted_from_top_level_reasoning_keys():
    datasets = qa_app._load_pipeline_records()
    paper_data_dir = qa_app._ROOT / "data" / "paper_data"

    for deeper_name in ("deeper_01.json",):
        record = _record_by_file_name(datasets["Deeper"], deeper_name)
        raw_data = json.loads((paper_data_dir / deeper_name).read_text(encoding="utf-8"))
        reasoning_keys = sorted(
            [
                key for key, value in raw_data.items()
                if key.startswith("deeper_reasoning_") and isinstance(value, str) and value.strip()
            ],
            key=qa_app._reasoning_sort_key,
        )

        assert record["example_count"] == len(reasoning_keys)
        assert [example["item_id"] for example in record["examples"]] == reasoning_keys

        for key, example in zip(reasoning_keys, record["examples"]):
            assert example["item_id"] == key
            assert example["reasoning_text"] == raw_data[key].strip()
