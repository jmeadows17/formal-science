import json
import sys
import time
import types
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


class _DummySession:
    def __init__(self, *args, **kwargs):
        self.session_id = kwargs.get("session_id")

    @classmethod
    def resume(cls, session_id, **kwargs):
        return cls(session_id=session_id, **kwargs)

    def prompt(self, _message):
        return {"result": "", "session_id": self.session_id}


sys.modules.setdefault("gradio", types.SimpleNamespace(update=lambda **kwargs: kwargs))
sys.modules.setdefault(
    "claude_cli",
    types.SimpleNamespace(ClaudeSession=_DummySession, CLAUDE_REASONING_EFFORTS=("low", "medium", "high")),
)
sys.modules.setdefault(
    "gpt_cli",
    types.SimpleNamespace(
        GPTSession=_DummySession,
        VALID_REASONING_EFFORTS={"minimal", "low", "medium", "high", "xhigh"},
    ),
)

import lean_app


def _accepted_critic_report() -> str:
    return """Decision: accept
Scope label: Unconditional

Target theorem fidelity: 4.5 / 5
Object fidelity: 4.2 / 5
Burden discharge: 4.4 / 5
Assumption hygiene: 4.8 / 5
Overall score: min = 4.2 / 5

Rejection conditions applied:
- None.

Score caps applied:
- None.

Re-informalization:
- The theorem proves exactly the requested claim.

Central burdens:
- Burden 1.

Surrogate objects:
- None.

Theorem-shaped assumptions:
- None.

Definitions encoding conclusions:
- None.

External facts requiring replacement or stronger justification:
- None.

Required repairs:
- None.
"""


def test_parse_critic_report_accepts_valid_report_and_allows_approval():
    report = lean_app._parse_critic_report(_accepted_critic_report())

    assert report["is_valid"] is True
    assert report["decision"] == "accept"
    assert report["overall_score"] == 4.2
    assert report["rejection_conditions_active"] is False
    assert report["score_caps_active"] is False
    assert lean_app._critic_report_allows_compile_fix(report) is True
    assert lean_app._critic_report_allows_approval(report) is True


def test_parse_critic_report_blocks_active_score_cap():
    report_text = _accepted_critic_report().replace("- None.\n\nRe-informalization", "- Surrogate object caps score at 2.0.\n\nRe-informalization")
    report = lean_app._parse_critic_report(report_text)

    assert report["is_valid"] is True
    assert report["score_caps_active"] is True
    assert lean_app._critic_report_allows_compile_fix(report) is False
    assert lean_app._critic_report_allows_approval(report) is False


def test_parse_critic_report_handles_bold_header_and_bare_score_format():
    report_text = """**Decision**

repair

**Scope label**

Not acceptable

**Target theorem fidelity**

2.5

**Object fidelity**

2.0

**Burden discharge**

2.0

**Assumption hygiene**

2.0

**Overall score**

2.0

**Rejection conditions applied**

- §10.4: central object replaced by surrogate (lowered Object fidelity to 2.0).

**Score caps applied**

- Surrogate object cap on Object fidelity.

**Re-informalization**

The theorem proves something weaker than asked.

**Required repairs**

- Replace surrogate definition with faithful object.
"""
    report = lean_app._parse_critic_report(report_text)

    assert report["is_valid"] is True, report["parse_errors"]
    assert report["decision"] == "repair"
    assert report["scope_label"] == "Not acceptable"
    assert report["target_theorem_fidelity"] == 2.5
    assert report["overall_score"] == 2.0
    assert report["rejection_conditions_active"] is True
    assert report["score_caps_active"] is True
    assert lean_app._critic_report_allows_compile_fix(report) is False
    assert lean_app._critic_report_allows_approval(report) is False


def test_parse_critic_report_blocks_active_rejection_condition():
    report_text = _accepted_critic_report().replace(
        "Rejection conditions applied:\n- None.",
        "Rejection conditions applied:\n- §10.5: central burden assumed instead of proved.",
    )
    report = lean_app._parse_critic_report(report_text)

    assert report["is_valid"] is True
    assert report["rejection_conditions_active"] is True
    assert lean_app._critic_report_allows_compile_fix(report) is False
    assert lean_app._critic_report_allows_approval(report) is False


def test_pipeline_timer_preserves_start_time_when_task_key_matches(tmp_path, monkeypatch):
    timer_path = tmp_path / "pipeline_timer.json"
    monkeypatch.setattr(lean_app, "DEFAULT_PIPELINE_TIMER_PATH", timer_path)

    key = lean_app._pipeline_task_key(0, "prompt 1")
    fixed_times = iter([1000.0, 2000.0])
    monkeypatch.setattr(lean_app.time, "time", lambda: next(fixed_times))

    first = lean_app._ensure_pipeline_start_time(key)
    second = lean_app._ensure_pipeline_start_time(key)

    assert first == 1000.0
    assert second == 1000.0  # regenerate / retry preserves the timer
    assert lean_app._read_pipeline_start_time() == 1000.0


def test_pipeline_timer_resets_on_new_task_key(tmp_path, monkeypatch):
    timer_path = tmp_path / "pipeline_timer.json"
    monkeypatch.setattr(lean_app, "DEFAULT_PIPELINE_TIMER_PATH", timer_path)

    key1 = lean_app._pipeline_task_key(0, "prompt 1")
    key2 = lean_app._pipeline_task_key(1, "prompt 2")
    fixed_times = iter([1000.0, 2000.0])
    monkeypatch.setattr(lean_app.time, "time", lambda: next(fixed_times))

    first = lean_app._ensure_pipeline_start_time(key1)
    second = lean_app._ensure_pipeline_start_time(key2)

    assert first == 1000.0
    assert second == 2000.0  # advancing to a new QA pair resets the timer
    assert lean_app._read_pipeline_start_time() == 2000.0


def test_persist_approved_entry_writes_artifacts_without_critic_gate(tmp_path, monkeypatch):
    proof_code = "import Mathlib.Tactic\n\ntheorem C1 : True := by\n  trivial\n"
    critic_output = "Decision: repair\n(rejected by critic, but budget exhausted)"

    output_path = tmp_path / "lean_output_data.json"
    structured_path = tmp_path / "structured_proofs.json"
    approved_batches_path = tmp_path / "approved_formal_batches.json"
    timer_path = tmp_path / "pipeline_timer.json"

    monkeypatch.setattr(lean_app, "DEFAULT_OUTPUT_PATH", output_path)
    monkeypatch.setattr(lean_app, "DEFAULT_STRUCTURED_OUTPUT_PATH", structured_path)
    monkeypatch.setattr(lean_app, "DEFAULT_APPROVED_FORMAL_BATCHES_PATH", approved_batches_path)
    monkeypatch.setattr(lean_app, "DEFAULT_CRITIC_COUNT_PATH", tmp_path / "critic_count.json")
    monkeypatch.setattr(lean_app, "DEFAULT_COMPILE_FIX_COUNT_PATH", tmp_path / "compile_fix_count.json")
    monkeypatch.setattr(lean_app, "DEFAULT_PIPELINE_TIMER_PATH", timer_path)
    monkeypatch.setattr(lean_app, "DEFAULT_CRITIC_DECISION_PATH", tmp_path / "critic_decision.json")
    monkeypatch.setattr(
        lean_app,
        "PROMPT_QA_PAIRS",
        [
            {
                "lean_prompt": "prompt 1",
                "qa_batch": [{"question": "q1", "answer": "a1"}],
                "source_batch_index": 1,
                "qa_index_in_batch": 1,
                "source_batch_size": 1,
            }
        ],
    )
    monkeypatch.setattr(
        lean_app,
        "ORIGINAL_QA_BATCHES",
        [[{"question": "q1", "answer": "a1"}]],
    )
    lean_app._write_critic_count(3)  # budget exhausted at MAX_AUTO_REPAIRS=3
    lean_app._write_compile_fix_count(7)
    lean_app._write_pipeline_start_time(1000.0, task_key="0\x1fprompt 1")
    monkeypatch.setattr(lean_app.time, "time", lambda: 1042.5)

    next_saved = lean_app._persist_approved_entry(
        proof_code, critic_output, 0, [], "prompt 1",
    )

    assert len(next_saved) == 1
    entry = next_saved[0]
    assert entry["output"] == proof_code
    assert entry["critic_output"] == critic_output
    assert entry["critic_count"] == 3
    assert entry["elapsed_seconds"] == 42.5
    assert entry["avg_compile_fix_per_critic_call"] == 7 / 4  # critic_count + 1
    approved = json.loads(approved_batches_path.read_text(encoding="utf-8"))
    assert approved == [[{"question": "q1", "answer": "a1", "formal_answer": proof_code}]]


def test_pipeline_task_key_differs_across_indices_and_prompts():
    base = lean_app._pipeline_task_key(0, "prompt")
    assert base != lean_app._pipeline_task_key(1, "prompt")
    assert base != lean_app._pipeline_task_key(0, "different prompt")


def test_build_compile_fix_message_omits_diff_section_on_first_attempt():
    msg = lean_app._build_compile_fix_message(
        "theorem C : True := by trivial\n",
        "error: unknown tactic\n",
        iteration=1,
    )
    assert "Compile attempt: 1" in msg
    assert "Compiler output:" in msg
    assert "Current Lean code:" in msg
    assert "Previous fix attempt diff" not in msg


def test_build_compile_fix_message_includes_diff_section_when_provided():
    diff = lean_app._build_brief_patch_diff(
        "theorem C : True := by sorry\n",
        "theorem C : True := by trivial\n",
        from_label="before_previous_fix.lean",
        to_label="after_previous_fix.lean",
    )
    assert diff  # sanity: non-empty diff for a real edit
    msg = lean_app._build_compile_fix_message(
        "theorem C : True := by trivial\n",
        "error: still broken\n",
        iteration=2,
        previous_attempt_diff=diff,
    )
    assert "Previous fix attempt diff" in msg
    assert "before_previous_fix.lean" in msg
    assert "after_previous_fix.lean" in msg
    assert "-theorem C : True := by sorry" in msg
    assert "+theorem C : True := by trivial" in msg


def test_format_patch_summary_includes_diff_for_changed_proof():
    summary = lean_app._format_patch_summary(
        "theorem old_name : True := by\n  trivial\n",
        "theorem new_name : True := by\n  trivial\n",
    )

    assert "Patch summary for `FSLean/proof.lean`:" in summary
    assert "--- previous_proof.lean" in summary
    assert "+++ current_proof.lean" in summary
    assert "-theorem old_name : True := by" in summary
    assert "+theorem new_name : True := by" in summary


def test_load_prompt_qa_pairs_flattens_batches_and_rebuilds_stale_prompt_cache(tmp_path):
    qa_path = tmp_path / "qa_data.json"
    prompt_path = tmp_path / "lean_prompt_data.json"

    qa_path.write_text(
        json.dumps(
            [
                [
                    {"question": "q1", "answer": "a1"},
                    {"question": "q2", "answer": "a2"},
                ],
                [
                    {"question": "q3", "answer": "a3"},
                ],
            ]
        ),
        encoding="utf-8",
    )
    prompt_path.write_text(json.dumps(["old batch prompt", "old batch prompt 2"]), encoding="utf-8")

    entries = lean_app._load_prompt_qa_pairs(prompt_path=prompt_path, qa_path=qa_path)

    assert len(entries) == 3
    assert entries[0]["qa_batch"] == [{"question": "q1", "answer": "a1"}]
    assert entries[0]["source_batch_index"] == 1
    assert entries[0]["qa_index_in_batch"] == 1
    assert entries[1]["qa_index_in_batch"] == 2
    assert entries[2]["source_batch_index"] == 2
    rebuilt_prompts = json.loads(prompt_path.read_text(encoding="utf-8"))
    assert len(rebuilt_prompts) == 3


def test_rebuild_batch_artifacts_from_saved_outputs_reconstructs_complete_batches(monkeypatch):
    monkeypatch.setattr(
        lean_app,
        "PROMPT_QA_PAIRS",
        [
            {
                "lean_prompt": "prompt 1",
                "qa_batch": [{"question": "q1", "answer": "a1"}],
                "source_batch_index": 1,
                "qa_index_in_batch": 1,
                "source_batch_size": 2,
            },
            {
                "lean_prompt": "prompt 2",
                "qa_batch": [{"question": "q2", "answer": "a2"}],
                "source_batch_index": 1,
                "qa_index_in_batch": 2,
                "source_batch_size": 2,
            },
            {
                "lean_prompt": "prompt 3",
                "qa_batch": [{"question": "q3", "answer": "a3"}],
                "source_batch_index": 2,
                "qa_index_in_batch": 1,
                "source_batch_size": 1,
            },
        ],
    )
    monkeypatch.setattr(
        lean_app,
        "ORIGINAL_QA_BATCHES",
        [
            [{"question": "q1", "answer": "a1"}, {"question": "q2", "answer": "a2"}],
            [{"question": "q3", "answer": "a3"}],
        ],
    )

    structured, approved = lean_app._rebuild_batch_artifacts_from_saved_outputs(
        [
            {
                "prompt": "prompt 1",
                "output": "import Mathlib.Tactic\n\ntheorem C1 : True := by\n  trivial\n",
                "critic_output": "critic 1",
                "critic_count": 0,
                "critic_decision": True,
                "elapsed_seconds": 12.5,
                "avg_compile_fix_per_critic_call": 2.0,
            },
            {
                "prompt": "prompt 2",
                "output": "import Mathlib.Data.Real.Basic\n\ntheorem C2 : True := by\n  trivial\n",
                "critic_output": "critic 2",
                "critic_count": 3,
                "critic_decision": False,
                "elapsed_seconds": 47.25,
                "avg_compile_fix_per_critic_call": 4.5,
            },
        ]
    )

    assert structured == [
        [
            {
                "question": "q1",
                "answer": "a1",
                "formal_proof": "import Mathlib.Tactic\n\ntheorem C1 : True := by\n  trivial\n",
                "critic_output": "critic 1",
                "critic_count": 0,
                "critic_decision": True,
                "elapsed_seconds": 12.5,
                "avg_compile_fix_per_critic_call": 2.0,
            },
            {
                "question": "q2",
                "answer": "a2",
                "formal_proof": "import Mathlib.Data.Real.Basic\n\ntheorem C2 : True := by\n  trivial\n",
                "critic_output": "critic 2",
                "critic_count": 3,
                "critic_decision": False,
                "elapsed_seconds": 47.25,
                "avg_compile_fix_per_critic_call": 4.5,
            },
        ]
    ]
    assert approved == [
        [
            {
                "question": "q1",
                "answer": "a1",
                "formal_answer": "import Mathlib.Tactic\n\ntheorem C1 : True := by\n  trivial\n",
            },
            {
                "question": "q2",
                "answer": "a2",
                "formal_answer": "import Mathlib.Data.Real.Basic\n\ntheorem C2 : True := by\n  trivial\n",
            },
        ]
    ]


def test_on_approve_persists_even_when_critic_did_not_accept(tmp_path, monkeypatch):
    """Manual Approve must never be gated by the critic verdict — the loop ends, we advance."""
    proof_text = "theorem C1 : True := by\n  trivial\n"
    proof_path = tmp_path / "proof.lean"
    proof_path.write_text(proof_text, encoding="utf-8")
    critic_path = tmp_path / "critic_output.md"
    critic_path.write_text(_accepted_critic_report().replace("Decision: accept", "Decision: repair"), encoding="utf-8")
    output_path = tmp_path / "lean_output_data.json"

    monkeypatch.setattr(lean_app, "DEFAULT_PROOF_PATH", proof_path)
    monkeypatch.setattr(lean_app, "DEFAULT_CRITIC_OUTPUT_PATH", critic_path)
    monkeypatch.setattr(lean_app, "DEFAULT_CRITIC_DECISION_PATH", tmp_path / "critic_decision.json")
    monkeypatch.setattr(lean_app, "DEFAULT_CRITIC_COUNT_PATH", tmp_path / "critic_count.json")
    monkeypatch.setattr(lean_app, "DEFAULT_COMPILE_FIX_COUNT_PATH", tmp_path / "compile_fix_count.json")
    monkeypatch.setattr(lean_app, "DEFAULT_PIPELINE_TIMER_PATH", tmp_path / "pipeline_timer.json")
    monkeypatch.setattr(lean_app, "DEFAULT_OUTPUT_PATH", output_path)

    list(
        lean_app.on_approve(
            proof_text, "GPT", None, "gpt-5.4", "high",
            0, [], "prompt",
        )
    )

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(saved) == 1
    assert saved[0]["output"] == proof_text
    assert "Decision: repair" in saved[0]["critic_output"]


def test_on_approve_persists_structured_and_approved_batches(tmp_path, monkeypatch):
    proof_path = tmp_path / "proof.lean"
    proof_path.write_text(
        "import Mathlib.Data.Real.Basic\n\n"
        "theorem C2 : True := by\n"
        "  trivial\n\n"
        "lemma helper : True := by\n"
        "  trivial\n",
        encoding="utf-8",
    )
    critic_path = tmp_path / "critic_output.md"
    critic_path.write_text(_accepted_critic_report(), encoding="utf-8")

    output_path = tmp_path / "lean_output_data.json"
    structured_path = tmp_path / "structured_proofs.json"
    approved_batches_path = tmp_path / "approved_formal_batches.json"

    monkeypatch.setattr(lean_app, "DEFAULT_PROOF_PATH", proof_path)
    monkeypatch.setattr(lean_app, "DEFAULT_CRITIC_OUTPUT_PATH", critic_path)
    monkeypatch.setattr(lean_app, "DEFAULT_CRITIC_DECISION_PATH", tmp_path / "critic_decision.json")
    monkeypatch.setattr(lean_app, "DEFAULT_CRITIC_COUNT_PATH", tmp_path / "critic_count.json")
    monkeypatch.setattr(lean_app, "DEFAULT_COMPILE_FIX_COUNT_PATH", tmp_path / "compile_fix_count.json")
    monkeypatch.setattr(lean_app, "DEFAULT_PIPELINE_TIMER_PATH", tmp_path / "pipeline_timer.json")
    monkeypatch.setattr(lean_app, "DEFAULT_OUTPUT_PATH", output_path)
    monkeypatch.setattr(lean_app, "DEFAULT_STRUCTURED_OUTPUT_PATH", structured_path)
    monkeypatch.setattr(lean_app, "DEFAULT_APPROVED_FORMAL_BATCHES_PATH", approved_batches_path)
    monkeypatch.setattr(
        lean_app,
        "PROMPT_QA_PAIRS",
        [
            {
                "lean_prompt": "prompt 1",
                "qa_batch": [{"question": "q1", "answer": "a1"}],
                "source_batch_index": 1,
                "qa_index_in_batch": 1,
                "source_batch_size": 2,
            },
            {
                "lean_prompt": "prompt",
                "qa_batch": [{"question": "q2", "answer": "a2"}],
                "source_batch_index": 1,
                "qa_index_in_batch": 2,
                "source_batch_size": 2,
            }
        ],
    )
    monkeypatch.setattr(
        lean_app,
        "ORIGINAL_QA_BATCHES",
        [[{"question": "q1", "answer": "a1"}, {"question": "q2", "answer": "a2"}]],
    )
    monkeypatch.setattr(lean_app, "TOTAL_PROMPTS", 1)
    output_path.write_text(
        json.dumps(
            [
                {
                    "prompt": "prompt 1",
                    "output": "import Mathlib.Tactic\n\ntheorem C1 : True := by\n  trivial\n",
                    "critic_output": "critic 1",
                    "critic_count": 0,
                    "critic_decision": True,
                }
            ]
        ),
        encoding="utf-8",
    )

    def _fake_send_prompt(*_args, **_kwargs):
        yield (
            "*Pipeline complete.*",
            "*Waiting...*",
            "*No compilation run yet.*",
            lean_app.CRITIC_WAITING,
            None,
            1,
            [
                {
                    "prompt": "prompt 1",
                    "output": "import Mathlib.Tactic\n\ntheorem C1 : True := by\n  trivial\n",
                },
                {"prompt": "prompt", "output": proof_path.read_text(encoding="utf-8")},
            ],
            "",
            {"visible": False},
            {"visible": False},
            {"visible": False},
            "done",
        )

    monkeypatch.setattr(lean_app, "send_prompt", _fake_send_prompt)

    list(
        lean_app.on_approve(
            proof_path.read_text(encoding="utf-8"),
            "GPT",
            None,
            "gpt-5.4",
            "high",
            1,
            json.loads(output_path.read_text(encoding="utf-8")),
            "prompt",
        )
    )

    structured = json.loads(structured_path.read_text(encoding="utf-8"))
    approved_batches = json.loads(approved_batches_path.read_text(encoding="utf-8"))

    assert structured == [
        [
            {
                "question": "q1",
                "answer": "a1",
                "formal_proof": "import Mathlib.Tactic\n\ntheorem C1 : True := by\n  trivial\n",
                "critic_output": "critic 1",
                "critic_count": 0,
                "critic_decision": True,
                "elapsed_seconds": 0.0,
                "avg_compile_fix_per_critic_call": 0.0,
            },
            {
                "question": "q2",
                "answer": "a2",
                "formal_proof": proof_path.read_text(encoding="utf-8"),
                "critic_output": critic_path.read_text(encoding="utf-8"),
                "critic_count": 0,
                "critic_decision": None,
                "elapsed_seconds": 0.0,
                "avg_compile_fix_per_critic_call": 0.0,
            },
        ]
    ]
    assert approved_batches == [
        [
            {
                "question": "q1",
                "answer": "a1",
                "formal_answer": "import Mathlib.Tactic\n\ntheorem C1 : True := by\n  trivial\n",
            },
            {
                "question": "q2",
                "answer": "a2",
                "formal_answer": proof_path.read_text(encoding="utf-8"),
            },
        ]
    ]


def test_send_prompt_uses_gpt_written_proof_file(tmp_path, monkeypatch):
    proof_path = tmp_path / "proof.lean"
    proof_path.write_text("stale contents", encoding="utf-8")
    critic_path = tmp_path / "critic_output.md"
    captured: dict[str, object] = {}
    expected_proof = "theorem C1 : True := by\n  trivial\n"

    class _WritingGPTSession:
        def __init__(self, *args, **kwargs):
            captured["session_kwargs"] = kwargs
            self.session_id = kwargs.get("session_id") or "session-123"

        @classmethod
        def resume(cls, session_id, **kwargs):
            return cls(session_id=session_id, **kwargs)

        def prompt(self, message):
            captured["message"] = message
            proof_path.write_text(expected_proof, encoding="utf-8")
            return {"result": "wrote proof file", "session_id": self.session_id}

    def _fake_compile_then_critic(
        prompt_display,
        provider,
        session_id,
        model,
        reasoning_effort,
        prompt_index,
        saved_outputs,
        current_prompt,
        previous_evaluated_code=None,
    ):
        captured["critic_args"] = {
            "prompt_display": prompt_display,
            "provider": provider,
            "session_id": session_id,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "prompt_index": prompt_index,
            "saved_outputs": saved_outputs,
            "current_prompt": current_prompt,
            "previous_evaluated_code": previous_evaluated_code,
        }
        yield (
            prompt_display,
            proof_path.read_text(encoding="utf-8"),
            "compile panel",
            "critic panel",
            session_id,
            prompt_index,
            saved_outputs,
            current_prompt,
            {"visible": False},
            {"visible": False},
            {"visible": False},
            "status",
        )

    prompt_text = "Prove `True` in Lean."
    monkeypatch.setattr(lean_app, "GPTSession", _WritingGPTSession)
    monkeypatch.setattr(lean_app, "DEFAULT_PROOF_PATH", proof_path)
    monkeypatch.setattr(lean_app, "DEFAULT_CRITIC_OUTPUT_PATH", critic_path)
    monkeypatch.setattr(lean_app, "_compile_then_critic", _fake_compile_then_critic)
    monkeypatch.setattr(lean_app, "_refresh_prompt_data", lambda: None)
    monkeypatch.setattr(lean_app, "PROMPTS", [prompt_text])
    monkeypatch.setattr(lean_app, "TOTAL_PROMPTS", 1)

    states = list(
        lean_app.send_prompt(
            "GPT",
            None,
            "gpt-5.4",
            "high",
            0,
            [],
        )
    )

    assert proof_path.read_text(encoding="utf-8") == expected_proof
    assert captured["critic_args"] == {
        "prompt_display": f"### Prompt 1 / 1\n\n---\n\n{prompt_text}",
        "provider": "GPT",
        "session_id": "session-123",
        "model": "gpt-5.4",
        "reasoning_effort": "high",
        "prompt_index": 0,
        "saved_outputs": [],
        "current_prompt": prompt_text,
        "previous_evaluated_code": None,
    }
    assert captured["session_kwargs"] == {
        "model": "gpt-5.4",
        "cwd": str(lean_app.REPO_ROOT),
        "reasoning_effort": "high",
        "tools": [str(lean_app.REPO_ROOT)],
    }
    assert captured["message"] == (
        f"{prompt_text}\n\n"
        f"{lean_app.CODE_ONLY_SUFFIX}\n"
    )
    assert states[0][1] == "*Writing `FSLean/proof.lean`...*"
    assert states[-1][1] == expected_proof


def test_send_prompt_persists_inline_gpt_response_when_model_does_not_write_file(tmp_path, monkeypatch):
    proof_path = tmp_path / "proof.lean"
    critic_path = tmp_path / "critic_output.md"
    expected_proof = "theorem C1 : True := by\n  trivial\n"
    captured: dict[str, object] = {}

    class _InlineOnlyGPTSession:
        def __init__(self, *args, **kwargs):
            self.session_id = kwargs.get("session_id") or "session-inline"

        @classmethod
        def resume(cls, session_id, **kwargs):
            return cls(session_id=session_id, **kwargs)

        def prompt(self, _message):
            return {"result": expected_proof, "session_id": self.session_id}

    def _fake_compile_then_critic(
        prompt_display,
        provider,
        session_id,
        model,
        reasoning_effort,
        prompt_index,
        saved_outputs,
        current_prompt,
        previous_evaluated_code=None,
    ):
        captured["critic_args"] = {
            "provider": provider,
            "session_id": session_id,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "prompt_index": prompt_index,
            "saved_outputs": saved_outputs,
            "current_prompt": current_prompt,
            "previous_evaluated_code": previous_evaluated_code,
        }
        yield (
            prompt_display,
            proof_path.read_text(encoding="utf-8"),
            "compile panel",
            "critic panel",
            session_id,
            prompt_index,
            saved_outputs,
            current_prompt,
            {"visible": False},
            {"visible": False},
            {"visible": False},
            "status",
        )

    prompt_text = "Prove `True` in Lean."
    monkeypatch.setattr(lean_app, "GPTSession", _InlineOnlyGPTSession)
    monkeypatch.setattr(lean_app, "DEFAULT_PROOF_PATH", proof_path)
    monkeypatch.setattr(lean_app, "DEFAULT_CRITIC_OUTPUT_PATH", critic_path)
    monkeypatch.setattr(lean_app, "_compile_then_critic", _fake_compile_then_critic)
    monkeypatch.setattr(lean_app, "_refresh_prompt_data", lambda: None)
    monkeypatch.setattr(lean_app, "PROMPTS", [prompt_text])
    monkeypatch.setattr(lean_app, "TOTAL_PROMPTS", 1)

    states = list(
        lean_app.send_prompt(
            "GPT",
            None,
            "gpt-5.4",
            "high",
            0,
            [],
        )
    )

    assert proof_path.read_text(encoding="utf-8") == expected_proof
    assert captured["critic_args"] == {
        "provider": "GPT",
        "session_id": "session-inline",
        "model": "gpt-5.4",
        "reasoning_effort": "high",
        "prompt_index": 0,
        "saved_outputs": [],
        "current_prompt": prompt_text,
        "previous_evaluated_code": None,
    }
    assert states[-1][1] == expected_proof


def test_load_prompt_qa_pairs_rebuilds_cache_when_old_c1_batch_template_is_present(tmp_path):
    qa_path = tmp_path / "qa_data.json"
    prompt_path = tmp_path / "lean_prompt_data.json"

    qa_path.write_text(
        json.dumps([[{"question": "q1", "answer": "a1"}]]),
        encoding="utf-8",
    )
    prompt_path.write_text(
        json.dumps(
            [
                (
                    "This prompt is for a single QA pair from a larger batch. "
                    "The target theorem must be named `C1` so it can be rebuilt into "
                    "the original batch of size 5."
                )
            ]
        ),
        encoding="utf-8",
    )

    entries = lean_app._load_prompt_qa_pairs(prompt_path=prompt_path, qa_path=qa_path)

    assert len(entries) == 1
    rebuilt_prompt = entries[0]["lean_prompt"]
    assert "This prompt covers exactly one QA pair." in rebuilt_prompt
    assert "must be named `C1`" not in rebuilt_prompt
    assert "batch of size 5" not in rebuilt_prompt


