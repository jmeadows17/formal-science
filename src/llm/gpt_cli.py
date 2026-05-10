"""
GPT CLI wrapper backed by the local Codex CLI.

This wrapper is intentionally shaped like claude_cli.py for the subset of
behavior this app needs:
  - one-shot prompts
  - persistent multi-turn sessions via native Codex thread resume
  - a streaming-like interface by tailing Codex's output file
  - agentic filesystem access within the workspace

Billing/auth behavior depends on how `codex` is logged in. When the local
Codex CLI is authenticated with ChatGPT, usage counts against ChatGPT/Codex
plan limits rather than OpenAI API token credits.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path


_CODEX_EMPTY_ERROR_HINT = (
    "Codex CLI exited without any error details. "
    "This usually indicates a local Codex/ChatGPT CLI rate limit, authentication problem, "
    "or another CLI-side failure before a final message was written."
)
_SANDBOX_RUNTIME_ERROR_MARKERS = (
    "bwrap",
    "bubblewrap",
    "sandbox launcher",
)


def _format_cli_error(prefix: str, returncode: int, *parts: str | None) -> str:
    details = []
    seen = set()
    for part in parts:
        text = (part or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        details.append(text)

    if details:
        return f"{prefix} exited with code {returncode}:\n" + "\n\n".join(details)
    return f"{prefix} exited with code {returncode}."


def _format_codex_error(returncode: int, *parts: str | None) -> str:
    message = _format_cli_error("Codex CLI", returncode, *parts)
    if message.endswith(f"code {returncode}."):
        return f"{message}\n{_CODEX_EMPTY_ERROR_HINT}"
    return message


def _extract_error_text(event: dict) -> str:
    candidates = [
        event.get("error"),
        event.get("message"),
        event.get("detail"),
    ]
    error_obj = event.get("error")
    if isinstance(error_obj, dict):
        candidates.extend([
            error_obj.get("message"),
            error_obj.get("detail"),
            error_obj.get("error"),
        ])

    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return ""


def _parse_jsonl(payload: str) -> tuple[list[dict], list[str]]:
    events: list[dict] = []
    raw_lines: list[str] = []
    for line in payload.splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            events.append(json.loads(text))
        except json.JSONDecodeError:
            raw_lines.append(text)
    return events, raw_lines


def _find_codex() -> str:
    """Locate the codex binary: PATH first, then VSCode extension fallback."""
    on_path = shutil.which("codex")
    if on_path:
        return on_path

    ext_base = Path.home() / ".vscode-server" / "extensions"
    if not ext_base.exists():
        ext_base = Path.home() / ".vscode" / "extensions"

    patterns = [
        "openai.chatgpt-*/bin/linux-x86_64/codex",
        "openai.chatgpt-*/bin/darwin-arm64/codex",
        "openai.chatgpt-*/bin/darwin-x86_64/codex",
        "openai.chatgpt-*/bin/win32-x64/codex.exe",
    ]
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(sorted(ext_base.glob(pattern)))

    if candidates:
        return str(candidates[-1])

    raise FileNotFoundError(
        "codex binary not found. Add it to PATH or install the OpenAI ChatGPT/Codex VSCode extension."
    )


CODEX_BIN = _find_codex()
VALID_REASONING_EFFORTS = {"minimal", "low", "medium", "high", "xhigh"}


class GPTSession:
    """
    A lightweight multi-turn GPT session implemented on top of `codex exec`.

    This mirrors the local Codex/VSCode agent more closely than transcript
    replay by preserving the real Codex thread ID and resuming it on later
    turns with `codex exec resume`.
    """

    def __init__(
        self,
        cwd: str | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        tools: list[str] | None = None,
        system_prompt: str | None = None,
        max_turns: int | None = None,
        verbose: bool = False,
        transcript: list[dict[str, str]] | None = None,
    ):
        self.cwd = str(cwd) if cwd else None
        self.model = model[0] if isinstance(model, list) else model
        effort = reasoning_effort[0] if isinstance(reasoning_effort, list) else reasoning_effort
        self.reasoning_effort = effort if effort in VALID_REASONING_EFFORTS else None
        self.tools = tools or []
        self.system_prompt = system_prompt[0] if isinstance(system_prompt, list) else system_prompt
        self.max_turns = max_turns
        self.verbose = verbose
        self.transcript = list(transcript or [])
        self.session_id: str | dict[str, list[dict[str, str]]] | None = None

    def _conversation_prompt(self, message: str) -> str:
        parts: list[str] = []
        if self.system_prompt:
            parts.append("System instructions:\n" + self.system_prompt.strip())
        if self.transcript and not isinstance(self.session_id, str):
            history = []
            for turn in self.transcript:
                role = turn.get("role", "user").strip().title()
                content = turn.get("content", "").strip()
                history.append(f"{role}:\n{content}")
            parts.append("Conversation so far:\n\n" + "\n\n".join(history))
        parts.append(message.strip())
        return "\n\n".join(parts)

    def _build_cmd(self, output_path: str, *, sandbox_mode: str | None = "workspace-write") -> list[str]:
        if isinstance(self.session_id, str) and self.session_id.strip():
            cmd = [
                CODEX_BIN,
                "exec",
                "resume",
                "--skip-git-repo-check",
                "--json",
                "--output-last-message",
                output_path,
            ]
        else:
            cmd = [
                CODEX_BIN,
                "exec",
                "--skip-git-repo-check",
                "--color",
                "never",
                "--json",
                "--output-last-message",
                output_path,
            ]
            if sandbox_mode:
                cmd.extend(["--sandbox", sandbox_mode])
        if self.model:
            cmd.extend(["--model", self.model])
        if self.reasoning_effort:
            cmd.extend(["-c", f'model_reasoning_effort="{self.reasoning_effort}"'])
        if not isinstance(self.session_id, str) and sandbox_mode == "workspace-write":
            for writable_tool_dir in self.tools:
                if writable_tool_dir:
                    cmd.extend(["--add-dir", str(writable_tool_dir)])

        if isinstance(self.session_id, str) and self.session_id.strip():
            cmd.extend([self.session_id, "-"])
        else:
            cmd.append("-")
        return cmd

    def _is_sandbox_runtime_failure(self, *parts: str | None) -> bool:
        combined = "\n".join((part or "") for part in parts).lower()
        return any(marker in combined for marker in _SANDBOX_RUNTIME_ERROR_MARKERS)

    def _read_output(self, output_path: str) -> str:
        path = Path(output_path)
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def _update_transcript_fallback(self, user_message: str, assistant_message: str) -> None:
        self.transcript.extend(
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message},
            ]
        )
        if not isinstance(self.session_id, str):
            self.session_id = {"transcript": list(self.transcript)}

    def _apply_exec_events(self, stdout_text: str) -> tuple[list[str], dict | None]:
        events, raw_stdout_lines = _parse_jsonl(stdout_text)
        usage = None
        for event in events:
            if event.get("type") == "thread.started":
                thread_id = event.get("thread_id")
                if isinstance(thread_id, str) and thread_id.strip():
                    self.session_id = thread_id
            if usage is None and isinstance(event.get("usage"), dict):
                usage = event.get("usage")
            if not isinstance(self.session_id, str):
                sid = event.get("session_id") or event.get("thread_id")
                if isinstance(sid, str) and sid.strip():
                    self.session_id = sid
        return raw_stdout_lines, usage

    def prompt(self, message: str, timeout: float | None = None) -> dict:
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            output_path = tmp.name

        try:
            prompt = self._conversation_prompt(message)
            cmd = self._build_cmd(output_path)

            import sys
            prompt_len = len(prompt)
            print(
                f"[gpt_cli] prompt (non-stream) chars={prompt_len}  "
                f"session={self.session_id or 'new'}  output_path={output_path}",
                file=sys.stderr,
                flush=True,
            )

            result = subprocess.run(
                cmd, input=prompt, capture_output=True, text=True, cwd=self.cwd,
                timeout=timeout,
            )
            raw_stdout_lines, usage = self._apply_exec_events(result.stdout)

            if (
                result.returncode != 0
                and not isinstance(self.session_id, str)
                and self._is_sandbox_runtime_failure(
                    result.stderr,
                    result.stdout,
                    self._read_output(output_path),
                )
            ):
                print(
                    "[gpt_cli] workspace-write sandbox unavailable; retrying without sandbox restrictions",
                    file=sys.stderr,
                    flush=True,
                )
                cmd = self._build_cmd(output_path, sandbox_mode="danger-full-access")
                result = subprocess.run(
                    cmd, input=prompt, capture_output=True, text=True, cwd=self.cwd,
                    timeout=timeout,
                )
                raw_stdout_lines, usage = self._apply_exec_events(result.stdout)

            if result.returncode != 0:
                error_events, _ = _parse_jsonl(result.stdout)
                raise RuntimeError(_format_codex_error(
                    result.returncode,
                    result.stderr,
                    "\n".join(_extract_error_text(event) for event in error_events if _extract_error_text(event)),
                    "\n".join(raw_stdout_lines),
                    self._read_output(output_path),
                ))

            text = self._read_output(output_path).strip()
            print(
                f"[gpt_cli] done (non-stream)  output_chars={len(text)}",
                file=sys.stderr,
                flush=True,
            )
            if not isinstance(self.session_id, str):
                self._update_transcript_fallback(message, text)
            return {"result": text, "session_id": self.session_id, "usage": usage}
        except subprocess.TimeoutExpired:
            timeout_text = "with no timeout value provided" if timeout is None else f"after {timeout:.0f}s"
            raise RuntimeError(
                f"Codex CLI timed out {timeout_text}. Prompt was {prompt_len} chars."
            )
        finally:
            Path(output_path).unlink(missing_ok=True)

    def text(self, message: str) -> str:
        return self.prompt(message).get("result", "")

    def prompt_stream(self, message: str, timeout: float | None = None):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            output_path = tmp.name

        try:
            prompt = self._conversation_prompt(message)
            cmd = self._build_cmd(output_path)

            import sys
            prompt_len = len(prompt)
            cmd_len = sum(len(a) for a in cmd)
            print(
                f"[gpt_cli] prompt chars={prompt_len}  cmd args={len(cmd)}  "
                f"session={self.session_id or 'new'}  "
                f"cmd total chars={cmd_len}  output_path={output_path}",
                file=sys.stderr,
                flush=True,
            )

            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.cwd,
            )
            assert proc.stdin is not None
            proc.stdin.write(prompt)
            proc.stdin.close()

            raw_stdout_lines: list[str] = []
            stderr_lines: list[str] = []
            error_events: list[str] = []
            usage_holder: dict[str, dict | None] = {"usage": None}

            def drain_stdout() -> None:
                assert proc.stdout is not None
                for line in proc.stdout:
                    text = line.strip()
                    if not text:
                        continue
                    try:
                        event = json.loads(text)
                    except json.JSONDecodeError:
                        raw_stdout_lines.append(text)
                        continue

                    if event.get("type") == "thread.started":
                        thread_id = event.get("thread_id")
                        if isinstance(thread_id, str) and thread_id.strip():
                            self.session_id = thread_id
                    elif not isinstance(self.session_id, str):
                        sid = event.get("session_id") or event.get("thread_id")
                        if isinstance(sid, str) and sid.strip():
                            self.session_id = sid

                    if usage_holder["usage"] is None and isinstance(event.get("usage"), dict):
                        usage_holder["usage"] = event.get("usage")

                    if event.get("type") == "error":
                        error_text = _extract_error_text(event)
                        if error_text:
                            error_events.append(error_text)

            def drain_stderr() -> None:
                assert proc.stderr is not None
                for line in proc.stderr:
                    stderr_lines.append(line.rstrip())

            stdout_thread = threading.Thread(target=drain_stdout, daemon=True)
            stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
            stdout_thread.start()
            stderr_thread.start()

            seen = ""
            start = time.monotonic()
            last_progress = start
            polls = 0
            while proc.poll() is None:
                elapsed = time.monotonic() - start
                if timeout is not None and elapsed > timeout:
                    proc.kill()
                    proc.wait()
                    raise RuntimeError(
                        f"Codex CLI timed out after {timeout:.0f}s with no output. "
                        f"Prompt was {prompt_len} chars. Output file had {len(seen)} chars."
                    )

                current = self._read_output(output_path)
                if current.startswith(seen):
                    delta = current[len(seen):]
                else:
                    delta = current
                if delta:
                    seen = current
                    last_progress = time.monotonic()
                    yield delta
                else:
                    polls += 1
                    if polls % 50 == 0:
                        idle = time.monotonic() - last_progress
                        print(
                            f"[gpt_cli] waiting... elapsed={elapsed:.1f}s  "
                            f"idle={idle:.1f}s  output_size={len(seen)}  "
                            f"proc_alive={proc.poll() is None}",
                            file=sys.stderr,
                            flush=True,
                        )
                time.sleep(0.1)

            elapsed = time.monotonic() - start
            final_text = self._read_output(output_path)
            if final_text.startswith(seen):
                final_delta = final_text[len(seen):]
            else:
                final_delta = final_text
            if final_delta:
                yield final_delta

            print(
                f"[gpt_cli] done  rc={proc.returncode}  elapsed={elapsed:.1f}s  "
                f"output_chars={len(final_text)}",
                file=sys.stderr,
                flush=True,
            )

            stdout_thread.join(timeout=1.0)
            stderr_thread.join(timeout=1.0)

            if proc.returncode != 0:
                raise RuntimeError(_format_codex_error(
                    proc.returncode,
                    "\n".join(stderr_lines),
                    "\n".join(error_events),
                    "\n".join(raw_stdout_lines),
                    final_text,
                ))

            if not isinstance(self.session_id, str):
                self._update_transcript_fallback(message, final_text.strip())
        finally:
            Path(output_path).unlink(missing_ok=True)

    @classmethod
    def resume(cls, session_id, **kwargs) -> "GPTSession":
        transcript = []
        session = cls(**kwargs)
        if isinstance(session_id, str) and session_id.strip():
            session.session_id = session_id
            return session
        if isinstance(session_id, dict):
            sid = session_id.get("thread_id") or session_id.get("session_id")
            if isinstance(sid, str) and sid.strip():
                session.session_id = sid
                return session
            transcript = session_id.get("transcript", []) or []
        session.transcript = list(transcript)
        if session.transcript:
            session.session_id = {"transcript": list(session.transcript)}
        return session


def prompt(message: str, **kwargs) -> dict:
    return GPTSession(**kwargs).prompt(message)


def chat(
    cwd: str | None = None,
    model: str | None = None,
    tools: list[str] | None = None,
) -> None:
    session = GPTSession(cwd=cwd, model=model, tools=tools)
    print("GPT CLI chat (type 'exit' to quit)\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue

        try:
            print(f"\nGPT: {session.text(user_input)}\n")
        except RuntimeError as e:
            print(f"Error: {e}")
