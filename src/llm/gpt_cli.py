"""
GPT CLI wrapper backed by the local Codex CLI.

This wrapper is intentionally shaped like claude_cli.py for the subset of
behavior this app needs:
  - one-shot prompts
  - pseudo-persistent multi-turn sessions via in-memory transcript replay
  - a streaming-like interface by tailing Codex's output file

Billing/auth behavior depends on how `codex` is logged in. When the local
Codex CLI is authenticated with ChatGPT, usage counts against ChatGPT/Codex
plan limits rather than OpenAI API token credits.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from pathlib import Path


_CODEX_EMPTY_ERROR_HINT = (
    "Codex CLI exited without any error details. "
    "This usually indicates a local Codex/ChatGPT CLI rate limit, authentication problem, "
    "or another CLI-side failure before a final message was written."
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


class GPTSession:
    """
    A lightweight multi-turn GPT session implemented on top of `codex exec`.

    Codex exec is naturally one-shot, so this class persists a simple
    user/assistant transcript and replays it into each new request.
    """

    def __init__(
        self,
        cwd: str | None = None,
        model: str | None = None,
        tools: list[str] | None = None,
        system_prompt: str | None = None,
        max_turns: int | None = None,
        verbose: bool = False,
        transcript: list[dict[str, str]] | None = None,
    ):
        self.cwd = str(cwd) if cwd else None
        self.model = model[0] if isinstance(model, list) else model
        self.tools = tools or []
        self.system_prompt = system_prompt[0] if isinstance(system_prompt, list) else system_prompt
        self.max_turns = max_turns
        self.verbose = verbose
        self.transcript = list(transcript or [])
        self.session_id: dict[str, list[dict[str, str]]] | None = (
            {"transcript": list(self.transcript)} if self.transcript else None
        )

    def _conversation_prompt(self, message: str) -> str:
        parts: list[str] = []
        if self.system_prompt:
            parts.append("System instructions:\n" + self.system_prompt.strip())
        if self.transcript:
            history = []
            for turn in self.transcript:
                role = turn.get("role", "user").strip().title()
                content = turn.get("content", "").strip()
                history.append(f"{role}:\n{content}")
            parts.append("Conversation so far:\n\n" + "\n\n".join(history))
        parts.append("User:\n" + message.strip())
        parts.append("Assistant:")
        return "\n\n".join(parts)

    def _build_cmd(self, prompt: str, output_path: str) -> list[str]:
        cmd = [
            CODEX_BIN,
            "exec",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--color",
            "never",
            "--output-last-message",
            output_path,
        ]
        if self.cwd:
            cmd.extend(["-C", self.cwd])
        if self.model:
            cmd.extend(["--model", self.model])
        cmd.extend(["--", prompt])
        return cmd

    def _read_output(self, output_path: str) -> str:
        path = Path(output_path)
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def _update_transcript(self, user_message: str, assistant_message: str) -> None:
        self.transcript.extend(
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message},
            ]
        )
        self.session_id = {"transcript": list(self.transcript)}

    def prompt(self, message: str, timeout: float | None = None) -> dict:
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            output_path = tmp.name

        try:
            prompt = self._conversation_prompt(message)
            cmd = self._build_cmd(prompt, output_path)

            import sys
            prompt_len = len(prompt)
            print(
                f"[gpt_cli] prompt (non-stream) chars={prompt_len}  output_path={output_path}",
                file=sys.stderr,
                flush=True,
            )

            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=self.cwd,
                timeout=timeout,
            )

            if result.returncode != 0:
                raise RuntimeError(_format_codex_error(
                    result.returncode,
                    result.stderr,
                    result.stdout,
                    self._read_output(output_path),
                ))

            text = self._read_output(output_path).strip()
            print(
                f"[gpt_cli] done (non-stream)  output_chars={len(text)}",
                file=sys.stderr,
                flush=True,
            )
            self._update_transcript(message, text)
            return {"result": text, "session_id": self.session_id, "usage": None}
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
            cmd = self._build_cmd(prompt, output_path)

            import sys
            prompt_len = len(prompt)
            cmd_len = sum(len(a) for a in cmd)
            print(
                f"[gpt_cli] prompt chars={prompt_len}  cmd args={len(cmd)}  "
                f"cmd total chars={cmd_len}  output_path={output_path}",
                file=sys.stderr,
                flush=True,
            )

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.cwd,
            )

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

            if proc.returncode != 0:
                stdout, stderr = proc.communicate()
                raise RuntimeError(_format_codex_error(
                    proc.returncode,
                    stderr,
                    stdout,
                    final_text,
                ))

            self._update_transcript(message, final_text.strip())
        finally:
            Path(output_path).unlink(missing_ok=True)

    @classmethod
    def resume(cls, session_id, **kwargs) -> "GPTSession":
        transcript = []
        if isinstance(session_id, dict):
            transcript = session_id.get("transcript", []) or []
        return cls(transcript=transcript, **kwargs)


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
