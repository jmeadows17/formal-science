"""
Claude CLI wrapper that mirrors the VSCode Claude Code extension behavior.

Key parity with VSCode:
  - Multi-turn sessions by default (conversation state persists across calls)
  - Context window auto-compaction handled by the CLI internally (same as VSCode)
  - Rate limits shared with VSCode (same account pool); CLI auto-retries with
    backoff just like VSCode does — no client-side retry logic needed
  - Reads CLAUDE.md, .claude/settings.json, .claude/rules/, MCP servers
    automatically — same as VSCode
  - Sessions stored in ~/.claude/sessions/ — same location as VSCode
"""

import subprocess
import json
import shutil
from pathlib import Path


_CLAUDE_EMPTY_ERROR_HINT = (
    "Claude CLI exited without any error details. "
    "This usually indicates a local Claude CLI rate limit, authentication problem, "
    "or another CLI-side failure before a structured response was emitted."
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


def _format_claude_error(returncode: int, *parts: str | None) -> str:
    message = _format_cli_error("Claude CLI", returncode, *parts)
    if message.endswith(f"code {returncode}."):
        return f"{message}\n{_CLAUDE_EMPTY_ERROR_HINT}"
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


def _find_claude() -> str:
    """Locate the claude binary: PATH first, then VSCode extension fallback."""
    on_path = shutil.which("claude")
    if on_path:
        return on_path

    # VSCode bundles claude inside the extension directory
    ext_base = Path.home() / ".vscode-server" / "extensions"
    if not ext_base.exists():
        ext_base = Path.home() / ".vscode" / "extensions"  # non-server VSCode

    candidates = sorted(ext_base.glob("anthropic.claude-code-*/resources/native-binary/claude"))
    if candidates:
        return str(candidates[-1])  # latest version

    raise FileNotFoundError(
        "claude binary not found. Add it to PATH or install the Claude Code VSCode extension."
    )


CLAUDE_BIN = _find_claude()
CLAUDE_REASONING_EFFORTS = ("low", "medium", "high", "max")


class ClaudeSession:
    """
    A persistent multi-turn Claude session, mirroring VSCode's chat panel.

    In VSCode, every message is part of an ongoing conversation. This class
    does the same: the first call starts a session, and all subsequent calls
    continue it. Context window compaction and rate-limit retries are handled
    internally by the CLI (identical to VSCode).
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
    ):
        """
        Args:
            cwd:            Working directory (like VSCode's workspace folder).
            model:          Model to use: "sonnet", "opus", "haiku", or a full
                            model ID. None uses the default (same as VSCode).
            reasoning_effort:
                            Claude reasoning effort: "low", "medium",
                            "high", or "max". None uses the CLI default.
            tools:          Tools to auto-approve, e.g. ["Read", "Edit", "Bash"].
                            Supports patterns like "Bash(git diff *)".
                            None = default tool permissions from settings.
            system_prompt:  Extra text appended to the built-in system prompt.
                            CLAUDE.md is always loaded automatically (same as VSCode).
            max_turns:      Max agentic turns per prompt. None = unlimited
                            (same as VSCode).
            verbose:        If True, passes --verbose for full turn-by-turn output.
        """
        self.cwd = str(cwd) if cwd else None
        self.model = model[0] if isinstance(model, list) else model
        effort = reasoning_effort[0] if isinstance(reasoning_effort, list) else reasoning_effort
        self.reasoning_effort = effort if effort in CLAUDE_REASONING_EFFORTS else None
        self.tools = tools
        self.system_prompt = system_prompt[0] if isinstance(system_prompt, list) else system_prompt
        self.max_turns = max_turns
        self.verbose = verbose
        self.session_id: str | None = None

    def _build_cmd(self, message: str) -> list[str]:
        cmd = [CLAUDE_BIN, "-p", message, "--output-format", "json"]

        # Resume existing session (multi-turn), matching VSCode's behavior
        # where every message continues the conversation.
        if self.session_id:
            cmd.extend(["--resume", self.session_id])

        if self.model:
            cmd.extend(["--model", self.model])
        if self.reasoning_effort:
            cmd.extend(["--effort", self.reasoning_effort])

        if self.tools is not None:
            for tool in self.tools:
                cmd.extend(["--allowedTools", tool])

        if self.system_prompt:
            cmd.extend(["--append-system-prompt", self.system_prompt])

        if self.max_turns is not None:
            cmd.extend(["--max-turns", str(self.max_turns)])

        if self.verbose:
            cmd.append("--verbose")

        return cmd

    def prompt(self, message: str, timeout: float | None = None) -> dict:
        """
        Send a message in this session, returns a normalized dict response.

        Handles both Claude CLI JSON output shapes:
        1. A single dict with keys like `result` and `session_id`
        2. A list of event objects, where the final useful payload is a
            `{"type": "result", ...}` event

        Returns:
            dict with at least:
            - result
            - session_id
            - raw_response

        Raises:
            RuntimeError: If the CLI exits with a non-zero code after
                        exhausting its internal retries.
        """
        import sys
        import time as _time

        cmd = self._build_cmd(message)
        prompt_len = len(message)
        print(
            f"[claude_cli] prompt (non-stream) chars={prompt_len}  "
            f"session={self.session_id or 'new'}",
            file=sys.stderr,
            flush=True,
        )

        start = _time.monotonic()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                stdin=subprocess.DEVNULL,
                text=True,
                cwd=self.cwd,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            timeout_text = "with no timeout value provided" if timeout is None else f"after {timeout:.0f}s"
            raise RuntimeError(
                f"Claude CLI timed out {timeout_text}. Prompt was {prompt_len} chars."
            )

        elapsed = _time.monotonic() - start

        if result.returncode != 0:
            print(
                f"[claude_cli] failed  rc={result.returncode}  elapsed={elapsed:.1f}s",
                file=sys.stderr,
                flush=True,
            )
            raise RuntimeError(_format_claude_error(
                result.returncode,
                result.stderr,
                result.stdout,
            ))

        try:
            parsed = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Claude CLI returned non-JSON output:\n{result.stdout}"
            ) from exc

        normalized: dict

        if isinstance(parsed, dict):
            # Old/simple shape
            result_text = parsed.get("result", "")
            self.session_id = parsed.get("session_id", self.session_id)
            normalized = dict(parsed)
            normalized.setdefault("result", result_text)
            normalized.setdefault("session_id", self.session_id)
            normalized["raw_response"] = parsed

        elif isinstance(parsed, list):
            # Event-list shape used by your installed Claude CLI
            final_result_event = None
            result_text_parts: list[str] = []

            for item in parsed:
                if not isinstance(item, dict):
                    continue

                # Track latest session_id anywhere it appears
                sid = item.get("session_id")
                if sid:
                    self.session_id = sid

                item_type = item.get("type")

                if item_type == "assistant":
                    message_obj = item.get("message", {})
                    if isinstance(message_obj, dict):
                        for block in message_obj.get("content", []):
                            if not isinstance(block, dict):
                                continue
                            if block.get("type") == "text":
                                text = block.get("text", "")
                                if text:
                                    result_text_parts.append(text)

                if item_type == "result":
                    final_result_event = item

            if final_result_event is not None:
                result_text = final_result_event.get("result", "")
                self.session_id = final_result_event.get("session_id", self.session_id)
                normalized = dict(final_result_event)
                normalized.setdefault("result", result_text)
                normalized.setdefault("session_id", self.session_id)
                normalized["raw_response"] = parsed
            else:
                result_text = "\n".join(part for part in result_text_parts if part).strip()
                normalized = {
                    "type": "result",
                    "subtype": "synthetic",
                    "result": result_text,
                    "session_id": self.session_id,
                    "raw_response": parsed,
                }

        else:
            raise RuntimeError(
                f"Unexpected Claude CLI JSON type: {type(parsed).__name__}"
            )

        print(
            f"[claude_cli] done (non-stream)  elapsed={elapsed:.1f}s  "
            f"output_chars={len(normalized.get('result', ''))}",
            file=sys.stderr,
            flush=True,
        )

        return normalized

    def text(self, message: str) -> str:
        """Send a message and return just the text response."""
        return self.prompt(message).get("result", "")

    def _build_stream_cmd(self, message: str) -> list[str]:
        """Build command with stream-json output format (requires --verbose)."""
        cmd = [CLAUDE_BIN, "-p", message, "--output-format", "stream-json", "--verbose"]

        if self.session_id:
            cmd.extend(["--resume", self.session_id])
        if self.model:
            cmd.extend(["--model", self.model])
        if self.reasoning_effort:
            cmd.extend(["--effort", self.reasoning_effort])
        if self.tools is not None:
            for tool in self.tools:
                cmd.extend(["--allowedTools", tool])
        if self.system_prompt:
            cmd.extend(["--append-system-prompt", self.system_prompt])
        if self.max_turns is not None:
            cmd.extend(["--max-turns", str(self.max_turns)])

        return cmd

    def prompt_stream(self, message: str, timeout: float | None = None):
        """
        Send a message and yield text chunks as they arrive.

        After iteration completes, self.session_id is updated.
        """
        import sys, time as _time

        cmd = self._build_stream_cmd(message)
        prompt_len = len(message)
        print(
            f"[claude_cli] prompt_stream chars={prompt_len}  "
            f"session={self.session_id or 'new'}",
            file=sys.stderr,
            flush=True,
        )

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL, text=True, cwd=self.cwd,
        )
        raw_stdout_lines: list[str] = []
        error_events: list[str] = []

        start = _time.monotonic()
        last_yield = start
        event_count = 0
        yielded_chars = 0

        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue

            elapsed = _time.monotonic() - start
            idle = _time.monotonic() - last_yield
            if idle > 30.0:
                print(
                    f"[claude_cli] no text yielded for {idle:.1f}s  "
                    f"elapsed={elapsed:.1f}s  events={event_count}  "
                    f"yielded_chars={yielded_chars}",
                    file=sys.stderr,
                    flush=True,
                )

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                raw_stdout_lines.append(line)
                continue

            event_count += 1
            etype = event.get("type", "")

            if event_count <= 5 or event_count % 50 == 0:
                print(
                    f"[claude_cli] event #{event_count} type={etype}  "
                    f"elapsed={elapsed:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )

            if etype == "assistant":
                message_obj = event.get("message", {})
                for block in message_obj.get("content", []):
                    if block.get("type") == "text":
                        text = block["text"]
                        yielded_chars += len(text)
                        last_yield = _time.monotonic()
                        yield text
                sid = event.get("session_id")
                if sid:
                    self.session_id = sid
            elif etype == "content_block_delta":
                delta = event.get("delta", {})
                if delta.get("type") == "text_delta":
                    text = delta.get("text", "")
                    yielded_chars += len(text)
                    last_yield = _time.monotonic()
                    yield text
            elif etype == "tool_use":
                tool_name = event.get("name", "unknown_tool")
                last_yield = _time.monotonic()
                yield f"\n\n*[Claude is using the `{tool_name}` tool...]*\n\n"
            elif etype == "tool_result":
                last_yield = _time.monotonic()
                yield f"\n*[Tool execution complete]*\n\n"
            elif etype == "result":
                self.session_id = event.get("session_id", self.session_id)
            elif etype == "error":
                error_text = _extract_error_text(event)
                if error_text:
                    error_events.append(error_text)
                    print(
                        f"[claude_cli] error event: {error_text}",
                        file=sys.stderr,
                        flush=True,
                    )
                    last_yield = _time.monotonic()
                    yield f"\n\n*[Claude Error: {error_text}]*\n\n"

        proc.wait()
        elapsed = _time.monotonic() - start
        print(
            f"[claude_cli] stream done  rc={proc.returncode}  elapsed={elapsed:.1f}s  "
            f"events={event_count}  yielded_chars={yielded_chars}",
            file=sys.stderr,
            flush=True,
        )

        if proc.returncode != 0:
            stderr = proc.stderr.read()
            raise RuntimeError(_format_claude_error(
                proc.returncode,
                stderr,
                "\n".join(error_events),
                "\n".join(raw_stdout_lines),
            ))

    @classmethod
    def resume(cls, session_id: str, **kwargs) -> "ClaudeSession":
        """Resume an existing session (e.g. one started in VSCode or a prior run)."""
        session = cls(**kwargs)
        session.session_id = session_id
        return session


def prompt(message: str, **kwargs) -> dict:
    """One-shot convenience: sends a single message with no session continuity."""
    return ClaudeSession(**kwargs).prompt(message)


def chat(
    cwd: str | None = None,
    model: str | None = None,
    tools: list[str] | None = None,
) -> None:
    """
    Interactive multi-turn chat loop, mirroring VSCode's chat panel.
    Type 'exit' or Ctrl-C to quit.
    """
    session = ClaudeSession(cwd=cwd, model=model, tools=tools)
    print("Claude CLI chat (type 'exit' to quit)\n")

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
            print(f"\nClaude: {session.text(user_input)}\n")
        except RuntimeError as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    # Example: multi-turn session (mirrors VSCode behavior)
    session = ClaudeSession(tools=["Read", "Bash"])

    r1 = session.prompt("What files are in the current directory?")
    print(r1.get("result"))

    # This continues the same conversation — Claude remembers the previous turn
    r2 = session.prompt("Which of those files is the most interesting and why?")
    print(r2.get("result"))
