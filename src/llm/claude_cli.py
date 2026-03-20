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

    def prompt(self, message: str) -> dict:
        """
        Send a message in this session, returns the parsed JSON response.

        Mirrors typing a message in VSCode's Claude panel:
          - First call creates a new session
          - Subsequent calls continue the same session (multi-turn)
          - Context compaction happens automatically when the window fills
          - Rate limits trigger automatic retries with backoff (like VSCode)

        Returns:
            dict with keys: result, session_id, usage, etc.

        Raises:
            RuntimeError: If the CLI exits with a non-zero code after
                          exhausting its internal retries.
        """
        cmd = self._build_cmd(message)
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.cwd)

        if result.returncode != 0:
            raise RuntimeError(
                f"Claude CLI exited with code {result.returncode}:\n"
                f"{result.stderr or result.stdout}"
            )

        response = json.loads(result.stdout)
        # Persist session ID so the next call continues this conversation.
        self.session_id = response.get("session_id", self.session_id)
        return response

    def text(self, message: str) -> str:
        """Send a message and return just the text response."""
        return self.prompt(message).get("result", "")

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
