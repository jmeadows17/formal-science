"""
Paste text into the Codex VS Code extension via local UI automation.

This is not a Codex API integration. It controls the local desktop by:
  1. copying text into the system clipboard
  2. activating VS Code if possible
  3. focusing the Codex input via a shortcut or a screen click
  4. sending the paste shortcut and optionally Enter

Important limitation:
  - This must run on the same machine that owns the VS Code GUI session.
    It will not control a local VS Code window when executed on a remote
    SSH/WSL/container host.
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import time
from pathlib import Path


def _read_text(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text
    if args.file is not None:
        return Path(args.file).read_text(encoding="utf-8")
    if args.stdin:
        return sys.stdin.read()
    raise ValueError("Provide one of --text, --file, or --stdin.")


def _copy_to_clipboard(text: str) -> None:
    try:
        import pyperclip
    except ImportError as exc:
        raise RuntimeError(
            "pyperclip is required. Install it with: pip install pyperclip"
        ) from exc

    pyperclip.copy(text)


def _load_pyautogui():
    try:
        import pyautogui
    except ImportError as exc:
        raise RuntimeError(
            "pyautogui is required. Install it with: pip install pyautogui"
        ) from exc

    pyautogui.FAILSAFE = True
    return pyautogui


def _activate_vscode(title_hint: str) -> bool:
    system = platform.system()

    if system == "Darwin":
        script = 'tell application "Visual Studio Code" to activate'
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0

    if system == "Linux":
        result = subprocess.run(
            ["xdotool", "search", "--name", title_hint, "windowactivate"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0

    if system == "Windows":
        ps = (
            "$wshell = New-Object -ComObject WScript.Shell; "
            f'$wshell.AppActivate("{title_hint}")'
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0

    return False


def _focus_input(pyautogui, args: argparse.Namespace) -> None:
    if args.focus_shortcut:
        keys = [part.strip() for part in args.focus_shortcut.split("+") if part.strip()]
        if not keys:
            raise ValueError("--focus-shortcut was provided but no keys were parsed.")
        pyautogui.hotkey(*keys)
        return

    if args.click_x is not None and args.click_y is not None:
        pyautogui.click(args.click_x, args.click_y)
        return

    raise ValueError(
        "Provide either --focus-shortcut or both --click-x and --click-y "
        "so the script can focus the Codex input."
    )


def _paste(pyautogui) -> None:
    if platform.system() == "Darwin":
        pyautogui.hotkey("command", "v")
        return
    pyautogui.hotkey("ctrl", "v")


def _submit(pyautogui) -> None:
    pyautogui.press("enter")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paste text into the Codex VS Code extension using UI automation."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Literal text to paste.")
    source.add_argument("--file", help="Read paste text from a file.")
    source.add_argument(
        "--stdin",
        action="store_true",
        help="Read paste text from stdin.",
    )
    parser.add_argument(
        "--focus-shortcut",
        help=(
            "Shortcut to focus the Codex input, for example "
            "'ctrl+shift+i'. Preferred over screen coordinates."
        ),
    )
    parser.add_argument("--click-x", type=int, help="X coordinate of the Codex input box.")
    parser.add_argument("--click-y", type=int, help="Y coordinate of the Codex input box.")
    parser.add_argument(
        "--activate-vscode",
        action="store_true",
        help="Try to bring the VS Code window to the foreground before pasting.",
    )
    parser.add_argument(
        "--vscode-title",
        default="Visual Studio Code",
        help="Window title hint used when activating VS Code.",
    )
    parser.add_argument(
        "--settle-delay",
        type=float,
        default=0.6,
        help="Seconds to wait between activation/focus/paste steps.",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Press Enter after pasting.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    text = _read_text(args)
    if not text:
        raise RuntimeError("Refusing to paste empty text.")

    _copy_to_clipboard(text)
    pyautogui = _load_pyautogui()

    if args.activate_vscode:
        activated = _activate_vscode(args.vscode_title)
        if not activated:
            print(
                "warning: could not activate VS Code automatically; continuing anyway",
                file=sys.stderr,
            )
        time.sleep(args.settle_delay)

    _focus_input(pyautogui, args)
    time.sleep(args.settle_delay)
    _paste(pyautogui)

    if args.submit:
        time.sleep(args.settle_delay)
        _submit(pyautogui)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
