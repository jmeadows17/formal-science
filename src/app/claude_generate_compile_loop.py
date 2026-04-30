from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT_DEFAULT = Path.cwd()
DEFAULT_PROMPT_FILE = Path("lean_prompt_data.json")


def _ensure_import_paths(repo_root: Path) -> None:
    """Add repo-local source folders so imports resolve from the project."""
    sys.path.insert(0, str(repo_root / "src" / "llm"))
    sys.path.insert(0, str(repo_root / "src" / "app"))


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _resolve_path(path: Path, repo_root: Path) -> Path:
    """
    Resolve a path robustly against likely repo locations.

    Search order:
    1. Absolute path as-is
    2. Relative to repo root
    3. Relative to this script's directory
    4. Relative to repo_root/src/app
    5. Relative to repo_root/src/app_data
    6. Relative to repo_root/data
    """
    candidates: list[Path] = []

    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend(
            [
                repo_root / path,
                SCRIPT_DIR / path,
                repo_root / "src" / "app" / path,
                repo_root / "src" / "app_data" / path,
                repo_root / "data" / path,
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    searched = "\n".join(str(p) for p in candidates)
    raise SystemExit(f"Could not find prompt file. Searched:\n{searched}")


def _load_prompt_list(path: Path) -> list[str]:
    """Load a JSON file containing a top-level list of prompt strings."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in prompt file {path}: {exc}") from exc

    if not isinstance(raw, list):
        raise SystemExit(f"Prompt file must contain a top-level JSON list: {path}")

    if not all(isinstance(item, str) for item in raw):
        raise SystemExit(f"Prompt file must contain only strings: {path}")

    return raw


CODE_ONLY_SUFFIX = """
# Task
Write the complete Lean 4 source code directly to `FSLean/proof.lean`.
Do not write to any other file.
Do not output the code as part of your response.
Stop immediately after writing the file.
""".strip()

COMPILE_FIX_SUFFIX = """
The Lean file did not compile.

Fix `FSLean/proof.lean` in place to resolve the compiler errors below.
Do not write to any other file.
Stop immediately after writing the updated file.
""".strip()


def _build_initial_message(prompt_text: str) -> str:
    return f"{prompt_text.rstrip()}\n\n{CODE_ONLY_SUFFIX}\n"


def _format_compile_output(returncode: int, stdout: str, stderr: str, iteration: int) -> str:
    sections = [f"Iteration: {iteration}", f"Exit code: {returncode}"]
    stdout = stdout.strip()
    stderr = stderr.strip()

    if returncode == 0:
        sections.append("Compilation succeeded.")
        if stdout:
            sections.append(f"STDOUT:\n{stdout}")
        if stderr:
            sections.append(f"STDERR:\n{stderr}")
        return "\n\n".join(sections)

    current_error = stderr or stdout or "No compiler output."
    sections.append(f"Current error:\n{current_error}")
    if stdout and stdout != current_error:
        sections.append(f"STDOUT:\n{stdout}")
    if stderr and stderr != current_error:
        sections.append(f"STDERR:\n{stderr}")
    return "\n\n".join(sections)


def _build_compile_fix_message(code: str, compiler_output: str, iteration: int) -> str:
    return (
        f"{COMPILE_FIX_SUFFIX}\n\n"
        f"Compile attempt: {iteration}\n\n"
        f"Compiler output:\n{compiler_output}\n\n"
        "Current Lean code:\n"
        "```lean\n"
        f"{code.rstrip()}\n"
        "```"
    )


def _read_proof_source(proof_path: Path) -> str:
    try:
        return proof_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _delete_proof_source(proof_path: Path) -> None:
    proof_path.unlink(missing_ok=True)


def run_loop(
    prompt_text: str,
    repo_root: Path,
    model: str,
    reasoning_effort: str,
    max_fix_attempts: int,
) -> int:
    _ensure_import_paths(repo_root)

    from claude_cli import ClaudeSession
    from compile_lean import compile_lean

    proof_path = repo_root / "FSLean" / "proof.lean"
    proof_path.parent.mkdir(parents=True, exist_ok=True)
    _delete_proof_source(proof_path)

    session = ClaudeSession(
        cwd=str(repo_root),
        model=model,
        reasoning_effort=reasoning_effort,
        tools=["Bash", "Edit", "Read", "Write", "Replace"],
        max_turns=10,
        verbose=True,
    )

    print(f"Repo root: {repo_root}")
    print(f"Proof path: {proof_path}")
    print("\n=== INITIAL GENERATION ===")
    initial_response = session.prompt(_build_initial_message(prompt_text))
    print(initial_response.get("result", ""))

    current_code = _read_proof_source(proof_path)
    if not current_code.strip():
        print("\nERROR: Claude did not write FSLean/proof.lean")
        return 1

    print("\n=== INITIAL proof.lean CONTENTS ===")
    print(current_code)

    for iteration in range(1, max_fix_attempts + 1):
        print(f"\n=== COMPILE ITERATION {iteration} ===")
        returncode, stdout, stderr = compile_lean(
            current_code,
            repo_root=repo_root,
            lean_folder_name="FSLean",
        )
        compile_text = _format_compile_output(returncode, stdout, stderr, iteration)
        print(compile_text)

        if returncode == 0:
            print("\nSUCCESS: proof.lean compiles.")
            return 0

        if returncode != 1:
            print("\nERROR: Unexpected compile return code.")
            return returncode

        fix_message = _build_compile_fix_message(current_code, compile_text, iteration)
        print("\n=== REQUESTING FIX FROM SAME CLAUDE SESSION ===")
        fix_response = session.prompt(fix_message)
        print(fix_response.get("result", ""))

        current_code = _read_proof_source(proof_path)
        if not current_code.strip():
            print("\nERROR: Claude did not update FSLean/proof.lean after compile failure.")
            return 1

        print("\n=== UPDATED proof.lean CONTENTS ===")
        print(current_code)

    print(f"\nERROR: Reached max fix attempts ({max_fix_attempts}) without success.")
    return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone Claude generate-compile-fix loop for FSLean/proof.lean"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Inline prompt text for Claude. Overrides --index/--prompt-file if provided.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=DEFAULT_PROMPT_FILE,
        help=(
            "Path to a JSON file containing a list of prompt strings "
            f"(default: {DEFAULT_PROMPT_FILE})."
        ),
    )
    parser.add_argument(
        "--index",
        type=int,
        help="Index of the prompt to load from the prompt list JSON file.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT_DEFAULT,
        help="Repo root containing src/ and FSLean/ (default: current working directory).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sonnet",
        help="Claude model alias to use (default: sonnet).",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="medium",
        help="Claude reasoning effort: low, medium, high, or max.",
    )
    parser.add_argument(
        "--max-fix-attempts",
        type=int,
        default=10,
        help="Maximum compile-fix iterations after initial generation.",
    )
    return parser.parse_args()


def resolve_prompt(args: argparse.Namespace, repo_root: Path) -> str:
    """
    Resolve the prompt from either:
    1. --prompt
    2. --index + JSON prompt file
    """
    if args.prompt and args.prompt.strip():
        return args.prompt.strip()

    if args.index is None:
        raise SystemExit("Provide either --prompt or --index.")

    if args.index < 0:
        raise SystemExit("--index must be non-negative.")

    prompt_file = _resolve_path(args.prompt_file, repo_root)
    prompts = _load_prompt_list(prompt_file)

    print(f"Loaded {len(prompts)} prompts from {prompt_file}")

    if not prompts:
        raise SystemExit(f"Prompt file is empty: {prompt_file}")

    if args.index >= len(prompts):
        raise SystemExit(
            f"--index {args.index} is out of range for {prompt_file} "
            f"(valid range: 0 to {len(prompts) - 1})"
        )

    prompt_text = prompts[args.index].strip()
    if not prompt_text:
        raise SystemExit(
            f"Prompt at index {args.index} in {prompt_file} is empty."
        )

    print(f"Using prompt index {args.index} from {prompt_file}")
    return prompt_text


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    prompt_text = resolve_prompt(args, repo_root)
    return run_loop(
        prompt_text=prompt_text,
        repo_root=repo_root,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        max_fix_attempts=args.max_fix_attempts,
    )


if __name__ == "__main__":
    raise SystemExit(main())