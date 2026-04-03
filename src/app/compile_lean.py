import os
import subprocess
from pathlib import Path


def compile_lean(
    code_string: str,
    repo_root: Path | None = None,
    lean_folder_name: str = "FSLean",
) -> tuple[int, str, str]:
    """Compile Lean code in a temporary file within the local Lake project."""
    repo_root = repo_root or Path(__file__).resolve().parents[2]
    lean_dir = repo_root / lean_folder_name
    scratch_dir = lean_dir / ".scratch"

    scratch_dir.mkdir(parents=True, exist_ok=True)
    lean_path = scratch_dir / "temp.lean"
    lean_path.write_text(code_string, encoding="utf-8")

    target_rel = os.path.relpath(lean_path, lean_dir)
    flags = [
        "-Dlinter.unnecessarySimpa=false",
        "-Dlinter.unusedSimpArgs=false",
        "-Dlinter.deprecated=false",
        "-Dlinter.unusedVariables=false",
    ]
    cmd = ["lake", "env", "lean", *flags, target_rel]
    proc = subprocess.run(
        cmd,
        cwd=str(lean_dir),
        capture_output=True,
        text=True,
    )

    lean_path.unlink(missing_ok=True)
    if scratch_dir.exists() and not any(scratch_dir.iterdir()):
        scratch_dir.rmdir()

    return proc.returncode, proc.stdout, proc.stderr
