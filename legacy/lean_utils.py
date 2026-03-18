import subprocess, os, shlex
from pathlib import Path

def compile_lean(code_string, repo_name = "formal-science", lean_folder_name = "FSLean"):

  repo_root = Path.home() / repo_name
  lean_dir = repo_root / lean_folder_name    

  scratch_dir = lean_dir / ".scratch"        

  # Create temp lean file
  scratch_dir.mkdir(parents=True, exist_ok=True)
  fname = f"temp.lean"
  lean_path = scratch_dir / fname

  with open(lean_path, "w", encoding="utf-8") as f:
      f.write(code_string)

  target_rel = os.path.relpath(lean_path, lean_dir)
  depr_warning = "-Dlinter.deprecated=false"
  simpa_warning = "-Dlinter.unnecessarySimpa=false"
  simpargs_warning = "-Dlinter.unusedSimpArgs=false"
  var_warning = "-Dlinter.unusedVariables=false"
  cmd = f"lake env lean {simpa_warning} {simpargs_warning} {depr_warning} {var_warning} {shlex.quote(target_rel)}"
  proc = subprocess.run(
      shlex.split(cmd),
      cwd=str(lean_dir),        # run from folder with lakefile.lean
      capture_output=True,
      text=True,
  )

  lean_path.unlink(missing_ok=True)
  if scratch_dir.exists() and not any(scratch_dir.iterdir()):
    scratch_dir.rmdir()

  return proc.returncode, proc.stdout