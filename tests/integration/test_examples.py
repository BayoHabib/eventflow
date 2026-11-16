"""Tests that execute the runnable examples in ``examples/``.

Each example is run as a program (``python examples/<name>.py``) so the
test proves what a user gets, not just what the library does in-process.
The subprocess inherits a ``PYTHONPATH`` pointing at this checkout's
``src/`` so the example runs against the code under test, never against
whatever ``eventflow`` happens to be installed in the environment.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples"


def _run_example(script: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    """Run one example script in a subprocess; return the completed process."""
    env = dict(os.environ)
    src = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, str(EXAMPLES_DIR / script), "--out-dir", str(tmp_path)],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        check=False,
    )


def test_chicago_quickstart(tmp_path: Path) -> None:
    """examples/chicago_quickstart.py runs the full Chicago recipe on 120
    synthetic events and asserts the panel invariants (516 rows, 62 cols,
    sum(count) == 120, no null counts)."""
    proc = _run_example("chicago_quickstart.py", tmp_path)
    assert proc.returncode == 0, (
        f"chicago_quickstart.py failed (rc={proc.returncode}):\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    assert "Quick start OK" in proc.stdout
    assert "sum(count):              120" in proc.stdout
    out = tmp_path / "chicago_quickstart.parquet"
    assert out.exists(), "quick start did not write its output Parquet"
    assert out.stat().st_size > 0
