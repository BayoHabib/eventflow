#!/usr/bin/env python3
"""One-command audit for EventFlow (phase 0).

Runs, in order: ruff, black --check, mypy, the full test suite with
coverage (which includes the README Quick Start examples), and a replay
of the Chicago recipe on real data when a data directory is available.
Writes an evidence pack (logs, coverage, manifest, replay report) and
prints a single PASS/FAIL verdict.

Usage (from the repository root, inside the dev venv):

    python scripts/audit.py [--out audit/phase-0] [--data-root DIR] [--skip-replay]
    make audit

The audit runs the code from this tree (``PYTHONPATH=<root>/src``) so it
never depends on whichever copy happens to be installed in the venv.
Exit code is 0 only if every step passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path.home() / "workspace" / "eventflow-eda" / "data" / "raw" / "chicago_crime"

# Files that define the audited tree (used for the copy fingerprint).
FINGERPRINT_DIRS = ("src", "tests", "configs", "scripts")
FINGERPRINT_FILES = ("pyproject.toml", "README.md", "TODO.md", "GETTING_STARTED.md")


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_fingerprint() -> str:
    """Hash of the audited tree (paths + contents, caches excluded)."""
    digest = hashlib.sha256()
    paths: list[Path] = []
    for name in FINGERPRINT_DIRS:
        paths.extend(p for p in (ROOT / name).rglob("*") if p.is_file())
    paths.extend(ROOT / name for name in FINGERPRINT_FILES if (ROOT / name).exists())
    for path in sorted(paths):
        rel = str(path.relative_to(ROOT))
        if "__pycache__" in rel or rel.endswith(".pyc"):
            continue
        digest.update(rel.encode())
        digest.update(sha256_file(path).encode())
    return digest.hexdigest()


def dependency_versions() -> dict[str, str]:
    """Versions of the runtime dependencies, as installed."""
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for name in ("polars", "pydantic", "numpy", "shapely", "pyproj", "typer", "yaml"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[name] = "NOT INSTALLED"
    # The version of the tree being audited — read from pyproject.toml,
    # never from an installed copy (which may point at another tree).
    versions["eventflow"] = "unknown"
    for line in (ROOT / "pyproject.toml").read_text().splitlines():
        if line.startswith("version = "):
            versions["eventflow"] = line.split('"')[1]
            break
    return versions


def run_step(name: str, argv: list[str], out_dir: Path, env: dict[str, str]) -> tuple[bool, Path]:
    """Run one audit step, tee its output to a log file, return (ok, log)."""
    log = out_dir / f"{name}.log"
    print(f"--- {name}: {' '.join(argv)}")
    with open(log, "w", encoding="utf-8") as f:
        proc = subprocess.run(
            argv, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, text=True
        )
    ok = proc.returncode == 0
    print(f"--- {name}: {'PASS' if ok else 'FAIL'} (exit {proc.returncode}, log {log.name})")
    return ok, log


def replay_chicago(out_dir: Path, data_root: Path, env: dict[str, str]) -> tuple[bool, dict]:
    """Replay the Chicago recipe on real data and reconcile the output."""
    import polars as pl

    output = out_dir / "chicago_features.parquet"
    ok, _ = run_step(
        "replay_chicago",
        [
            sys.executable,
            "-m",
            "eventflow.cli.main",
            "run",
            "--dataset",
            "chicago_crime",
            "--recipe",
            "chicago_crime_v1",
            "--config",
            "configs/recipes/chicago_crime_v1.yaml",
            "--dataset-config",
            "configs/datasets/chicago_crime_example.yaml",
            "--input",
            str(data_root),
            "--output",
            str(output),
        ],
        out_dir,
        env,
    )
    report: dict = {"ran": True, "command_ok": ok, "output": str(output)}
    if not ok or not output.exists():
        report["checks_ok"] = False
        return False, report
    df = pl.read_parquet(output)
    total = int(df["count"].sum())
    report.update(
        {
            "rows": df.height,
            "columns": sorted(df.columns),
            "n_columns": len(df.columns),
            "sum_count": total,
            "count_min": int(df["count"].min()),
            "count_max": int(df["count"].max()),
            "expected_rows": 56491,
            "expected_sum_count": 60235,
            "checks": {
                "rows_match": df.height == 56491,
                "sum_count_match": total == 60235,
                "no_zeros": int(df["count"].min()) >= 1,
            },
        }
    )
    report["checks_ok"] = all(report["checks"].values())
    (out_dir / "replay_chicago.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report["checks_ok"], report


def main() -> int:
    """Run the audit and write the evidence pack."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="audit/phase-0", help="Evidence pack directory")
    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_DATA_ROOT),
        help="Directory with the raw Chicago crime data (Q1 2024)",
    )
    parser.add_argument("--skip-replay", action="store_true", help="Skip the Chicago replay step")
    args = parser.parse_args()

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")

    results: dict[str, bool] = {}
    results["ruff"] = run_step(
        "ruff", [sys.executable, "-m", "ruff", "check", "src", "tests"], out_dir, env
    )[0]
    results["black"] = run_step(
        "black", [sys.executable, "-m", "black", "--check", "src", "tests"], out_dir, env
    )[0]
    results["mypy"] = run_step("mypy", [sys.executable, "-m", "mypy", "src"], out_dir, env)[0]
    results["pytest"] = run_step(
        "pytest",
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "-o",
            "addopts=",
            "--cov=eventflow",
            "--cov-report=term",
            f"--cov-report=xml:{out_dir / 'coverage.xml'}",
        ],
        out_dir,
        env,
    )[0]

    data_root = Path(args.data_root)
    replay: dict = {"ran": False, "checks_ok": True, "reason": None}
    if args.skip_replay:
        replay["reason"] = "skipped by --skip-replay"
        results["replay_chicago"] = True  # skipped steps do not fail the audit
    elif not data_root.exists():
        replay["reason"] = f"data directory not found: {data_root}"
        results["replay_chicago"] = True
    else:
        ok, replay = replay_chicago(out_dir, data_root, env)
        results["replay_chicago"] = ok

    manifest = {
        "audit": "eventflow phase-0 audit",
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(ROOT),
        "tree_fingerprint_sha256": tree_fingerprint(),
        "versions": dependency_versions(),
        "config_hashes": {
            "configs/recipes/chicago_crime_v1.yaml": sha256_file(
                ROOT / "configs" / "recipes" / "chicago_crime_v1.yaml"
            ),
            "configs/datasets/chicago_crime_example.yaml": sha256_file(
                ROOT / "configs" / "datasets" / "chicago_crime_example.yaml"
            ),
        },
        "results": results,
        "replay": replay,
        "verdict": "PASS" if all(results.values()) else "FAIL",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "verdict.txt").write_text(manifest["verdict"] + "\n", encoding="utf-8")

    print()
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    print(f"VERDICT: {manifest['verdict']}  (evidence pack: {out_dir})")
    return 0 if manifest["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
