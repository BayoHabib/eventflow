"""Tests for the audit script (scripts/audit.py, phase 0).

Issue #31: a skipped Chicago replay was reported as PASS. The audit
runs ruff, black, mypy and the full test suite, so these tests exercise
``main()`` with the heavy steps stubbed out (they run in milliseconds)
and assert the manifest status, the verdict, the exit code, and the
console summary for both skip paths. The real end-to-end behaviour was
also verified by running the script itself with ``--skip-replay`` and
with a missing data directory.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_audit_module():
    """Load scripts/audit.py as a module (it is not part of the package)."""
    spec = importlib.util.spec_from_file_location(
        "eventflow_audit", REPO_ROOT / "scripts" / "audit.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stub_steps(module, monkeypatch, *, failing: str | None = None):
    """Stub run_step so no lint/type/test step actually runs."""

    def fake_run_step(name, argv, out_dir, env):
        return (name != failing, out_dir / f"{name}.log")

    monkeypatch.setattr(module, "run_step", fake_run_step)


def _read_manifest(out_dir: Path) -> dict:
    return json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))


def test_skip_replay_is_reported_as_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """Issue #31, path A: --skip-replay must not be reported as PASS."""
    audit = _load_audit_module()
    _stub_steps(audit, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["audit.py", "--skip-replay", "--out", str(tmp_path)])

    exit_code = audit.main()
    manifest = _read_manifest(tmp_path)
    out = capsys.readouterr().out

    assert manifest["results"]["replay_chicago"] == "SKIP"
    assert manifest["replay"] == {
        "ran": False,
        "checks_ok": True,
        "reason": "skipped by --skip-replay",
    }
    assert manifest["verdict"] == "INCOMPLETE"
    assert (tmp_path / "verdict.txt").read_text(encoding="utf-8").strip() == "INCOMPLETE"
    assert exit_code == 2
    assert "replay_chicago: SKIP" in out
    assert "VERDICT: INCOMPLETE" in out
    assert "replay_chicago: PASS" not in out


def test_missing_data_is_reported_as_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """Issue #31, path B: a missing data directory must not be PASS."""
    audit = _load_audit_module()
    _stub_steps(audit, monkeypatch)
    missing = tmp_path / "does-not-exist"
    monkeypatch.setattr(
        sys, "argv", ["audit.py", "--data-root", str(missing), "--out", str(tmp_path)]
    )

    exit_code = audit.main()
    manifest = _read_manifest(tmp_path)
    out = capsys.readouterr().out

    assert manifest["results"]["replay_chicago"] == "SKIP"
    assert manifest["replay"]["ran"] is False
    assert "data directory not found" in manifest["replay"]["reason"]
    assert manifest["verdict"] == "INCOMPLETE"
    assert (tmp_path / "verdict.txt").read_text(encoding="utf-8").strip() == "INCOMPLETE"
    assert exit_code == 2
    assert "replay_chicago: SKIP" in out
    assert "VERDICT: INCOMPLETE" in out
    assert "replay_chicago: PASS" not in out


def test_full_run_still_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """Positive control: a replay that actually runs and passes is PASS."""
    audit = _load_audit_module()
    _stub_steps(audit, monkeypatch)
    monkeypatch.setattr(
        audit,
        "replay_chicago",
        lambda out_dir, data_root, env: (True, {"ran": True, "checks_ok": True}),
    )
    monkeypatch.setattr(
        sys, "argv", ["audit.py", "--data-root", str(tmp_path), "--out", str(tmp_path)]
    )

    exit_code = audit.main()
    manifest = _read_manifest(tmp_path)
    out = capsys.readouterr().out

    assert manifest["results"] == {
        "ruff": "PASS",
        "black": "PASS",
        "mypy": "PASS",
        "pytest": "PASS",
        "replay_chicago": "PASS",
    }
    assert manifest["verdict"] == "PASS"
    assert (tmp_path / "verdict.txt").read_text(encoding="utf-8").strip() == "PASS"
    assert exit_code == 0
    assert "replay_chicago: PASS" in out
    assert "VERDICT: PASS" in out


def test_failed_step_still_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """A failed step is FAIL (and beats a skip) with exit code 1."""
    audit = _load_audit_module()
    _stub_steps(audit, monkeypatch, failing="ruff")
    monkeypatch.setattr(sys, "argv", ["audit.py", "--skip-replay", "--out", str(tmp_path)])

    exit_code = audit.main()
    manifest = _read_manifest(tmp_path)
    out = capsys.readouterr().out

    assert manifest["results"]["ruff"] == "FAIL"
    assert manifest["results"]["replay_chicago"] == "SKIP"
    assert manifest["verdict"] == "FAIL"
    assert (tmp_path / "verdict.txt").read_text(encoding="utf-8").strip() == "FAIL"
    assert exit_code == 1
    assert "VERDICT: FAIL" in out


@pytest.mark.parametrize(
    ("results", "expected"),
    [
        ({"ruff": "PASS", "pytest": "PASS"}, "PASS"),
        ({"ruff": "PASS", "replay_chicago": "SKIP"}, "INCOMPLETE"),
        ({"ruff": "FAIL", "replay_chicago": "SKIP"}, "FAIL"),
        ({"ruff": "FAIL", "pytest": "PASS"}, "FAIL"),
    ],
)
def test_compute_verdict(results: dict, expected: str) -> None:
    """Verdict precedence: any FAIL wins, then any SKIP, else PASS."""
    audit = _load_audit_module()
    assert audit.compute_verdict(results) == expected


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [("PASS", 0), ("FAIL", 1), ("INCOMPLETE", 2)],
)
def test_exit_code_for_verdict(verdict: str, expected: int) -> None:
    """Exit codes: 0 fully verified, 1 failed, 2 partially verified."""
    audit = _load_audit_module()
    assert audit.exit_code_for_verdict(verdict) == expected
