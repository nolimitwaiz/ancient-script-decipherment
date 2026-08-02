"""The §1 CI gate: pretrained-weight loading fails the build outside
contrib/openbook/. Forbidden strings are assembled at runtime here so this
test file itself stays clean under the gate's grep."""

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "check_no_pretrained.sh"

FORBIDDEN_CALL = "model = " + "Auto" + "Model." + "from_" + "pretrained('bert-base-uncased')\n"


def _run(root: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(SCRIPT), str(root)], capture_output=True, text=True)


def test_repo_is_clean():
    result = _run(REPO)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "clean" in result.stdout


def test_planted_violation_is_caught(tmp_path):
    (tmp_path / "glyphos").mkdir()
    (tmp_path / "glyphos" / "bad.py").write_text(FORBIDDEN_CALL)
    result = _run(tmp_path)
    assert result.returncode == 1
    assert "FORBIDDEN" in result.stdout
    assert "bad.py" in result.stdout


def test_openbook_directory_is_exempt(tmp_path):
    exempt = tmp_path / "glyphos" / "contrib" / "openbook"
    exempt.mkdir(parents=True)
    (exempt / "openbook_model.py").write_text(FORBIDDEN_CALL)
    result = _run(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
