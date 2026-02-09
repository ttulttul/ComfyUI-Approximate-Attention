from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_paper.sh"


def test_build_paper_script_exists_and_is_executable():
    assert SCRIPT_PATH.exists(), f"missing script: {SCRIPT_PATH}"
    assert SCRIPT_PATH.is_file(), f"script path is not a file: {SCRIPT_PATH}"
    assert SCRIPT_PATH.stat().st_mode & 0o111, "build script is not executable"


def test_build_paper_script_dry_run_succeeds():
    result = subprocess.run(
        ["bash", str(SCRIPT_PATH), "--dry-run"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "flux2ttr_v2_paper.tex" in result.stdout
