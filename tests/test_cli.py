import json
import os
from pathlib import Path
import subprocess
import sys


def test_cli_writes_parseable_receipt(tmp_path):
    out = tmp_path / "receipt.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    completed = subprocess.run(
        [sys.executable, "run_experiment.py", "--out", str(out), "--random-repeats", "4"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert set(("config", "wave", "diffusion")).issubset(payload)
    assert "optimized schedule" in completed.stdout.lower()
