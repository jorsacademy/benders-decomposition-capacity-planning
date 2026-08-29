import json
import subprocess
import sys


def test_cli_emits_valid_json() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "benders_capacity",
            "--tolerance",
            "1e-5",
            "--max-iterations",
            "80",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["benders"]["converged"] is True
    assert abs(payload["objective_difference"]) < 1e-3
