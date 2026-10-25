"""Submission runner for branch sub-v1-c1full-d: python src/run_submission.py <case_dir> --run-name <name> --team <team_id>"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import runners

if __name__ == "__main__":
    runners.main(["v1"] + sys.argv[1:])
