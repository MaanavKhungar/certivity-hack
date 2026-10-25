#!/bin/bash
# Final pipeline for one case: stage A2 -> c1full (product filter, prompt v7, stage B sections only) -> submission.
#   ./run_case.sh <case_dir> <selection.json> <profile.json> <name> [team]
set -e
CASE=$(cd "$1" && pwd); SEL=$(cd "$(dirname "$2")" && pwd)/$(basename "$2"); PROF=$(cd "$(dirname "$3")" && pwd)/$(basename "$3")
NAME=$4; TEAM=${5:-team-05}
set -a; . "$(dirname "$0")/../.env"; set +a
cd "$(dirname "$0")/src"
../../.venv/bin/python stage_a2.py "$CASE" --selection "$SEL" --profile "$PROF" --out "../runs/$NAME-stage_a2.json"
REGS=$(../../.venv/bin/python -c "import json;print('\n'.join(r for r,o in json.load(open('../runs/$NAME-stage_a2.json')).items() if o['include']))")
RUNS=()
while read -r R; do
  N="$NAME-c1full-$(echo $R | sed 's/UN //; s/ //g')"; RUNS+=("../runs/$N")
  ../../.venv/bin/python c1full.py "$R" --role base --run-name "$N" --product --prompt v7 --sections \
      --selection "$SEL" --profile "$PROF" --workers 6 > "../runs/$N.console.txt" 2>&1 &
done <<< "$REGS"
wait
../../.venv/bin/python c1full_submit.py "${RUNS[@]}" --case "$(basename "$CASE")" --team "$TEAM" --out "../runs/submission-$NAME"
