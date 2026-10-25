#!/bin/bash
# Maanav's c1full v4 + product filter, on the regulations stage A2 includes (front bumper)
set -a; . ../../.env; set +a
cd "$(dirname "$0")/src"
SEL=../../team-repo/runs/bumper-v2/selection.json; PROF=../../team-repo/runs/bumper-v2/profile.json
REGS=$(../../.venv/bin/python -c "import json;print('\n'.join(r for r,o in json.load(open('../runs/bumper-stage_a2.json')).items() if o['include']))")
RUNS=()
while read -r R; do
  N="bumper-c1full-v4p-$(echo $R | sed 's/UN //; s/ //g')"; RUNS+=("../runs/$N")
  ../../.venv/bin/python c1full.py "$R" --role base --run-name "$N" --product --prompt v4 --selection $SEL --profile $PROF --workers 6 > "../runs/$N.console.txt" 2>&1 &
done <<< "$REGS"
wait
../../.venv/bin/python c1full_submit.py "${RUNS[@]}" --case heldout-front-bumper --team team-05 --out ../runs/submission-bumper-c1full-v4p-a2
