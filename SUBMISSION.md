# Branch sub-v1-c1full-d: variant v1


See the docstring of src/runners.py for what variant v1 does and its rear-lamp score.

```
cd package && git pull && cd ..                      # held-out case appears under package/cases/
pip install rapidfuzz scikit-learn                   # once
python src/run_submission.py ../industrial-ai-hackathon/cases/<case> --run-name <case>-v1 --team <team_id>
# -> runs/<case>-v1/submission.json (validated); upload that file
```
The API key is read from .env in the repo or one folder up. Profile (vision) and stage A/B run from scratch;
pass --profile / --selection to reuse saved ones.
