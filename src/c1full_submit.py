"""Merge c1full runs (one regulation each) into one challenge submission.json; optionally judge it.

    python src/c1full_submit.py runs/r148-c1full-v3p runs/r48-c1full-v3p ... --case dev-rear-lamp \
        --out runs/submission-c1full-v3p [--expected ../industrial-ai-hackathon/cases/dev-rear-lamp/expected.json]

Meant for product-filtered runs (c1full --product): a product-free run is a full-PDF extraction, not a submission.
Requirements keep their run order; ids become R-0001.. with a mapping back to the c1full ids.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline as rp


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--case", required=True, help="case_id, e.g. dev-rear-lamp")
    ap.add_argument("--team", default="team-05")
    ap.add_argument("--out", required=True)
    ap.add_argument("--core-only", action="store_true", help="keep only requirements C1 marked priority=core (v6)")
    ap.add_argument("--expected", help="expected.json: also judge the submission (evaluation only)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    reqs, mapping, per = [], {}, {}
    for d in map(Path, a.runs):
        m = json.loads((d / "extraction_manifest.json").read_text())
        if not m.get("product_filter"):
            print(f"WARNING {d}: not a product-filtered run (c1full --product)")
        if m["calls"]["failed_batches"]:
            print(f"WARNING {d}: failed batches {m['calls']['failed_batches']}")
        rs = json.loads((d / "requirements_with_context.json").read_text())["requirements"]
        if a.core_only:
            rs = [r for r in rs if r.get("priority", "core") == "core"]
        per[m["regulation"]] = len(rs)
        for r in rs:
            rid = f"R-{len(reqs) + 1:04d}"
            reqs.append({"id": rid, "text": r["text"],
                         "source": {k: r["source"][k] for k in ("regulation", "version", "page", "paragraph")}})
            mapping[rid] = {"run": d.name, "c1full_id": r["id"]}
    sub = {"team_id": a.team, "case_id": a.case, "requirements": reqs}
    problems = rp.validate(sub)
    (out / "submission.json").write_text(json.dumps(sub, indent=1, ensure_ascii=False))
    (out / "submission_mapping.json").write_text(json.dumps(mapping, indent=1, ensure_ascii=False))
    print(f"{len(reqs)} requirements {per} -> {out / 'submission.json'}; validation: {'OK' if not problems else problems[:5]}")
    if a.expected:
        import judge
        (out / "judge").mkdir(exist_ok=True)
        J = judge.judge(sub, json.loads(Path(a.expected).read_text()), out / "judge")
        s = J["summary"]
        print(f"judge: submitted {s['submitted']}, matched {s['matched']}/{s['expected']}, "
              f"precision {s['precision']:.3f}, recall {s['recall']:.3f}, F2 {s['f2']:.3f}")
        for reg, v in J["per_regulation"].items():
            print(f"  {reg:<8} matched {v['matched']:>3}/{v['expected']:<3} of {v['submitted']} submitted")


if __name__ == "__main__":
    main()
