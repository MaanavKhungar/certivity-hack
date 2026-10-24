"""End to end for one case: extended profile -> requirements for every included regulation -> one submission.

    python src/pipeline.py ../industrial-ai-hackathon/cases/dev-rear-lamp --run-name rearlamp-v4
    python src/pipeline.py <case> --skip-profile      # reuse data/product/<case>.extended.json

Writes runs/<run>/submission.json, runs validate.py and score_local.py on it, and, when the case has an
expected.json, the per-regulation evaluation (report_<reg>.md) plus an overall summary.
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reqagent import PKG, REPO, SYSTEM, extract
import evaluate as ev


def product_for(ext: dict, entry: dict) -> dict:
    base = ext["base_profile"]
    return {"product_summary": ext["product_summary"],
            "profile": base.get("model") or base.get("deterministic"),
            "regulation_mapping": entry.get("mapping")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("case_dir")
    ap.add_argument("--run-name")
    ap.add_argument("--team", default="team-05")
    ap.add_argument("--threshold", type=float, default=None, help="override the extended profile's threshold")
    ap.add_argument("--skip-profile", action="store_true")
    a = ap.parse_args(argv)
    case = Path(a.case_dir).resolve()

    ext_path = REPO / "data/product" / f"{case.name}.extended.json"
    if not a.skip_profile or not ext_path.exists():
        import extend_profile
        extend_profile.extend(case, a.threshold or 0.3, 0.15)
    ext = json.loads(ext_path.read_text())
    thr = a.threshold if a.threshold is not None else ext["threshold"]
    regs = [e for e in ext["regulations"] if e["likelihood"] >= thr]
    print(f"\nextracting from {len(regs)} regulations: {', '.join(e['regulation'] for e in regs)}")

    run = REPO / "runs" / (a.run_name or datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + case.name)
    run.mkdir(parents=True, exist_ok=True)

    def one(e):
        reqs, raw = extract(e["regulation"], product_for(ext, e), workers=6,
                            log=lambda s, r=e["regulation"]: print(f"[{r}] {s}"))
        return e["regulation"], reqs, raw

    with ThreadPoolExecutor(3) as pool:
        results = list(pool.map(one, regs))

    all_reqs, all_raw = [], {}
    for reg, reqs, raw in results:
        all_raw[reg] = raw
        for r in reqs:
            all_reqs.append({**r, "id": f"R-{len(all_reqs)+1:04d}"})
    sub = {"team_id": a.team, "case_id": case.name, "requirements": all_reqs}
    (run / "submission.json").write_text(json.dumps(sub, indent=1, ensure_ascii=False))
    (run / "raw.json").write_text(json.dumps(all_raw, indent=1, ensure_ascii=False))
    (run / "system_prompt.txt").write_text(SYSTEM)
    (run / "extended_profile.json").write_text(json.dumps(ext, indent=1, ensure_ascii=False))
    print(f"\n{len(all_reqs)} requirements -> {run / 'submission.json'}\n")

    subprocess.run([sys.executable, str(PKG / "tools/validate.py"), str(run / "submission.json")])
    expected = case / "expected.json"
    if not expected.exists():
        return
    subprocess.run([sys.executable, str(PKG / "tools/score_local.py"), str(run / "submission.json"), "--expected", str(expected)])

    # per-regulation requirement-level evaluation over the union of cited and expected regulations
    exp = json.loads(expected.read_text())["requirements"]
    names = sorted({r["source"]["regulation"] for r in exp} | {r["source"]["regulation"] for r in all_reqs})
    tot_m = tot_p = tot_e = 0
    rows = []
    for name in names:
        n_e = sum(r["source"]["regulation"] == name for r in exp)
        n_p = sum(r["source"]["regulation"] == name for r in all_reqs)
        if n_e and n_p:
            res = ev.evaluate(run / "submission.json", name, expected)
            (run / f"report_{name.replace(' ', '_')}.md").write_text((run / "report.md").read_text())
            m = res["requirement_level"]["n_matched"]
        else:
            m = 0
        rows.append((name, n_p, n_e, m)); tot_m += m; tot_p += n_p; tot_e += n_e
    (run / "report.md").unlink(missing_ok=True); (run / "eval.json").unlink(missing_ok=True)
    P, R = tot_m / max(1, tot_p), tot_m / max(1, tot_e)
    lines = [f"# {case.name}: {run.name}\n", "| regulation | predicted | expected | matched | F2 | recall |", "|---|---|---|---|---|---|"]
    lines += [f"| {n} | {p} | {e} | {m} | {ev.f2(m/max(1,p), m/max(1,e)):.2f} | {m/max(1,e):.2f} |" for n, p, e, m in rows]
    lines.append(f"| **all** | {tot_p} | {tot_e} | {tot_m} | **{ev.f2(P, R):.2f}** | **{R:.2f}** |")
    lines.append(f"\n**requirement level: F2 {ev.f2(P, R):.3f}, recall {R:.3f}** (own judge, per regulation)")
    (run / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
