"""Recall check of c1full runs against a case's expected.json (evaluation only; never feeds back into extraction).

    python src/c1full_eval.py ../industrial-ai-hackathon/cases/dev-rear-lamp runs/r148-c1full runs/r48-c1full ... \
        --baseline runs/rearlamp-bb2 --out runs/c1full-eval

c1full reads whole PDFs with no product filter, so precision / F2 against a product-specific expected.json say
nothing; recall is the measure: did full extraction lose requirements the product pipeline's C1 found?
1. paragraph coverage (code only): per expected requirement, was its paragraph in a processed batch, and does some
   c1full requirement cite that paragraph (same node, or its parent / sub-paragraph)?
2. requirement-level recall with judge.judge (the same LLM matcher as runs/*/judge_c1_only), per regulation:
   "full" = every c1full requirement; "bb2 scope" = only requirements from paragraphs the baseline's C1 read
   (its traversal extract items), the like-for-like comparison with the baseline's judge_c1_only.
"""
from __future__ import annotations
import argparse, json, re, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import judge
import reqagent
from regparse import paragraph_key


def exp_key(paragraph: str) -> str:
    return paragraph_key(re.sub(r"\.?\s*Table\s+\S+$", "", str(paragraph)))


def related(a: str, b: str) -> bool:
    """Same paragraph, or one is a sub-paragraph of the other ("5.1" ~ "5.1.2")."""
    return a == b or a.startswith(b + ".") or b.startswith(a + ".")


def load_run(d: Path) -> dict:
    m = json.loads((d / "extraction_manifest.json").read_text())
    reqs = json.loads((d / "requirements_with_context.json").read_text())["requirements"]
    return {"dir": d, "manifest": m, "reg": m["regulation"], "reqs": reqs}


def coverage(run: dict, expected: list) -> list[dict]:
    m, reg = run["manifest"], run["reg"]
    status = {}                                           # paragraph key -> batch status of that node
    for q, s in m.get("node_status", {}).items():
        status[paragraph_key(q.split("|", 2)[2].split("#")[0])] = s
    cited = [(paragraph_key(r["source"]["paragraph"].split(". Table")[0]), r["id"]) for r in run["reqs"]]
    rows = []
    for e in expected:
        k = exp_key(e["source"]["paragraph"])
        st = status.get(k)
        kk = k
        while st is None and "." in kk:                   # expected label deeper than our nodes: walk up
            kk = kk.rsplit(".", 1)[0]; st = status.get(kk)
        hits = [rid for ck, rid in cited if related(ck, k)]
        verdict = ("not_in_corpus" if st is None else "batch_" + st if st != "processed" else
                   "cited" if hits else "processed_not_cited")
        rows.append({"expected_id": e["id"], "regulation": reg, "paragraph": e["source"]["paragraph"],
                     "verdict": verdict, "c1full_ids": hits[:10], "text": e["text"][:200]})
    return rows


def bb2_scope(baseline: Path, reg: str) -> set[str] | None:
    p = baseline / "traversal.json"
    if not p.exists():
        return None
    T = json.loads(p.read_text())
    return {nk for it in T["items"] if it["regulation"] == reg and it["mode"] == "extract" for nk in it["nodes"]}


def submission(reqs: list, case_id: str) -> dict:
    return {"team_id": "team-05", "case_id": case_id,
            "requirements": [{"id": r["id"], "text": r["text"], "source": {k: r["source"][k] for k in ("regulation", "version", "page", "paragraph")}}
                             for r in reqs]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("case_dir")
    ap.add_argument("runs", nargs="+", help="c1full run dirs (one regulation each)")
    ap.add_argument("--baseline", default=str(reqagent.REPO / "runs/rearlamp-bb2"), help="run with judge_c1_only/ and traversal.json")
    ap.add_argument("--out", default=str(reqagent.REPO / "runs/c1full-eval"))
    ap.add_argument("--no-judge", action="store_true", help="paragraph coverage only (no model calls)")
    a = ap.parse_args(argv)
    reqagent.load_env()
    case, base, out = Path(a.case_dir).resolve(), Path(a.baseline), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    exp = json.loads((case / "expected.json").read_text())["requirements"]
    runs = [load_run(Path(d)) for d in a.runs]
    base_j = json.loads((base / "judge_c1_only/judge.json").read_text())["per_regulation"] if (base / "judge_c1_only/judge.json").exists() else {}
    lines = []
    log = lambda s: (print(s, flush=True), lines.append(s))

    # 1. paragraph coverage ---------------------------------------------------------------------------------------
    cov = []
    for r in runs:
        E = [e for e in exp if e["source"]["regulation"] == r["reg"]]
        vers = {e["source"].get("version") for e in E}
        if E and r["manifest"]["version"] not in vers:
            log(f"WARNING {r['reg']}: c1full read version {r['manifest']['version']}, expected.json cites {sorted(map(str, vers))}")
        cov += coverage(r, E)
    (out / "coverage.json").write_text(json.dumps(cov, indent=1, ensure_ascii=False))
    log("## Paragraph coverage (code only)\n\n| regulation | expected | cited | processed, not cited | batch failed / not processed | not in corpus |\n|---|---|---|---|---|---|")
    for reg in sorted({c["regulation"] for c in cov}):
        c = Counter(x["verdict"] for x in cov if x["regulation"] == reg)
        log(f"| {reg} | {sum(c.values())} | {c['cited']} | {c['processed_not_cited']} | "
            f"{c['batch_failed'] + c['batch_not_processed']} | {c['not_in_corpus']} |")
    missing_regs = sorted({e["source"]["regulation"] for e in exp} - {r["reg"] for r in runs})
    if missing_regs:
        log(f"\nno c1full run given for: {', '.join(missing_regs)} (left out of all numbers below)")
    if a.no_judge:
        (out / "summary.md").write_text("\n".join(lines) + "\n"); return

    # 2. requirement-level recall ---------------------------------------------------------------------------------
    rows = []
    for r in runs:
        E = {"requirements": [e for e in exp if e["source"]["regulation"] == r["reg"]]}
        if not E["requirements"]:
            continue
        scope = bb2_scope(base, r["reg"])
        for name, reqs in [("full", r["reqs"])] + ([("bb2 scope", [q for q in r["reqs"] if any(
                paragraph_key(i.split("|", 2)[2].split("#")[0]) in scope for i in q["source_node_ids"])])] if scope else []):
            d = out / f"judge_{r['reg'].replace(' ', '_')}_{name.replace(' ', '_')}"
            d.mkdir(exist_ok=True)
            J = judge.judge(submission(reqs, case.name), E, d)
            pr = J["per_regulation"].get(r["reg"], {})
            rows.append((r["reg"], name, len(reqs), pr.get("matched", 0), len(E["requirements"]),
                         Counter(u["reason"] for u in pr.get("unmatched_expected", []))))
    log("\n## Requirement-level recall (judge.py)\n\n| regulation | c1full set | submitted | matched | expected | recall | "
        "baseline C1 recall | misses by reason |\n|---|---|---|---|---|---|---|---|")
    tot = {}
    for reg, name, n, m, e, why in rows:
        b = base_j.get(reg)
        brec = f"{b['matched'] / b['expected']:.3f} ({b['matched']}/{b['expected']})" if b and b["expected"] else "–"
        log(f"| {reg} | {name} | {n} | {m} | {e} | {m / e:.3f} | {brec} | {dict(why)} |")
        t = tot.setdefault(name, [0, 0]); t[0] += m; t[1] += e
    bm = sum(base_j[r]["matched"] for r in {x[0] for x in rows} if r in base_j)
    be = sum(base_j[r]["expected"] for r in {x[0] for x in rows} if r in base_j)
    for name, (m, e) in tot.items():
        log(f"| **all** | {name} | | {m} | {e} | **{m / e:.3f}** | {f'{bm / be:.3f} ({bm}/{be})' if be else '–'} | |")
    log("\nPrecision/F2 are not reported: c1full extracts the whole PDF, expected.json is product-specific.")
    (out / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
