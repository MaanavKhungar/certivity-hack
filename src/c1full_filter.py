"""Product filter on c1full output, then precision / recall / F2 of the filtered list (evaluation experiment; c1full
itself stays product-free, artifacts/new-c1-traverse-integrate.md "Deliberate boundaries").

    python src/c1full_filter.py ../industrial-ai-hackathon/cases/dev-rear-lamp runs/r148-c1full-v3 runs/r10-c1full-v3 \
        --selection runs/rearlamp-v7b/selection.json --profile data/product/dev-rear-lamp.profile.json --out runs/c1full-v3-d2

Filter  bbpipe's stage D2: one call per regulation, every candidate scored 0..1 (probability an expert lists it for
        this product) plus duplicate_of, with the same product inputs bb2 used (stage A summary, stage B mapping of the
        regulation, profile). Candidates are grouped by section (main-body top-level section / annex part) with the
        section's title and kind. c1full covers the whole PDF, so the prompt says so instead of "selected / reached by
        reference".
Select  duplicates dropped, then score >= threshold, one judged submission per --thresholds value (one scoring pass)
Compare --baseline's submission_d2.json (C1 -> D2 of the product pipeline) judged on the same regulations
expected.json is read only by the judge.
"""
from __future__ import annotations
import argparse, json, sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bbpipe
import judge
import pipeline as rp
import reqagent
from regparse import Corpus

OLD = ("each section line gives its title, its kind (requirement, test,\nconformity, ...) and whether it was selected "
       "directly or reached through a cross-reference.")
NEW = ("each section line gives its id, title and kind (requirement, test,\nconformity, ...). The candidates were extracted "
       "from the WHOLE regulation, for every device type, vehicle\ncategory and variant it covers, without looking at the "
       "product: expect many sections and candidates that do not\nconcern this product. Use the section ids exactly as "
       "given in the sections array.")
assert OLD in bbpipe.D2_SYSTEM
SYSTEM = bbpipe.D2_SYSTEM.replace(OLD, NEW)


def section_of(n) -> str:
    return f"§{n.label.split('.')[0]}" if n.doc == "main" else n.doc


def score(reg: str, reqs: list, C: Corpus, role: str, A: dict, b: dict, profile: dict) -> dict:
    """{requirement id: {score, duplicate_of, reason, section, section_score}} for one regulation (one call)."""
    nodes = C.by_reg[(reg, role)]
    by_label = {}                                         # "5.2.1" / "Annex 5 1.2" (the label part of a c1full id)
    for n in nodes:
        by_label.setdefault(n.label if n.doc == "main" else f"{n.doc} {n.label}", n)
    secs: dict[str, dict] = {}
    for n in nodes:                                       # section title / kind from its nodes
        s = secs.setdefault(section_of(n), {"title": "", "kinds": Counter()})
        s["title"] = s["title"] or (n.doc_title if n.doc != "main" else (n.title or ""))
        s["kinds"][n.section_kind or "requirement"] += 1
    groups: dict[str, list] = {}
    for r in reqs:
        n = by_label.get(r["source_node_ids"][0].split("|", 2)[2].split("#")[0])
        groups.setdefault(section_of(n) if n else "?", []).append(r)
    ids = {k: f"S{i + 1:02d}" for i, k in enumerate(groups)}
    lines = []
    for k, rs in groups.items():
        s = secs.get(k, {"title": "", "kinds": Counter()})
        kind = s["kinds"].most_common(1)[0][0] if s["kinds"] else "?"
        lines.append(f"\n### {ids[k]} Section {k}: {s['title']} [kind: {kind}]")
        lines += [f"{r['id']} [§{r['source']['paragraph']}] {r['text']}" for r in rs]
    user = ("## Product\n" + A["product_summary"] + "\n\n## Product profile\n" + json.dumps(profile, indent=1, ensure_ascii=False) +
            "\n\n## How this regulation applies (earlier analysis)\n" +
            json.dumps({k: b[k] for k in ("how_it_applies", "part_mapping", "open_options")}, indent=1, ensure_ascii=False) +
            f"\n\n## Candidates from {reg} ({len(reqs)}), by section\n" + "\n".join(lines) +
            "\n\nScore every section (by its id) and every candidate id.")
    res = rp.call_json(SYSTEM, [{"type": "text", "text": user}], bbpipe.D2_SCHEMA, max_tokens=64000,
                       tag=f"C1F_D2_{reg.replace(' ', '_')}")
    sec_score = {s["section"].split()[0]: s["expert_lists_it"] for s in res["sections"]}
    out = {d["id"]: {"score": d["score"], "duplicate_of": d["duplicate_of"], "reason": d["reason"]} for d in res["candidates"]}
    for k, rs in groups.items():
        for r in rs:
            out.setdefault(r["id"], {"score": None, "duplicate_of": None, "reason": "not scored by the model"})
            out[r["id"]].update({"section": k, "section_score": sec_score.get(ids[k])})
    return out


def submission(reqs: list, case_id: str, team: str) -> dict:
    return {"team_id": team, "case_id": case_id,
            "requirements": [{"id": f"R-{i + 1:04d}", "text": r["text"],
                              "source": {k: r["source"][k] for k in ("regulation", "version", "page", "paragraph")}}
                             for i, r in enumerate(reqs)]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("case_dir")
    ap.add_argument("runs", nargs="+", help="c1full run dirs (one regulation each)")
    ap.add_argument("--selection", default=str(reqagent.REPO / "runs/rearlamp-v7b/selection.json"))
    ap.add_argument("--profile", default=str(reqagent.REPO / "data/product/dev-rear-lamp.profile.json"))
    ap.add_argument("--baseline", default=str(reqagent.REPO / "runs/rearlamp-bb2"), help="run with submission_d2.json")
    ap.add_argument("--thresholds", default="0.3,0.5,0.7")
    ap.add_argument("--team", default="team-05")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    reqagent.load_env()
    case, out = Path(a.case_dir).resolve(), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sel = json.loads(Path(a.selection).read_text())
    A, B = sel["stageA"], sel["stageB"]
    base = json.loads(Path(a.profile).read_text())
    profile = base.get("model") or base.get("deterministic") or base
    C = Corpus(reqagent.REPO / "data/pdf")
    runs = []
    for d in a.runs:
        d = Path(d)
        m = json.loads((d / "extraction_manifest.json").read_text())
        runs.append((m["regulation"], m["role"], json.loads((d / "requirements_with_context.json").read_text())["requirements"]))
    regs = [r for r, _, _ in runs]
    exp_all = json.loads((case / "expected.json").read_text())
    exp = {"requirements": [e for e in exp_all["requirements"] if e["source"]["regulation"] in regs]}
    lines = []
    log = lambda s: (print(s, flush=True), lines.append(s))

    with ThreadPoolExecutor(4) as pool:
        parts = list(pool.map(lambda x: score(x[0], x[2], C, x[1], A, B[x[0]], profile), runs))
    scores = {k: v for p in parts for k, v in p.items()}
    (out / "d2_scores.json").write_text(json.dumps(scores, indent=1, ensure_ascii=False))
    allreqs = [r for _, _, rs in runs for r in rs]
    unscored = sum(scores[r["id"]]["score"] is None for r in allreqs)
    dups = sum(bool(scores[r["id"]]["duplicate_of"]) for r in allreqs)
    log(f"D2: {len(allreqs)} candidates scored ({unscored} missing a score, kept as 0), {dups} marked duplicate")

    rows = []

    def judged(name, sub):
        d = out / f"judge_{name.replace(' ', '_').replace('>=', 'ge')}"
        d.mkdir(exist_ok=True)
        (d / "submission.json").write_text(json.dumps(sub, indent=1, ensure_ascii=False))
        problems = rp.validate(sub)
        J = judge.judge(sub, exp, d)
        s = J["summary"]
        per = {r: (v["matched"], v["submitted"], v["expected"]) for r, v in J["per_regulation"].items()}
        rows.append((name, s, per, problems))

    bpath = Path(a.baseline) / "submission_d2.json"
    if bpath.exists():
        b = json.loads(bpath.read_text())
        b["requirements"] = [r for r in b["requirements"] if r["source"]["regulation"] in regs]
        judged("baseline C1 -> D2 (bb2)", b)
    judged("c1full v3, no filter", submission(allreqs, case.name, a.team))
    for t in [float(x) for x in a.thresholds.split(",")]:
        kept = [r for r in allreqs if not scores[r["id"]]["duplicate_of"] and (scores[r["id"]]["score"] or 0) >= t]
        judged(f"c1full v3 -> D2 score >= {t}", submission(kept, case.name, a.team))

    log(f"\n## Filtered lists on {', '.join(regs)} (judge.py)\n\n| list | submitted | matched | expected | precision | recall | F2 | "
        + " | ".join(regs) + " (matched/submitted) |\n|---|---|---|---|---|---|---|" + "---|" * len(regs))
    for name, s, per, problems in rows:
        cells = " | ".join(f"{per[r][0]}/{per[r][1]}" if r in per else "–" for r in regs)
        log(f"| {name} | {s['submitted']} | {s['matched']} | {s['expected']} | {s['precision']:.3f} | {s['recall']:.3f} | "
            f"**{s['f2']:.3f}** | {cells} |" + (f" validation: {problems[:2]}" if problems else ""))
    (out / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
