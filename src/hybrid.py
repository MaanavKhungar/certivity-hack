"""Hybrid pipeline (v6): regulation cards + product mapping (pipeline_cards) x section selection, cross-references,
cache and dedupe (pipeline).

    python src/hybrid.py ../industrial-ai-hackathon/cases/dev-rear-lamp --run-name rearlamp-v6

Stage A  select regulations: screening over all regulation cards -> likelihood; consolidated regulations exclude the
         ones they replace (pipeline.SUPERSEDED); BM25 over the corpus adds lexical candidates.
Stage B  per candidate: card + table of contents -> parts mapped to defined terms, open options (product features
         only), selected section keys, and a final applies/not. Candidates below --sure are kept only if B says applies.
Stage C  per selected section: the v3 extraction rules (reqagent.SYSTEM) + mapping + referenced paragraphs as context
         + style examples from the sub-goal's expert requirements (a different regulation, so no answer leak).
Post     dedupe per paragraph, ids, page/paragraph from the parsed node, validation, evaluation if expected.json exists.
Every model call goes through pipeline.call_json, so re-runs only pay for calls whose input changed.
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline as rp                                     # remote module: cache, TOC keys, sections, refs, BM25, dedupe
import reqagent                                           # extraction rules (SYSTEM), chunking, rendering
import evaluate as ev
from regcards import CARDS
from extend_profile import COMPACT, SCREEN_SYSTEM, SCREEN_SCHEMA, product_blocks
from regparse import Corpus

REPO = reqagent.REPO
PKG = rp.PKG
LIGHT_SOURCE_REGS = {"UN R37", "UN R99", "UN R128"}

# ------------------------------------------------------------------------------------------------ Stage A ---------
SCREEN_RULES = """

Two general rules:
- Consolidated device regulations replace the individual ones for new approvals: UN R148 replaces R4, R6, R7, R23,
  R38, R50, R77, R87, R91; UN R149 replaces R19, R98, R112, R113, R123; UN R150 replaces R3, R27, R69, R70, R104. When
  the consolidated one applies, give the replaced ones a low probability.
- Light-source regulations (R37 filament, R99 gas-discharge, R128 LED) apply only to replaceable light sources, not to
  light sources that are part of a non-replaceable module of the lamp."""


def stage_a(profile_blocks, profile, thr: float, log) -> dict:
    cards = {c["regulation"]: c for c in (json.loads(p.read_text()) for p in sorted(CARDS.glob("*.json")))}
    compact = [{k: c[k] for k in COMPACT} for c in cards.values()]
    res = rp.call_json(SCREEN_SYSTEM + SCREEN_RULES, profile_blocks + [{"type": "text", "text":
                       f"## Regulation cards ({len(compact)})\n" + json.dumps(compact, indent=1, ensure_ascii=False)}],
                       SCREEN_SCHEMA, max_tokens=64000, tag="H_A")
    regs = {r["regulation"]: r for r in res["regulations"] if r["regulation"] in cards}
    likely = {r for r, v in regs.items() if v["likelihood"] >= thr}
    superseded = {old for new, olds in rp.SUPERSEDED.items() if new in likely for old in olds}
    C = Corpus(REPO / "data/pdf")
    bm = rp.bm25_regulations(C, profile.get("search_terms", []) + profile.get("regulated_functions", []), top=8)
    cand = {}
    for r, v in regs.items():
        why = None
        if r in superseded:
            v["excluded"] = "replaced by a consolidated regulation that applies"
        elif r in LIGHT_SOURCE_REGS and v["likelihood"] < 0.6:
            v["excluded"] = "light-source regulation; light sources not confirmed replaceable"
        elif v["likelihood"] >= thr:
            why = "screening"
        elif r in dict(bm):
            why = "bm25"
        if why:
            cand[r] = {**v, "via": why}
    log(f"stage A: {len(cand)} candidates: " + ", ".join(f"{r} {v['likelihood']:.2f}{'*' if v['via']=='bm25' else ''}"
                                                      for r, v in sorted(cand.items(), key=lambda kv: -kv[1]['likelihood'])))
    return {"product_summary": res["product_summary"], "regulations": regs, "candidates": cand, "bm25": bm, "cards": cards, "C": C}


# ------------------------------------------------------------------------------------------------ Stage B ---------
B_SYSTEM = """You are a vehicle type-approval engineer. You receive a product (specification, exploded view, parsed
profile), the reference card of one UN Regulation that may apply to it, and the regulation's table of contents with
section keys. Work out how the regulation applies:
- applies: does the regulation set obligations this product must fulfil (directly, through installation of devices of
  this kind, or through the product's technology)? Be honest: say false when it does not.
- map each part or function of the product to the regulation's defined terms and classes, with evidence.
- open_options: product features the inputs do not settle and that change which requirements apply (e.g. replaceable
  or non-replaceable light source, watertight or not, steady or variable intensity, logo or not). List every plausible
  option; do not pick one. Only features of the product itself, not how a vehicle manufacturer might use it.
- selected_sections: the keys (exactly as printed before the first '|') of every section with obligations the product
  must fulfil: marking provisions the device must carry, general requirements, the specific sections for every
  function or class the product has or may have (open options), conformity-of-production tolerances, and test annexes
  whose tests or pass criteria apply to the product. Exclude application/approval procedure, communication forms,
  modifications/extensions, penalties, production discontinued, technical services, transitional provisions,
  definitions, and sections for devices, classes or vehicle categories the product confirmed does not have.
  A missed section loses all its requirements, an extra one costs a little: when in doubt, include it."""

B_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["applies", "how_it_applies", "part_mapping", "open_options", "selected_sections"],
    "properties": {
        "applies": {"type": "boolean"},
        "how_it_applies": {"type": "string"},
        "part_mapping": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["part", "defined_term", "class_or_variant", "confirmed"],
            "properties": {"part": {"type": "string"}, "defined_term": {"type": "string"},
                           "class_or_variant": {"type": ["string", "null"]}, "confirmed": {"type": "boolean"}}}},
        "open_options": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["feature", "options"],
            "properties": {"feature": {"type": "string"}, "options": {"type": "array", "items": {"type": "string"}}}}},
        "selected_sections": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["key", "reason"],
            "properties": {"key": {"type": "string"}, "reason": {"type": "string"}}}},
    },
}


def stage_b(A, profile_blocks, sure: float, log) -> dict:
    C, cards = A["C"], A["cards"]

    def one(reg):
        toc = rp.toc_lines(C, reg)
        res = rp.call_json(B_SYSTEM, profile_blocks + [{"type": "text", "text":
                           "## Regulation card\n" + json.dumps(cards[reg], indent=1, ensure_ascii=False) +
                           "\n\n## Table of contents (key | pages | title)\n" + "\n".join(toc)}],
                           B_SCHEMA, max_tokens=32000, tag=f"H_B_{reg.replace(' ', '_')}")
        return reg, res

    with ThreadPoolExecutor(8) as pool:
        out = dict(pool.map(one, A["candidates"]))
    for reg, b in out.items():
        v = A["candidates"][reg]
        b["include"] = (v["via"] == "screening" and v["likelihood"] >= sure) or b["applies"]
        b["unresolved_keys"] = [s["key"] for s in b["selected_sections"] if not rp.section_nodes(C, reg, s["key"])]
        log(f"stage B: {reg:<9} p={v['likelihood']:.2f} via={v['via']:<9} applies={b['applies']!s:<5} "
            f"-> {'INCLUDE' if b['include'] else 'drop':<7} {len(b['selected_sections'])} sections"
            + (f" (unresolved: {b['unresolved_keys']})" if b["unresolved_keys"] else ""))
    return out


# ------------------------------------------------------------------------------------------------ Stage C ---------
C_EXTRA = """

Each requirement also gets a test_case: one short sentence on how a test engineer would verify it (measurement,
inspection or test), or null. Paragraphs under "Referenced paragraphs" are context: cite one only when the obligation is
stated in it and binds the product. The style examples come from an expert-written list for a different product and
regulation: match their granularity and phrasing, not their content."""

C_SCHEMA = json.loads(json.dumps(reqagent.SCHEMA))
C_SCHEMA["properties"]["requirements"]["items"]["required"].append("test_case")
C_SCHEMA["properties"]["requirements"]["items"]["properties"]["test_case"] = {"type": ["string", "null"]}


def style_examples(k: int = 12) -> str:
    """Expert requirements from the sub-goal's old version (UN R175): a different regulation, so no leak into the
    dev-case score. Administrative ones (application, approval) are left out."""
    old = json.loads((PKG / "subgoal/dev/requirements_old.json").read_text())
    ok = [r for r in old if not r["source"].get("paragraph", "").startswith(("3.", "4.")) and "manufacturer shall" not in r["text"]]
    step = max(1, len(ok) // k)
    return "\n".join(f"- [{r['source']['regulation']} §{r['source']['paragraph']}] {r['text']}" for r in ok[::step][:k])


def section_jobs(A, B, profile) -> list[tuple]:
    """One job per chunk of selected sections: (reg, name, part nodes, referenced nodes, product context)."""
    C, jobs = A["C"], []
    for reg, b in B.items():
        if not b["include"]:
            continue
        nodes = C.by_reg[(reg, "base")]
        dicts = [{**asdict(n), "id": f"n{i}"} for i, n in enumerate(nodes)]
        pos = {(n.doc, n.label): i for i, n in enumerate(nodes)}
        chosen = set()
        for s in b["selected_sections"]:
            chosen |= {pos[(n.doc, n.label)] for n in rp.section_nodes(C, reg, s["key"])}
        chosen = sorted(i for i in chosen if nodes[i].section_kind != "definition")
        product = {"product_summary": A["product_summary"], "profile": profile,
                   "regulation_mapping": {k: b[k] for k in ("how_it_applies", "part_mapping", "open_options")}}
        for name, part in reqagent.chunks([dicts[i] for i in chosen]):
            refs = [r for r in rp.expand_refs(C, reg, [nodes[int(d["id"][1:])] for d in part])
                    if pos[(r.doc, r.label)] not in chosen][:80]
            jobs.append((reg, name, part, [dicts[pos[(r.doc, r.label)]] for r in refs], product))
    return jobs


def job_prompt(A, job, examples: str) -> str:
    reg, name, part, refs, product = job
    version = part[0]["version"]
    toc = reqagent.toc([{**asdict(n), "id": ""} for n in A["C"].by_reg[(reg, "base")]])
    return (f"## Product profile\n{json.dumps(product, indent=1, ensure_ascii=False)}\n\n"
            f"## Style examples\n{examples}\n\n## Regulation\n{reg}, version {version}\n\n## Table of contents\n{toc}\n\n"
            f"## Section {name}\n\n" + "\n\n".join(reqagent.render(n) for n in part) +
            ("\n\n## Referenced paragraphs (context)\n\n" + "\n\n".join(reqagent.render(n) for n in refs) if refs else ""))


def job_tag(prefix: str, job) -> str:
    return f"{prefix}_{job[0].replace(' ', '_')}_{job[1].replace(' ', '-').replace('§', 's')[:40]}"


def tie(job, reqs: list[dict]) -> list[dict]:
    """Attach regulation, section and the cited node (from the section or its references) to model output."""
    reg, name, part, refs, _ = job
    by_id = {d["id"]: d for d in part + refs}
    return [{**q, "regulation": reg, "section": name, "node": by_id.get(q["node_id"])} for q in reqs]


def stage_c(A, B, profile, log) -> list[dict]:
    examples, jobs = style_examples(), section_jobs(A, B, profile)
    log(f"stage C: {len(jobs)} section calls")

    def one(job):
        res = rp.call_json(reqagent.SYSTEM + C_EXTRA, [{"type": "text", "text": job_prompt(A, job, examples)}], C_SCHEMA,
                           max_tokens=64000, tag=job_tag("H_C", job))
        return tie(job, res["requirements"])

    with ThreadPoolExecutor(8) as pool:
        return [q for qs in pool.map(one, jobs) for q in qs]


# ------------------------------------------------------------------------------------------------ Post ------------
STOP = set("a an the of to in on at for by with and or shall be is are its it this that which as from any all "
           "not no each when where if than then into under such".split())


def _content(norm: str) -> set[str]:
    return {t for t in norm.split() if t not in STOP}


def same_obligation(a: str, b: str) -> bool:
    """Near-duplicate only when the texts are near-identical AND no number or content word differs: '1 minute' vs
    '10 minutes', 'S1' vs 'S2', 'narrowband' vs 'broadband', 'cracking' vs 'distortion' are different obligations."""
    if a == b:
        return True
    return rp._similar(a, b) and not (_content(a) ^ _content(b))


def assemble(qs: list[dict], case_id: str, team: str) -> tuple[dict, list[dict]]:
    reqs, dropped, seen = [], [], {}
    for q in qs:
        n = q["node"]
        if n is None:
            dropped.append({**q, "reason": "unknown node id"}); continue
        norm = rp._norm(q["text"])
        key = (q["regulation"], n["key"])
        if any(same_obligation(s, norm) for s in seen.get(key, [])):
            dropped.append({**q, "node": n["key"], "reason": "near-duplicate"}); continue
        seen.setdefault(key, []).append(norm)
        para = reqagent.full_label(n) + (f". Table {q['table']}" if q.get("table") else "")
        item = {"id": f"R-{len(reqs)+1:04d}", "text": q["text"].strip()[:2000],
                "source": {"regulation": q["regulation"], "version": n["version"], "page": n["page_start"], "paragraph": para}}
        if q.get("test_case"):
            item["test_case"] = q["test_case"]
        reqs.append(item)
    return {"team_id": team, "case_id": case_id, "requirements": reqs}, dropped


def evaluate_run(run: Path, sub: dict, expected: Path) -> str:
    exp = json.loads(expected.read_text())["requirements"]
    names = sorted({r["source"]["regulation"] for r in exp} | {r["source"]["regulation"] for r in sub["requirements"]})
    rows, tm, tp, te = [], 0, 0, 0
    for name in names:
        n_e = sum(r["source"]["regulation"] == name for r in exp)
        n_p = sum(r["source"]["regulation"] == name for r in sub["requirements"])
        m = 0
        if n_e and n_p:
            m = ev.evaluate(run / "submission.json", name, expected)["requirement_level"]["n_matched"]
            (run / f"report_{name.replace(' ', '_')}.md").write_text((run / "report.md").read_text())
        rows.append((name, n_p, n_e, m)); tm += m; tp += n_p; te += n_e
    (run / "report.md").unlink(missing_ok=True); (run / "eval.json").unlink(missing_ok=True)
    P, R = tm / max(1, tp), tm / max(1, te)
    lines = [f"# {run.name}\n", "| regulation | predicted | expected | matched | F2 | recall |", "|---|---|---|---|---|---|"]
    lines += [f"| {n} | {p} | {e} | {m} | {ev.f2(m/max(1,p), m/max(1,e)):.2f} | {m/max(1,e):.2f} |" for n, p, e, m in rows]
    lines.append(f"| **all** | {tp} | {te} | {tm} | **{ev.f2(P, R):.2f}** | **{R:.2f}** |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("case_dir")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--team", default="team-05")
    ap.add_argument("--threshold", type=float, default=0.3, help="screening likelihood to become a candidate")
    ap.add_argument("--sure", type=float, default=0.6, help="at or above this, include without stage B's confirmation")
    a = ap.parse_args(argv)
    reqagent.load_env()
    case = Path(a.case_dir).resolve()
    run = REPO / "runs" / a.run_name
    run.mkdir(parents=True, exist_ok=True)
    log_lines = []
    log = lambda s: (print(s), log_lines.append(s))

    prof_path = REPO / "data/product/product.json"
    saved = json.loads(prof_path.read_text()) if prof_path.exists() else {}
    if saved.get("case_id") == case.name and saved.get("model"):
        base = saved
    else:
        import profile as prof
        base = prof.build_profile(case, use_model=True)
    profile = base.get("model") or base["deterministic"]
    blocks = product_blocks(case, base)

    A = stage_a(blocks, profile, a.threshold, log)
    B = stage_b(A, blocks, a.sure, log)
    qs = stage_c(A, B, profile, log)
    sub, dropped = assemble(qs, case.name, a.team)
    problems = rp.validate(sub)
    log(f"{len(sub['requirements'])} requirements ({len(dropped)} dropped), validation: {'OK' if not problems else problems[:5]}")

    (run / "submission.json").write_text(json.dumps(sub, indent=1, ensure_ascii=False))
    (run / "dropped.json").write_text(json.dumps(dropped, indent=1, ensure_ascii=False, default=str))
    (run / "selection.json").write_text(json.dumps({"stageA": {k: A[k] for k in ("product_summary", "regulations", "candidates", "bm25")},
                                                    "stageB": B}, indent=1, ensure_ascii=False))
    (run / "raw.json").write_text(json.dumps([{k: v for k, v in q.items() if k != "node"} | {"node": (q["node"] or {}).get("key")}
                                              for q in qs], indent=1, ensure_ascii=False))
    subprocess.run([sys.executable, str(PKG / "tools/validate.py"), str(run / "submission.json"), "--strict"])
    expected = case / "expected.json"
    if expected.exists():
        subprocess.run([sys.executable, str(PKG / "tools/score_local.py"), str(run / "submission.json"), "--expected", str(expected)])
        summary = evaluate_run(run, sub, expected)
        (run / "summary.md").write_text(summary + "\n")
        print("\n" + summary)
    (run / "log.txt").write_text("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()
