"""Three-stage variant of the product filter (src/regulatory_pipeline_3stage.py), on this repo's model client.

    python src/three_stage.py runs/r148-c1full-v4 runs/r48-c1full-v4 ... --out runs/submission-3stage \
        --expected ../industrial-ai-hackathon/cases/dev-rear-lamp/expected.json

Stage 1 EXTRACT    c1full runs without --product (one call per batch, every obligation, no product); read here
Stage 2 CLASSIFY   one call per batch: each obligation APPLICABLE / CONDITIONAL / NOT_APPLICABLE / UNKNOWN for the
                   product (the template's prompt; product inputs as c1full --product: stage A summary, stage B
                   mapping of the regulation, profile)
Stage 3 TRANSLATE  one call per batch: APPLICABLE + CONDITIONAL obligations -> engineering "shall" requirements, each
                   tied to its obligation (template's prompt)
Writes submission.json (stage 3 texts) and submission_no_translate.json (stage 1 texts of the same selection), with
judge results when --expected is given. Obligations the classifier skipped become UNKNOWN, never dropped silently.
"""
from __future__ import annotations
import argparse, json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c1full
import pipeline as rp
import reqagent

# prompts from regulatory_pipeline_3stage.py, output shapes moved into the JSON schemas below
CLASSIFY = """For EACH supplied obligation, determine applicability to THIS product using the supplied profile.
Statuses: APPLICABLE (product evidence establishes scope), CONDITIONAL (an explicit open design option
triggers scope), NOT_APPLICABLE (product evidence establishes exclusion), UNKNOWN (insufficient data).
Include own product functions and relevant general installation obligations; exclude unrelated vehicle
functions/signals and other categories only with evidence. Absence of mention is NOT evidence of absence.
Never alter obligation text. Every input obligation must produce exactly one classification."""

TRANSLATE = """For each APPLICABLE/CONDITIONAL obligation, produce a faithful atomic engineering 'shall'
requirement. Preserve numeric limits, conditions, exceptions, source references, and parent IDs.
Do not create thresholds, tests, designs, or facts not in the regulation. If verification method is not
prescribed, leave it null and record an open question. Never silently merge distinct obligations."""

# v4p-aligned stages 2 and 3: the classifier applies the single-stage v4p product filter (its rules are taken from
# c1full.WITH_PRODUCT_V4, so the two cannot drift apart); the translator keeps the stage-1 text, which is already
# written to the v4 extraction rules, as v4p does
_V4_HEAD = "not from what the regulation allows in general:\n"
_V4_TAIL = "\nKeep each output requirement's own conditions in its text."
assert _V4_HEAD in c1full.WITH_PRODUCT_V4 and c1full.WITH_PRODUCT_V4.endswith(_V4_TAIL[1:])
V4P_RULES = c1full.WITH_PRODUCT_V4.split(_V4_HEAD, 1)[1].rsplit(_V4_TAIL, 1)[0]

CLASSIFY_V4P = """For EACH supplied obligation (extracted from a UN Regulation without looking at the product), decide
whether it binds THIS product. You receive the product (summary, profile, and an earlier analysis of how this
regulation applies to it: parts mapped to the regulation's terms, relevant sections, open options).
Apply the product filter below. OUTPUT means APPLICABLE, or CONDITIONAL when the obligation's condition is one of the
listed open_options; SKIP means NOT_APPLICABLE. Decide from the product data, not from what the regulation allows in
general:
""" + V4P_RULES + """
Use UNKNOWN only when the obligation's subject cannot be identified from its text, not for doubtful applicability:
decide OUTPUT or SKIP with the rules above.
Never alter obligation text. Every input obligation must produce exactly one classification."""

TRANSLATE_V4P = """Each supplied obligation is already written as an engineering requirement (one obligation per
requirement, the regulation's wording, its conditions and exceptions kept) and was classified APPLICABLE or
CONDITIONAL for this product. Return exactly one requirement per obligation, with its obligation_id:
- Keep the obligation's text as it is: same wording, numbers, units, conditions, exceptions and references. Do not
  split, merge, shorten, paraphrase or add detail.
- Only when the text is not understandable on its own (it starts with "it", "they", "this"), name the subject instead.
- For a CONDITIONAL obligation whose text does not state its condition, put the condition from its decision in front
  ("In case of ..., ...").
- verification_method: null unless the obligation itself prescribes the test; open_questions: [] unless the text is
  ambiguous."""

PROMPT_SETS = {"template": (CLASSIFY, TRANSLATE), "v4p": (CLASSIFY_V4P, TRANSLATE_V4P)}

STATUSES = ["APPLICABLE", "CONDITIONAL", "NOT_APPLICABLE", "UNKNOWN"]
CLASSIFY_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["decisions"], "properties": {
    "decisions": {"type": "array", "items": {"type": "object", "additionalProperties": False,
        "required": ["obligation_id", "status", "reason", "product_evidence"],
        "properties": {"obligation_id": {"type": "string"}, "status": {"type": "string", "enum": STATUSES},
                       "reason": {"type": "string"}, "product_evidence": {"type": "array", "items": {"type": "string"}}}}}}}
TRANSLATE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["requirements"], "properties": {
    "requirements": {"type": "array", "items": {"type": "object", "additionalProperties": False,
        "required": ["obligation_id", "text", "verification_method", "open_questions"],
        "properties": {"obligation_id": {"type": "string"}, "text": {"type": "string"},
                       "verification_method": {"type": ["string", "null"]},
                       "open_questions": {"type": "array", "items": {"type": "string"}}}}}}}


def obligation(r: dict) -> dict:
    return {"id": r["id"], "text": r["text"], "paragraph": r["source"]["paragraph"], "page": r["source"]["page"]}


def run_reg(d: Path, selection: Path, profile: Path, log, prompts: str = "template") -> dict:
    m = json.loads((d / "extraction_manifest.json").read_text())
    if m.get("product_filter"):
        raise SystemExit(f"{d}: stage 1 must be a run WITHOUT --product")
    reg = m["regulation"]
    reqs = json.loads((d / "requirements_with_context.json").read_text())["requirements"]
    product = c1full.product_block(selection, profile, reg)
    classify, translate = PROMPT_SETS[prompts]
    return classify_translate(reg, m["version"], reqs, product, classify, translate, log=log)


def classify_translate(reg: str, version: str, reqs: list, product: str, classify: str = CLASSIFY,
                       translate: str = TRANSLATE, on_batch=None, translate_step: bool = True, workers: int = 8,
                       log=print) -> dict:
    """Stages 2 and 3 for one regulation's stage-1 obligations (c1full candidates with "batch", "id", "text",
    "source"). classify / translate: prompt overrides (dashboard); on_batch(stage, batch id, info): progress callback;
    translate_step=False stops after classification."""
    batches: dict[str, list] = {}
    for r in reqs:
        batches.setdefault(r["batch"], []).append(r)

    def one(item):
        bid, rs = item
        tag = f"3S_{c1full.short(reg)}_{bid}"
        user = product + f"## Regulation\n{reg}, version {version}\n\n## Obligations ({len(rs)})\n" + \
            json.dumps([obligation(r) for r in rs], indent=1, ensure_ascii=False)
        content = [{"type": "text", "text": user}]
        hit = c1full.cached(classify, content, CLASSIFY_SCHEMA, tag + "_classify")
        dec = rp.call_json(classify, content, CLASSIFY_SCHEMA, max_tokens=64000, tag=tag + "_classify")
        by = {x["obligation_id"]: x for x in dec["decisions"]}
        for r in rs:                                      # never lose an obligation to imperfect output
            by.setdefault(r["id"], {"obligation_id": r["id"], "status": "UNKNOWN", "reason": "classification missing",
                                    "product_evidence": []})
        sel = [r for r in rs if by[r["id"]]["status"] in ("APPLICABLE", "CONDITIONAL")]
        if on_batch:
            on_batch("classify", bid, {"obligations": len(rs), "selected": len(sel), "cache_hit": hit,
                                       "counts": {s: sum(by[r["id"]]["status"] == s for r in rs) for s in STATUSES}})
        out = []
        if sel and translate_step:
            user = product + f"## Regulation\n{reg}, version {version}\n\n## Obligations\n" + \
                json.dumps([obligation(r) for r in sel], indent=1, ensure_ascii=False) + "\n\n## Decisions\n" + \
                json.dumps([by[r["id"]] for r in sel], indent=1, ensure_ascii=False)
            content = [{"type": "text", "text": user}]
            hit = c1full.cached(translate, content, TRANSLATE_SCHEMA, tag + "_translate")
            res = rp.call_json(translate, content, TRANSLATE_SCHEMA, max_tokens=64000, tag=tag + "_translate")
            out = res["requirements"]
            if on_batch:
                on_batch("translate", bid, {"requirements": len(out), "cache_hit": hit})
        return bid, rs, by, sel, out

    with ThreadPoolExecutor(workers) as pool:
        parts = list(pool.map(one, sorted(batches.items())))
    decisions, selected, translated, issues = {}, [], [], []
    src = {r["id"]: r for r in reqs}
    for bid, rs, by, sel, out in parts:
        decisions.update({r["id"]: by[r["id"]] for r in rs})
        selected += sel
        ok = {r["id"] for r in sel}
        for t in out:
            if t["obligation_id"] not in ok:
                issues.append({"batch": bid, "note": "requirement with unknown parent id", "item": t}); continue
            translated.append({**t, "source": src[t["obligation_id"]]["source"]})
        if translate_step:
            covered = {t["obligation_id"] for t in out}
            issues += [{"batch": bid, "obligation_id": r["id"], "note": "selected obligation without requirement"}
                       for r in sel if r["id"] not in covered]
    counts = {s: sum(x["status"] == s for x in decisions.values()) for s in STATUSES}
    log(f"{reg}: {len(reqs)} obligations -> {counts} -> {len(translated)} requirements ({len(issues)} review items)")
    return {"regulation": reg, "decisions": decisions, "selected": selected, "translated": translated, "issues": issues}


def submission(items: list, case: str, team: str) -> dict:
    return {"team_id": team, "case_id": case, "requirements": [
        {"id": f"R-{i + 1:04d}", "text": x["text"], "source": {k: x["source"][k] for k in ("regulation", "version", "page", "paragraph")}}
        for i, x in enumerate(items)]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="c1full runs WITHOUT --product (stage 1), one regulation each")
    ap.add_argument("--selection", default=str(reqagent.REPO / "runs/rearlamp-v7b/selection.json"))
    ap.add_argument("--profile", default=str(reqagent.REPO / "data/product/dev-rear-lamp.profile.json"))
    ap.add_argument("--case", default="dev-rear-lamp")
    ap.add_argument("--team", default="team-05")
    ap.add_argument("--out", required=True)
    ap.add_argument("--expected")
    ap.add_argument("--prompts", choices=sorted(PROMPT_SETS), default="template", help="stage 2/3 prompts: the template's or v4p-aligned")
    a = ap.parse_args(argv)
    reqagent.load_env()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(5) as pool:
        res = list(pool.map(lambda d: run_reg(Path(d), Path(a.selection), Path(a.profile), print, a.prompts), a.runs))
    (out / "three_stage.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    subs = {"submission.json": submission([t for r in res for t in r["translated"]], a.case, a.team),
            "submission_no_translate.json": submission([s for r in res for s in r["selected"]], a.case, a.team)}
    for name, sub in subs.items():
        (out / name).write_text(json.dumps(sub, indent=1, ensure_ascii=False))
        problems = rp.validate(sub)
        print(f"{name}: {len(sub['requirements'])} requirements, validation {'OK' if not problems else problems[:3]}")
        if a.expected:
            import judge
            d = out / ("judge_" + name.replace(".json", ""))
            d.mkdir(exist_ok=True)
            J = judge.judge(sub, json.loads(Path(a.expected).read_text()), d)
            s = J["summary"]
            print(f"  judge (whole data): submitted {s['submitted']}, matched {s['matched']}/{s['expected']}, "
                  f"precision {s['precision']:.3f}, recall {s['recall']:.3f}, F2 {s['f2']:.3f}")
            for reg, v in J["per_regulation"].items():
                print(f"    {reg:<8} matched {v['matched']:>3}/{v['expected']:<3} of {v['submitted']} submitted")


if __name__ == "__main__":
    main()
