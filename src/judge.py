"""Local requirement-level judge: match our requirements one-to-one against the expected ones (same obligation:
same subject, condition and constraint), per regulation, with a reason category for every unmatched expected item.
Approximates the platform's requirement-level recall/precision so we can iterate without uploading."""
from __future__ import annotations
import json
from pathlib import Path
from pipeline import call_json

SYSTEM = """You are the grader of a regulatory-requirements extraction task. You get EXPECTED requirements (written by
regulatory experts) and SUBMITTED requirements, both for the same UN Regulation and the same product. Match them
one-to-one: a submitted requirement matches an expected one when it states the SAME engineering obligation — same
subject, same condition, same constraint/limit. Paraphrase is fine; a different limit, a missing condition, or a
requirement that merges several expected obligations into one is NOT a match (a merged one may match at most ONE
of them). Each submitted id can be used at most once, each expected id at most once.

For every expected requirement that has no match, classify why:
- "not_extracted": nothing submitted covers that obligation (paragraph missing or skipped)
- "merged": the obligation is inside a submitted requirement together with other obligations
- "wrong_granularity": submitted requirements split or cut it so none states the full obligation
- "wording": a submitted requirement is about it but changes the subject/condition/constraint
- "wrong_paragraph": a submitted requirement states it but cites a different paragraph
Be strict but fair: the platform uses a language model with the same instruction."""

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["matches", "unmatched_expected"],
    "properties": {
        "matches": {"type": "array", "items": {"type": "object", "additionalProperties": False,
            "required": ["expected_id", "submitted_id"], "properties": {"expected_id": {"type": "string"}, "submitted_id": {"type": "string"}}}},
        "unmatched_expected": {"type": "array", "items": {"type": "object", "additionalProperties": False,
            "required": ["expected_id", "reason", "closest_submitted_id", "note"],
            "properties": {"expected_id": {"type": "string"},
                           "reason": {"type": "string", "enum": ["not_extracted", "merged", "wrong_granularity", "wording", "wrong_paragraph"]},
                           "closest_submitted_id": {"type": ["string", "null"]}, "note": {"type": "string"}}}}}}


def judge(sub: dict, exp: dict, run_dir: Path) -> dict:
    regs = sorted({r["source"]["regulation"] for r in exp["requirements"]} | {r["source"]["regulation"] for r in sub["requirements"]})
    out, tot_m, tot_e, tot_s = {}, 0, 0, 0
    for reg in regs:
        E = [r for r in exp["requirements"] if r["source"]["regulation"] == reg]
        S = [r for r in sub["requirements"] if r["source"]["regulation"] == reg]
        tot_e += len(E); tot_s += len(S)
        if not E or not S:
            out[reg] = {"expected": len(E), "submitted": len(S), "matched": 0, "matches": [], "unmatched_expected":
                        [{"expected_id": e["id"], "reason": "not_extracted", "closest_submitted_id": None, "note": "regulation not submitted"} for e in E]}
            continue
        fmt = lambda r: f"{r['id']} [§{r['source']['paragraph']}] {r['text']}"
        user = [{"type": "text", "text": f"## REGULATION {reg}\n\n## EXPECTED ({len(E)})\n" + "\n".join(fmt(r) for r in E)
                 + f"\n\n## SUBMITTED ({len(S)})\n" + "\n".join(fmt(r) for r in S)}]
        res = call_json(SYSTEM, user, SCHEMA, max_tokens=32000, tag=f"J_{reg.replace(' ', '_')}")
        eids, sids = {e["id"] for e in E}, {s["id"] for s in S}
        seen_e, seen_s, matches = set(), set(), []
        for m in res["matches"]:
            if m["expected_id"] in eids and m["submitted_id"] in sids and m["expected_id"] not in seen_e and m["submitted_id"] not in seen_s:
                matches.append(m); seen_e.add(m["expected_id"]); seen_s.add(m["submitted_id"])
        tot_m += len(matches)
        out[reg] = {"expected": len(E), "submitted": len(S), "matched": len(matches), "matches": matches,
                    "unmatched_expected": [u for u in res["unmatched_expected"] if u["expected_id"] not in seen_e]}
    p, r = tot_m / tot_s if tot_s else 0, tot_m / tot_e if tot_e else 0
    f2 = 5 * p * r / (4 * p + r) if p + r else 0
    summary = {"precision": round(p, 3), "recall": round(r, 3), "f2": round(f2, 3), "matched": tot_m, "submitted": tot_s, "expected": tot_e}
    (run_dir / "judge.json").write_text(json.dumps({"summary": summary, "per_regulation": out}, indent=1, ensure_ascii=False))
    return {"summary": summary, "per_regulation": out}
