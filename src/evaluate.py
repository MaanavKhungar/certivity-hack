"""Score one regulation's extracted requirements against the expected answer for that regulation.
Headline metrics are F2 and recall: the hackathon ranks by F2, with recall as the tie-break.

    python src/evaluate.py runs/<run>/submission.json --regulation "UN R150"

Levels:
  * page        tools/score_local.py of the hackathon package, on the expected file filtered to this regulation
  * paragraph   recall of expected paragraphs (table suffix stripped), precision of cited paragraphs
  * requirement one-to-one matching by Claude, using the platform's rule (same scope, condition and constraint)
Writes eval.json and report.md next to the submission. The report splits misses into "paragraph never cited"
(selection) and "paragraph cited, obligation not matched" (splitting / wording).
"""
from __future__ import annotations
import argparse, importlib.util, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reqagent import MODEL, PKG, load_env

_spec = importlib.util.spec_from_file_location("score_local", PKG / "tools/score_local.py")
score_local = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(score_local)

JUDGE_SYSTEM = """You compare requirement lists for a regulatory-compliance benchmark. EXPECTED requirements were written
by regulatory experts; PREDICTED requirements were produced by a system. Match them one to one: a predicted requirement
matches an expected one when both state the same engineering obligation (same scope, same condition, same constraint
or limit). Wording may differ. A predicted requirement that merges several expected obligations matches at most one of
them. Each expected and each predicted id may appear in at most one pair. Only return pairs you are confident about."""

JUDGE_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["pairs"],
    "properties": {"pairs": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["expected_id", "predicted_id"],
        "properties": {"expected_id": {"type": "string"}, "predicted_id": {"type": "string"}}}}},
}


def pkey(p) -> str | None:
    k = score_local._paragraph_key(p)
    return re.sub(r"\s+table\s+\S+$", "", k) if k else k


def f2(p, r):
    return 0.0 if p + r == 0 else 5 * p * r / (4 * p + r)


def judge(expected: list[dict], predicted: list[dict]) -> list[tuple[str, str]]:
    import anthropic
    load_env()
    fmt = lambda r: f"{r['id']} [{r['source'].get('paragraph')}] {r['text']}"
    user = ("## EXPECTED\n" + "\n".join(fmt(r) for r in expected) +
            "\n\n## PREDICTED\n" + "\n".join(fmt(r) for r in predicted))
    with anthropic.Anthropic().beta.messages.stream(
        model=MODEL, max_tokens=64000, system=JUDGE_SYSTEM, thinking={"type": "adaptive"},
        output_config={"effort": "high", "format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        messages=[{"role": "user", "content": user}],
    ) as stream:
        msg = stream.get_final_message()
    if msg.stop_reason == "refusal":
        raise RuntimeError(f"judge refused: {msg.stop_details}")
    pairs = json.loads(next(b.text for b in msg.content if b.type == "text"))["pairs"]
    e_ids, p_ids, seen_e, seen_p, out = {r["id"] for r in expected}, {r["id"] for r in predicted}, set(), set(), []
    for p in pairs:                                     # enforce one-to-one and valid ids
        e, q = p["expected_id"], p["predicted_id"]
        if e in e_ids and q in p_ids and e not in seen_e and q not in seen_p:
            out.append((e, q)); seen_e.add(e); seen_p.add(q)
    return out


def evaluate(sub_path: Path, regulation: str, expected_path: Path, use_judge=True) -> dict:
    sub = json.loads(sub_path.read_text())
    exp_all = json.loads(expected_path.read_text())
    exp = [r for r in exp_all["requirements"] if r["source"]["regulation"] == regulation]
    pred = [r for r in sub["requirements"] if r["source"]["regulation"] == regulation]

    page = score_local.score({"requirements": pred}, {"requirements": exp})["page_level"]
    exp_paras, pred_paras = {pkey(r["source"].get("paragraph")) for r in exp}, {pkey(r["source"].get("paragraph")) for r in pred}
    para = {"recall": len(exp_paras & pred_paras) / len(exp_paras), "precision": len(exp_paras & pred_paras) / max(1, len(pred_paras)),
            "n_expected": len(exp_paras), "n_cited": len(pred_paras)}
    page["f2"], para["f2"] = f2(page["precision"], page["recall"]), f2(para["precision"], para["recall"])
    res = {"regulation": regulation, "n_expected": len(exp), "n_predicted": len(pred), "page_level": page, "paragraph_level": para}

    lines = [f"# {regulation}: {sub_path.parent.name}\n",
             f"- predicted {len(pred)} / expected {len(exp)}",
             f"- page level: F2 {f2(page['precision'], page['recall']):.2f}  R {page['recall']:.2f}",
             f"- paragraph level: F2 {f2(para['precision'], para['recall']):.2f}  R {para['recall']:.2f}  ({len(exp_paras & pred_paras)}/{len(exp_paras)} expected paragraphs cited, {len(pred_paras)} cited)"]
    if use_judge:
        pairs = judge(exp, pred)
        P, R = len(pairs) / max(1, len(pred)), len(pairs) / len(exp)
        res["requirement_level"] = {"precision": P, "recall": R, "f2": f2(P, R), "n_matched": len(pairs), "pairs": pairs}
        lines.append(f"- **requirement level: F2 {f2(P, R):.2f}  R {R:.2f}** ({len(pairs)} matched)")
        me, mp = {e for e, _ in pairs}, {q for _, q in pairs}
        by_para: dict[str, list[dict]] = {}
        for r in pred:
            by_para.setdefault(pkey(r["source"].get("paragraph")), []).append(r)
        sel = [r for r in exp if r["id"] not in me and pkey(r["source"].get("paragraph")) not in pred_paras]
        split = [r for r in exp if r["id"] not in me and pkey(r["source"].get("paragraph")) in pred_paras]
        lines.append(f"\n## Missed, paragraph never cited ({len(sel)})")
        lines += [f"- {r['id']} p{r['source']['page']} [{r['source'].get('paragraph')}] {r['text']}" for r in sel]
        lines.append(f"\n## Missed, paragraph cited but obligation not matched ({len(split)})")
        for r in split:
            lines.append(f"- {r['id']} p{r['source']['page']} [{r['source'].get('paragraph')}] {r['text']}")
            lines += [f"    - ours {q['id']}{' (matched elsewhere)' if q['id'] in mp else ''}: {q['text']}"
                      for q in by_para.get(pkey(r["source"].get("paragraph")), [])]
        extra = [r for r in pred if r["id"] not in mp]
        lines.append(f"\n## Extra, not matched to any expected ({len(extra)})")
        lines += [f"- {r['id']} p{r['source']['page']} [{r['source'].get('paragraph')}] {r['text']}" for r in extra]
        res["missed_selection"] = [r["id"] for r in sel]
        res["missed_split"] = [r["id"] for r in split]
        res["extra"] = [r["id"] for r in extra]

    (sub_path.parent / "eval.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    (sub_path.parent / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:6]))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--regulation", required=True)
    ap.add_argument("--expected", default=str(PKG / "cases/dev-rear-lamp/expected.json"))
    ap.add_argument("--no-judge", action="store_true")
    a = ap.parse_args(argv)
    evaluate(Path(a.submission), a.regulation, Path(a.expected), use_judge=not a.no_judge)


if __name__ == "__main__":
    main()
