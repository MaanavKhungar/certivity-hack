"""Stage D scorer: every C1 candidate gets p = P(an expert lists it), from categorical LLM judgments + structural
features + similarity to expert-written requirements, combined by a logistic regression fitted on a judged run.

    python src/dscore.py fit runs/rearlamp-bb2 ../industrial-ai-hackathon/cases/dev-rear-lamp     # fit + LORO report
    (bbpipe --final d3 applies data/scorer/dscore_model.json to a new case)

Features per candidate
  LLM (one call per regulation, candidates grouped by section; categorical answers + one probability):
      bound      product | vehicle_installation | manufacturer_or_paperwork | authority_or_service | laboratory
      nature     design_or_performance | test_severity_or_pass_criterion | test_procedure | permission_or_exemption |
                 marking | announcement_or_summary | definition_or_scope
      part_link  confirmed | open_option | not_indicated | general
      duplicate_of, expert_prob, section score
  structure:  annex, section kind, paragraph depth, conditional/permission wording, hop, stage-A regulation
              probability, candidates in the same paragraph, text length
  similarity: max TF-IDF cosine to expert-written requirements (UN R175 sub-goal set + labelled cases' expected
              lists, never the case being scored nor, in evaluation, the same regulation)
Evaluation: leave-one-regulation-out (train on the other regulations, predict the held-out one), pooled out-of-fold
AUC and F2 per threshold. With calibrated p, the F2-optimal rule is keep p > F2/5.
"""
from __future__ import annotations
import argparse, json, math, re, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline as rp
import reqagent

MODEL_PATH = reqagent.REPO / "data/scorer/dscore_model.json"

BOUND = ["product", "vehicle_installation", "manufacturer_or_paperwork", "authority_or_service", "laboratory"]
NATURE = ["design_or_performance", "test_severity_or_pass_criterion", "test_procedure", "permission_or_exemption",
          "marking", "announcement_or_summary", "definition_or_scope"]
PART = ["confirmed", "open_option", "not_indicated", "general"]

D3_SYSTEM = """You classify candidate engineering requirements extracted from ONE UN Regulation for a specific
product. The candidates are grouped by the section they come from. Answer for every candidate, from the regulation
text and the product information only:

bound: who must do or fulfil it?
  product                    the device itself: design, construction, performance, durability, its markings
  vehicle_installation       how the device is installed on / works with the vehicle (position, visibility,
                             switching, tell-tales), where it concerns this kind of device
  manufacturer_or_paperwork  documents, declarations, applications, conformity-of-production administration
  authority_or_service       the approval authority or technical service
  laboratory                 how the test laboratory proceeds
nature: what kind of statement is it?
  design_or_performance, test_severity_or_pass_criterion (a stress the product must withstand or a result it must
  achieve), test_procedure (set-up, equipment, sequence, handling), permission_or_exemption, marking,
  announcement_or_summary (only points to or repeats other paragraphs), definition_or_scope
part_link: how does it relate to the product?
  confirmed      concerns a part or function the product has (see the part mapping)
  open_option    concerns one of the open options (a feature the inputs do not settle)
  not_indicated  concerns a function, device, class or feature nothing in the inputs suggests the product has
  general        applies to any product of this kind
duplicate_of: the id of another candidate stating the same obligation (the clearer one), else null
expert_prob: probability (0-1) that an expert engineer lists this candidate as its own requirement for this product
For every section also give expert_section: probability that the expert lists anything from that section."""

D3_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["sections", "candidates"], "properties": {
    "sections": {"type": "array", "items": {"type": "object", "additionalProperties": False,
        "required": ["section", "expert_section"],
        "properties": {"section": {"type": "string"}, "expert_section": {"type": "number"}}}},
    "candidates": {"type": "array", "items": {"type": "object", "additionalProperties": False,
        "required": ["id", "bound", "nature", "part_link", "duplicate_of", "expert_prob"],
        "properties": {"id": {"type": "string"}, "bound": {"type": "string", "enum": BOUND},
                       "nature": {"type": "string", "enum": NATURE}, "part_link": {"type": "string", "enum": PART},
                       "duplicate_of": {"type": ["string", "null"]}, "expert_prob": {"type": "number"}}}}}}


# ------------------------------------------------------------------------------------------------ LLM part ------
def llm_features(cands, items, B, product_summary, workers=6) -> dict:
    by_reg = {}
    for c in cands:
        by_reg.setdefault(c["source"]["regulation"], []).append(c)

    def one(reg):
        b, groups = B[reg], {}
        for c in by_reg[reg]:
            groups.setdefault(c.get("item") or "?", []).append(c)
        lines = []
        for iid, cs in groups.items():
            it = items.get(iid, {})
            lines.append(f"\n### Section {it.get('key', iid)}: {it.get('title') or ''} [kind: {it.get('kind', '?')}]")
            lines += [f"{c['id']} [§{c['source']['paragraph']}] {c['text']}" for c in cs]
        user = ("## Product\n" + product_summary + "\n\n## How this regulation applies (earlier analysis)\n" +
                json.dumps({k: b[k] for k in ("how_it_applies", "part_mapping", "open_options")}, indent=1, ensure_ascii=False) +
                f"\n\n## Candidates from {reg} ({len(by_reg[reg])}), by section\n" + "\n".join(lines) +
                "\n\nAnswer for every section and every candidate id.")
        res = rp.call_json(D3_SYSTEM, [{"type": "text", "text": user}], D3_SCHEMA, max_tokens=64000, tag=f"BB_D3_{reg.replace(' ', '_')}")
        sec = {s["section"]: s["expert_section"] for s in res["sections"]}
        out = {d["id"]: d for d in res["candidates"]}
        for c in by_reg[reg]:
            k = items.get(c.get("item") or "", {}).get("key", "")
            d = out.setdefault(c["id"], {"id": c["id"], "bound": "product", "nature": "design_or_performance",
                                         "part_link": "general", "duplicate_of": None, "expert_prob": 0.5, "missing": True})
            d["expert_section"] = next((v for s, v in sec.items() if s == k or s.endswith(" " + k) or (k and k in s)), 0.5)
        return out

    res = {}
    with ThreadPoolExecutor(workers) as pool:
        for r in pool.map(one, sorted(by_reg)):
            res.update(r)
    return res


# ------------------------------------------------------------------------------------------------ similarity ----
def expert_refs(exclude_case: str | None = None) -> list[dict]:
    """Expert-written requirements: UN R175 sub-goal set + every case's expected.json except the one being scored."""
    refs = [{"reg": r["source"]["regulation"], "text": r["text"]}
            for r in json.loads((rp.PKG / "subgoal/dev/requirements_old.json").read_text())]
    for exp in sorted((rp.PKG / "cases").glob("*/expected.json")):
        if exp.parent.name != exclude_case:
            refs += [{"reg": r["source"]["regulation"], "text": r["text"]} for r in json.loads(exp.read_text())["requirements"]]
    return refs


def similarity(cands, refs) -> list[float]:
    """Max cosine to a reference written for a DIFFERENT regulation (no leak from the regulation being scored)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    if not refs:
        return [0.0] * len(cands)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True).fit([r["text"] for r in refs] + [c["text"] for c in cands])
    M = cosine_similarity(vec.transform([c["text"] for c in cands]), vec.transform([r["text"] for r in refs]))
    regs = [r["reg"] for r in refs]
    out = []
    for i, c in enumerate(cands):
        row = [M[i, j] for j in range(len(refs)) if regs[j] != c["source"]["regulation"]]
        out.append(float(max(row)) if row else 0.0)
    return out


# ------------------------------------------------------------------------------------------------ features ------
COND = re.compile(r"^(In case|In the case|If|Where|When|For)\b")
PERM = re.compile(r"\bmay\b|allowed|permitted|need not|does not apply|do not apply")


def feature_rows(cands, items, llm, sims, reg_prob) -> tuple[list[str], list[list[float]]]:
    per_para = {}
    for c in cands:
        k = (c["source"]["regulation"], c["source"]["paragraph"])
        per_para[k] = per_para.get(k, 0) + 1
    rows = []
    for c, sim in zip(cands, sims):
        it, d = items.get(c.get("item") or "", {}), llm[c["id"]]
        para = str(c["source"]["paragraph"])
        f = {"expert_prob": d["expert_prob"], "expert_section": d["expert_section"], "duplicate": float(bool(d["duplicate_of"])),
             **{f"bound_{b}": float(d["bound"] == b) for b in BOUND[:3]},
             **{f"nature_{n}": float(d["nature"] == n) for n in NATURE},
             **{f"part_{p}": float(d["part_link"] == p) for p in PART[:3]},
             "annex": float(para.startswith("Annex")), "kind_requirement": float(it.get("kind") == "requirement"),
             "kind_test": float(it.get("kind") == "test"), "depth": float(re.sub(r"^Annex \S+( Part \S+)? ?", "", para).count(".")),
             "conditional": float(bool(COND.match(c["text"]))), "permission": float(bool(PERM.search(c["text"]))),
             "hop": float(it.get("hop", 0)), "reg_prob": float(reg_prob.get(c["source"]["regulation"], 0.5)),
             "same_paragraph": math.log1p(per_para[(c["source"]["regulation"], c["source"]["paragraph"])]),
             "length": math.log1p(len(c["text"])), "sim_expert": sim}
        rows.append(f)
    names = list(rows[0]) if rows else []
    return names, [[r[n] for n in names] for r in rows]


# ------------------------------------------------------------------------------------------------ model ---------
def fit_lr(X, y, C=0.5):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    X = np.asarray(X, float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    m = LogisticRegression(C=C, max_iter=2000).fit((X - mu) / sd, y)
    return {"mu": mu.tolist(), "sd": sd.tolist(), "coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0])}


def predict(model, X) -> list[float]:
    out = []
    for x in X:
        z = model["intercept"] + sum(c * (v - m) / s for c, v, m, s in zip(model["coef"], x, model["mu"], model["sd"]))
        out.append(1 / (1 + math.exp(-z)))
    return out


def f2_curve(p, y, n_expected, thresholds=(0.05, 0.1, 0.14, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5)):
    rows = []
    for t in thresholds:
        keep = [yy for pp, yy in zip(p, y) if pp >= t]
        m, n = sum(keep), len(keep)
        P, R = (m / n if n else 0), m / n_expected
        rows.append({"threshold": t, "submitted": n, "matched": m, "precision": round(P, 3), "recall": round(R, 3),
                     "f2": round(5 * P * R / (4 * P + R), 3) if P + R else 0})
    return rows


def auc(p, y):
    pos = [a for a, b in zip(p, y) if b]; neg = [a for a, b in zip(p, y) if not b]
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg)) if pos and neg else None


# ------------------------------------------------------------------------------------------------ fit CLI -------
def labels(run: Path) -> set:
    """Candidate ids the judge matched in the run's C1-only submission (R-000k = k-th candidate)."""
    sub = json.loads((run / "submission_c1_only.json").read_text())["requirements"]
    J = json.loads((run / "judge_c1_only/judge.json").read_text())["per_regulation"]
    rid2c = {r["id"]: f"C-{i + 1:04d}" for i, r in enumerate(sub)}
    return {rid2c[m["submitted_id"]] for v in J.values() for m in v["matches"]}


def prepare(run: Path, case: Path, selection: Path):
    cands = json.loads((run / "c1_candidates.json").read_text())["candidates"]
    items = {i["id"]: i for i in json.loads((run / "traversal.json").read_text())["items"]}
    sel = json.loads(selection.read_text())
    reg_prob = {r: v["likelihood"] for r, v in sel["stageA"]["candidates"].items()}
    llm = llm_features(cands, items, sel["stageB"], sel["stageA"]["product_summary"])
    sims = similarity(cands, expert_refs(exclude_case=case.name))
    names, X = feature_rows(cands, items, llm, sims, reg_prob)
    return cands, names, X, llm


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit"])
    ap.add_argument("run")
    ap.add_argument("case")
    ap.add_argument("--selection", default=str(reqagent.REPO / "runs/rearlamp-v7b/selection.json"))
    ap.add_argument("--C", type=float, default=0.5)
    a = ap.parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bbpipe
    bbpipe.load_env()
    run, case = Path(a.run).resolve(), Path(a.case).resolve()
    cands, names, X, llm = prepare(run, case, Path(a.selection))
    good = labels(run)
    y = [int(c["id"] in good) for c in cands]
    n_exp = len(json.loads((case / "expected.json").read_text())["requirements"])
    regs = sorted({c["source"]["regulation"] for c in cands})
    oof = [0.0] * len(cands)
    for r in regs:                                        # leave one regulation out
        tr = [i for i, c in enumerate(cands) if c["source"]["regulation"] != r]
        te = [i for i, c in enumerate(cands) if c["source"]["regulation"] == r]
        m = fit_lr([X[i] for i in tr], [y[i] for i in tr], a.C)
        for i, pp in zip(te, predict(m, [X[i] for i in te])):
            oof[i] = pp
    model = fit_lr(X, y, a.C)
    model.update({"features": names, "trained_on": [str(run)], "n": len(y), "positives": sum(y), "C": a.C})
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_text(json.dumps(model, indent=1))
    single = {n: auc([x[k] for x in X], y) for k, n in enumerate(names)}
    report = {"oof_auc": auc(oof, y), "llm_expert_prob_auc": single["expert_prob"], "curve_oof": f2_curve(oof, y, n_exp),
              "per_regulation_auc": {r: auc([oof[i] for i, c in enumerate(cands) if c["source"]["regulation"] == r],
                                            [y[i] for i, c in enumerate(cands) if c["source"]["regulation"] == r]) for r in regs},
              "coef": sorted(zip(names, model["coef"]), key=lambda t: -abs(t[1])), "single_feature_auc": single}
    (run / "dscore_report.json").write_text(json.dumps(report, indent=1))
    (run / "dscore_llm.json").write_text(json.dumps(llm, indent=1, ensure_ascii=False))
    (run / "dscore_oof.json").write_text(json.dumps({c["id"]: round(p, 4) for c, p in zip(cands, oof)}, indent=1))
    print(f"OOF AUC {report['oof_auc']:.3f}  (LLM expert_prob alone {single['expert_prob']:.3f})")
    print("per regulation:", {r: (round(v, 3) if v else v) for r, v in report["per_regulation_auc"].items()})
    for row in report["curve_oof"]:
        print(row)
    print("top weights:", [(n, round(c, 2)) for n, c in report["coef"][:10]])


if __name__ == "__main__":
    main()
