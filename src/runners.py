"""End-to-end submission runners: case folder (spec.md + exploded view) -> validated submission.json.

    python src/runners.py v1 <case_dir> --run-name bumper-v1 --team <team_id>

v1  full-PDF C1 + product-aware C1, agreement + paragraph D                      (rear lamp: P .382 R .900 F2 .708)
    profile -> A/B -> T -> product-aware C1 (bbpipe) ; full-PDF C1 (c1full) per included regulation ;
    keep a full-PDF candidate when its paragraph is in a B-selected, non-administrative section, the paragraph-D
    category is not other-device / vehicle-level-for-component / framework, the product-aware C1 also wrote from that
    paragraph, and the regulation's stage-A probability is >= 0.5
v2  T -> C1 -> stage D (hybrid7 reviewer)                                        (rear lamp: P .390 R .832 F2 .678)
v3  T -> C1 -> logistic-regression scorer (data/scorer/dscore_model.json, fitted on the rear lamp's labels),
    top KEEP_SHARE of candidates per regulation                                   (rear lamp, out-of-fold: F2 ~.648)

Every variant builds the profile (vision) and runs stage A/B itself unless --profile / --selection are given, writes
runs/<run-name>/submission.json and validates it. expected.json, when the case has one, is used only for scoring.
"""
from __future__ import annotations
import argparse, json, re, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bbpipe
import pipeline as rp
import reqagent
from regparse import Corpus, paragraph_key

RUNS = reqagent.REPO / "runs"
ZERO_CATEGORIES = {"other_device_or_function", "vehicle_level_for_component", "framework_or_summary"}
MIN_REG_PROB = 0.5
KEEP_SHARE = 0.9


def base_args(a, run_name, extra):
    argv = [str(a.case_dir), "--run-name", run_name, "--team", a.team, "--workers", str(a.workers)] + extra
    if a.profile:
        argv += ["--profile", a.profile]
    if a.selection:
        argv += ["--selection", a.selection]
    return argv


def product_c1(a, run_name) -> Path:
    """profile -> A/B -> T -> product-aware C1 (scored on its own when the case has expected.json)."""
    bbpipe.main(base_args(a, run_name, ["--stop-after", "c1"]))
    return RUNS / run_name


def selection_of(a, run: Path) -> dict:
    return json.loads(Path(a.selection).read_text()) if a.selection else json.loads((run / "selection.json").read_text())


def profile_of(a, run: Path) -> dict:
    base = json.loads(Path(a.profile).read_text()) if a.profile else json.loads((run / "profile.json").read_text())
    return base.get("model") or base.get("deterministic") or base


def finish(sub: dict, run: Path, case: Path, name="submission.json"):
    (run / name).write_text(json.dumps(sub, indent=1, ensure_ascii=False))
    problems = rp.validate(sub)
    print(f"\n{len(sub['requirements'])} requirements -> {run / name}; validation: {'OK' if not problems else problems[:5]}")
    exp = case / "expected.json"
    if exp.exists():
        import judge
        d = run / ("judge_" + name.replace(".json", ""))
        d.mkdir(exist_ok=True)
        J = judge.judge(sub, json.loads(exp.read_text()), d)["summary"]
        print(f"judge: submitted {J['submitted']}, matched {J['matched']}, P {J['precision']:.3f}, R {J['recall']:.3f}, F2 {J['f2']:.3f}")


# ------------------------------------------------------------------------------------------------ v1 -------------
def run_v1(a):
    import c1full, dpara
    run = product_c1(a, a.run_name)
    sel, profile = selection_of(a, run), profile_of(a, run)
    A, B = sel["stageA"], sel["stageB"]
    regs = [r for r, b in B.items() if b.get("include") in (True, "True")]
    print(f"v1: full-PDF C1 for {regs}")

    def full(reg):
        name = f"{a.run_name}-c1full-{reg.replace('UN ', '').replace(' ', '')}"
        c1full.main([reg, "--role", "base", "--run-name", name, "--workers", str(a.workers)])
        return reg, json.loads((RUNS / name / "requirements_with_context.json").read_text())["requirements"]

    with ThreadPoolExecutor(3) as pool:
        by_reg = dict(pool.map(full, regs))
    C = Corpus(reqagent.REPO / "data/pdf")
    with ThreadPoolExecutor(5) as pool:
        cats = dict(zip(regs, pool.map(lambda reg: dpara.score_paragraphs(reg, by_reg[reg], C, A, B[reg], profile), regs)))
    (run / "v1_paragraph_scores.json").write_text(json.dumps(cats, indent=1, ensure_ascii=False))
    pk = lambda reg, para: (reg, paragraph_key(re.sub(r"\.?\s*Table\s+\S+$", "", str(para))))
    agree = {pk(c["source"]["regulation"], c["source"]["paragraph"])
             for c in json.loads((run / "c1_candidates.json").read_text())["candidates"]}
    pA = {r: v["likelihood"] for r, v in A["candidates"].items()}
    kept, why = [], {}
    for reg in regs:
        gate = dpara.section_gate(C, reg, B[reg])
        for r in by_reg[reg]:
            q = dpara.para_key(r)
            reason = ("outside B's sections" if dpara.label_of_q(q) not in gate else
                      "paragraph category " + cats[reg][q]["category"] if cats[reg][q]["category"] in ZERO_CATEGORIES else
                      "product-aware C1 wrote nothing from this paragraph" if pk(reg, r["source"]["paragraph"]) not in agree else
                      "weak regulation (stage A)" if pA.get(reg, 1) < MIN_REG_PROB else None)
            why[r["id"]] = reason
            if reason is None:
                kept.append(r)
    (run / "v1_filter.json").write_text(json.dumps(why, indent=1))
    print(f"v1: kept {len(kept)} of {sum(len(v) for v in by_reg.values())} full-PDF candidates")
    finish(dpara.submission(kept, Path(a.case_dir).name, a.team), run, Path(a.case_dir).resolve())


# ------------------------------------------------------------------------------------------------ v2 -------------
def run_v2(a):
    bbpipe.main(base_args(a, a.run_name, ["--final", "d"]))
    run = RUNS / a.run_name
    print(f"\nv2 submission: {run / 'submission.json'} (scored above when the case has expected.json)")


# ------------------------------------------------------------------------------------------------ v3 -------------
def run_v3(a):
    import dscore
    run = product_c1(a, a.run_name)
    sel = selection_of(a, run)
    cands = json.loads((run / "c1_candidates.json").read_text())["candidates"]
    items = {i["id"]: i for i in json.loads((run / "traversal.json").read_text())["items"]}
    model = json.loads(dscore.MODEL_PATH.read_text())
    reg_prob = {r: v["likelihood"] for r, v in sel["stageA"]["candidates"].items()}
    llm = dscore.llm_features(cands, items, sel["stageB"], sel["stageA"]["product_summary"])
    sims = dscore.similarity(cands, dscore.expert_refs(exclude_case=Path(a.case_dir).name))
    names, X = dscore.feature_rows(cands, items, llm, sims, reg_prob)
    cols = [names.index(f) for f in model["features"]]
    p = dscore.predict(model, [[x[k] for k in cols] for x in X])
    score = {c["id"]: pp for c, pp in zip(cands, p)}
    (run / "v3_scores.json").write_text(json.dumps(score, indent=1))
    kept = []
    for reg in sorted({c["source"]["regulation"] for c in cands}):
        cs = sorted([c for c in cands if c["source"]["regulation"] == reg], key=lambda c: -score[c["id"]])
        kept += cs[:max(1, round(len(cs) * KEEP_SHARE))]
    keep = {c["id"] for c in kept}
    kept = [c for c in cands if c["id"] in keep]
    sub, _ = bbpipe.submission(kept, [{"requirement_id": c["id"], "decision": "applicable"} for c in kept], {"applicable"},
                               a.team, Path(a.case_dir).name)
    print(f"v3: kept {len(kept)} of {len(cands)} (top {KEEP_SHARE:.0%} per regulation)")
    finish(sub, run, Path(a.case_dir).resolve())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=["v1", "v2", "v3"])
    ap.add_argument("case_dir")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--team", default="team-05")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--profile", help="saved profile JSON (skips the vision call)")
    ap.add_argument("--selection", help="saved selection.json (skips stage A/B)")
    a = ap.parse_args(argv)
    bbpipe.load_env()
    {"v1": run_v1, "v2": run_v2, "v3": run_v3}[a.variant](a)


if __name__ == "__main__":
    main()
