"""Hybrid pipeline v7: v6 (hybrid.py) + the second-reader and filter passes of pipeline.py v2.

    python src/hybrid7.py ../industrial-ai-hackathon/cases/dev-rear-lamp --run-name rearlamp-v7 \
        --profile data/product/dev-rear-lamp.profile.json

Profile  spec + vision on the exploded view (callouts as hints)                        [profile.py]
A        select regulations: cards -> likelihood, supersession + light-source rules, BM25     [hybrid.stage_a]
B        per candidate: applies gate, parts -> defined terms, open options, section keys;
         table of contents now carries section-kind hints and annex references (pipeline v2);
         nested selections collapsed                                                    [hybrid.stage_b + here]
T        traversal: selected sections + references (same PDF up to 3 hops, other PDFs 1 hop), with origin per
         item; written to traversal.json (C does not read it yet)                        [traverse.py]
C        per section chunk: v3 extraction rules + mapping + references + R175 style examples  [hybrid.stage_c]
C2       per section chunk: second reader adds obligations C missed or merged           [here, idea from pipeline v2]
Post     dedupe, page/paragraph from parsed nodes                                       [hybrid.assemble]
D        per regulation: senior reviewer keeps / drops / merges candidates              [here, idea from pipeline v2]
Final    renumber, validate, score (our judge + pipeline's judge)

Not taken from pipeline v2: the rule that one limit over several bands is one requirement (it merged R10's per-pulse
test levels the experts split), and the style examples taken from the dev case's expected answer.
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hybrid as H
import pipeline as rp
import reqagent
from extend_profile import product_blocks
import traverse

# ------------------------------------------------------------------------------------------------ B additions -----
H.B_SYSTEM += """

Each table-of-contents line carries a rule-based kind hint in brackets ([requirement], [test], [conformity],
[administrative], [definition]) and, for annexes, which main-body paragraphs refer to it. The hints can be wrong
(judge by the title): use them to skip administrative content and annexes no selected section refers to."""


def collapse_nested(B) -> None:
    """A selected sub-section whose parent chapter/annex is also selected is already covered."""
    for b in B.values():
        keys = [rp.paragraph_key(s["key"]) for s in b["selected_sections"]]
        b["selected_sections"] = [s for s, k in zip(b["selected_sections"], keys)
                                  if not any(o != k and (k.startswith(o + ".") or k.startswith(o + " ")) for o in keys)]


# ------------------------------------------------------------------------------------------------ C2 --------------
RECALL_EXTRA = """

You are now the SECOND reader of this section. Under "Requirements already written" you get what a first reader
extracted from it. Return ONLY binding product obligations in the section that the first reader missed, or that it
merged into another requirement (so each distinct obligation gets its own requirement). Do not repeat or rephrase
requirements that are already there. Return an empty list if nothing is missing."""


def stage_c2(A, jobs, c_by_job: dict, log) -> list[dict]:
    examples = H.style_examples()

    def one(i):
        job = jobs[i]
        have = "\n".join(f"- [{q['node_id']}] {q['text']}" for q in c_by_job.get(i, [])) or "(none)"
        user = H.job_prompt(A, job, examples) + "\n\n## Requirements already written\n" + have
        res = rp.call_json(reqagent.SYSTEM + H.C_EXTRA + RECALL_EXTRA, [{"type": "text", "text": user}], H.C_SCHEMA,
                           max_tokens=32000, tag=H.job_tag("H7_C2", job))
        return [{**q, "pass": "recall"} for q in H.tie(job, res["requirements"])]

    with ThreadPoolExecutor(8) as pool:
        out = [q for qs in pool.map(one, range(len(jobs))) for q in qs]
    log(f"stage C2: second reader added {len(out)}")
    return out


# ------------------------------------------------------------------------------------------------ D ---------------
FILTER_SYSTEM = """You are the senior reviewer of a list of engineering requirements extracted from ONE UN Regulation for a
specific product. Each candidate carries an id, the paragraph it cites and its text. Decide for every candidate "keep"
or "drop", and merge candidates that split one obligation into pieces.

The expert list you are matched against is thorough: within an applicable section it contains every obligation on the
device, its parts, its markings, its performance, its tests or its installation. The default is KEEP.

Drop a candidate only for one of these reasons:
1. The obligation falls on the applicant, the manufacturer's paperwork, the approval authority, the technical service
   or an inspector (submitting samples or documents, granting approval, notifying), not on the device.
2. It describes how the laboratory tests rather than what the product must withstand or achieve: measurement set-up,
   equipment and receiver settings, antenna or sample positioning, how many samples go to which test, the order of
   tests, sampling plans and statistics. Keep the severities, levels, durations, ranges and pass/fail criteria.
3. It is a definition, an informative table or figure, or a pure announcement that other paragraphs apply.
4. It is about a function, device category, class, configuration or vehicle category the product does not have at all
   (the profile's open options are NOT in this group: they may apply).
5. It duplicates another kept candidate (same obligation, same paragraph). Keep the clearer one.

Do not drop conditional obligations ("where optional lamps are installed", "in case replaceable light source modules
are used", "lamps incorporating a logo"), test severities and conditions applied to the product, pass/fail criteria,
conformity-of-production tolerances, or marking provisions.

Merge only when several candidates split ONE limit of ONE quantity under ONE test condition into its values (e.g. the
rows of one photometric table for one function): keep one, give merged text with all values, and drop the others with
reason "merged into <id>". Different test conditions are different obligations even when the quantity is the same
(e.g. levels for different pulses, frequency bands with their own test, different angles of different devices): do not
merge them. Never merge list items (a)/(b)/(c) that each impose something."""

FILTER_SCHEMA = rp.FILTER_SCHEMA


def stage_d(A, B, cand: list[dict], log) -> list[dict]:
    by_reg: dict[str, list[dict]] = {}
    for q in cand:
        by_reg.setdefault(q["source"]["regulation"], []).append(q)

    def one(reg):
        items = by_reg[reg]
        b = B[reg]
        user = ("## Product\n" + A["product_summary"] + "\n\n## How this regulation applies\n" +
                json.dumps({k: b[k] for k in ("how_it_applies", "part_mapping", "open_options")}, indent=1, ensure_ascii=False) +
                f"\n\n## Candidates from {reg} ({len(items)})\n" +
                "\n".join(f"{q['id']} [§{q['source']['paragraph']}] {q['text']}" for q in items) +
                "\n\nReturn a decision for every id.")
        res = rp.call_json(FILTER_SYSTEM, [{"type": "text", "text": user}], FILTER_SCHEMA, max_tokens=64000,
                           tag=f"H7_D_{reg.replace(' ', '_')}")
        dec = {d["id"]: d for d in res["decisions"]}
        kept, dropped = [], []
        for q in items:
            d = dec.get(q["id"])
            if d is None or d["decision"] == "keep":
                item = dict(q)
                if d and d.get("merged_text"):
                    item["text"] = d["merged_text"].strip()[:2000]
                kept.append(item)
            else:
                dropped.append({**q, "drop_reason": d["reason"]})
        return kept, dropped

    with ThreadPoolExecutor(6) as pool:
        res = list(pool.map(one, sorted(by_reg)))
    kept = [q for k, _ in res for q in k]
    dropped = [q for _, d in res for q in d]
    log(f"stage D: kept {len(kept)} of {len(cand)} ({len(dropped)} dropped or merged)")
    return kept, dropped


# ------------------------------------------------------------------------------------------------ run -------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("case_dir")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--team", default="team-05")
    ap.add_argument("--profile", help="saved profile JSON (from profile.build_profile); default: build it with vision")
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--sure", type=float, default=0.6)
    a = ap.parse_args(argv)
    reqagent.load_env()
    case = Path(a.case_dir).resolve()
    run = reqagent.REPO / "runs" / a.run_name
    run.mkdir(parents=True, exist_ok=True)
    lines = []
    log = lambda s: (print(s, flush=True), lines.append(s))

    if a.profile:
        base = json.loads(Path(a.profile).read_text())
    else:
        import profile as prof
        base = prof.build_profile(case, use_model=True)
        (run / "profile.json").write_text(json.dumps(base, indent=1, ensure_ascii=False))
    profile = base.get("model") or base["deterministic"]
    blocks = product_blocks(case, base)

    A = H.stage_a(blocks, profile, a.threshold, log)
    B = H.stage_b(A, blocks, a.sure, log)
    collapse_nested(B)
    T = traverse.traverse(A["C"], B)
    (run / "traversal.json").write_text(json.dumps(T, indent=1, ensure_ascii=False))
    log("stage T: " + ", ".join(f"{r} {s['items']} items ({s['extract']} extract)" for r, s in T["summary"].items())
        + f"; {len(T['unresolved'])} unresolved, {len(T['skipped'])} skipped")
    jobs = H.section_jobs(A, B, profile)
    log(f"stage C: {len(jobs)} section calls")
    examples = H.style_examples()

    def c_one(job):
        res = rp.call_json(reqagent.SYSTEM + H.C_EXTRA, [{"type": "text", "text": H.job_prompt(A, job, examples)}],
                           H.C_SCHEMA, max_tokens=64000, tag=H.job_tag("H_C", job))
        return H.tie(job, res["requirements"])

    with ThreadPoolExecutor(8) as pool:
        c_res = list(pool.map(c_one, jobs))
    c_qs = [q for qs in c_res for q in qs]
    log(f"stage C: {len(c_qs)} requirements")
    c2_qs = stage_c2(A, jobs, dict(enumerate(c_res)), log)

    cand, dropped_post = H.assemble(c_qs + c2_qs, case.name, a.team)
    log(f"post: {len(cand['requirements'])} candidates after dedupe ({len(dropped_post)} dropped)")
    kept, dropped_d = stage_d(A, B, cand["requirements"], log)
    reqs = []
    for q in kept:
        item = {"id": f"R-{len(reqs)+1:04d}", "text": q["text"], "source": q["source"]}
        if q.get("test_case"):
            item["test_case"] = q["test_case"]
        reqs.append(item)
    sub = {"team_id": a.team, "case_id": case.name, "requirements": reqs}
    problems = rp.validate(sub)
    log(f"final: {len(reqs)} requirements, validation: {'OK' if not problems else problems[:5]}")

    (run / "submission.json").write_text(json.dumps(sub, indent=1, ensure_ascii=False))
    (run / "candidates.json").write_text(json.dumps(cand, indent=1, ensure_ascii=False))
    (run / "dropped.json").write_text(json.dumps({"post": dropped_post, "filter": dropped_d}, indent=1, ensure_ascii=False, default=str))
    (run / "selection.json").write_text(json.dumps({"stageA": {k: A[k] for k in ("product_summary", "regulations", "candidates", "bm25")},
                                                    "stageB": B}, indent=1, ensure_ascii=False))
    subprocess.run([sys.executable, str(H.PKG / "tools/validate.py"), str(run / "submission.json"), "--strict"])
    expected = case / "expected.json"
    if expected.exists():
        (run / "trace.json").write_text(json.dumps(traverse.trace(T, json.loads(expected.read_text())), indent=1, ensure_ascii=False))
        subprocess.run([sys.executable, str(H.PKG / "tools/score_local.py"), str(run / "submission.json"), "--expected", str(expected)])
        summary = H.evaluate_run(run, sub, expected)
        import judge
        J = judge.judge(sub, json.loads(expected.read_text()), run)
        summary += f"\n\npipeline.py judge: F2 {J['summary']['f2']:.3f}, recall {J['summary']['recall']:.3f}"
        (run / "summary.md").write_text(summary + "\n")
        print("\n" + summary)
    (run / "log.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
