"""Traversal -> C1 -> applicability blackbox (artifacts/traversal-blackbox-design.md, first experiment).

    python src/bbpipe.py ../industrial-ai-hackathon/cases/dev-rear-lamp --run-name rearlamp-bb \
        --selection runs/rearlamp-v7b/selection.json --profile data/product/dev-rear-lamp.profile.json

A/B     reused from a saved selection.json (--selection) or run as in hybrid7
T       traverse.traverse: reading list with origin per item                                   [traverse.py]
C1      one call per chunk of T's extract items, current C1 prompt and schema (hybrid/reqagent); context nodes
        are the T-covered targets of the chunk's own references; prompt node ids are qualified ("R148|5.3.1")
Bundles per candidate: cited paragraph, its parent paragraphs, what it and its parent refer to, definitions of terms
        it uses; whole records only, lowest priority left out first (listed in context_issues)
Product spec.md passages + profile parts / drawing observations; facts explicit only when the spec states the value
BB      blackbox.BlackBox per candidate and variant (direct, retrieval_rules), both on pipeline.call_json (Anthropic);
        conditions [] in this first experiment; errors are evaluation_failed, never not_applicable
Policy  submission_applicable.json (applicable only) and submission_keep.json (everything not not_applicable)
C2, stage D and hybrid.assemble's dedupe are not used. expected.json is read only for scoring at the end.
"""
from __future__ import annotations
import argparse, hashlib, json, re, sys, traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blackbox as bb
import hybrid as H
import pipeline as rp
import reqagent
import traverse as tv
from regparse import Corpus, paragraph_key

VARIANTS = {"direct": "direct_gpt", "retrieval_rules": "retrieval_rules"}   # our name -> blackbox variant
CONTEXT_PER_JOB = 80          # supporting nodes per C1 call (as before)
CONTEXT_CHARS = 50_000        # per candidate bundle
REF_NODES = 30                # nodes one reference contributes to a bundle
MAX_DEFS = 6


def load_env():
    for env in (reqagent.REPO / ".env", reqagent.REPO.parent / ".env"):
        if env.exists():
            for line in env.read_text().splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    k, v = line.split("=", 1)
                    import os
                    os.environ.setdefault(k.strip(), v.strip().strip('"'))


def qid(n) -> str:
    return f"{n.reg.replace('UN ', '')}|{tv.label_of(n)}"


def render(n, own_reg: str) -> str:
    pages = f"{n.page_start}" if n.page_start == n.page_end else f"{n.page_start}-{n.page_end}"
    reg = "" if n.reg == own_reg else f"{n.reg} "
    return f"[{qid(n)}] {reg}p.{pages} {tv.label_of(n)}\n{n.text}"


class Run:
    def __init__(self, C: Corpus, T: dict, B: dict, A: dict, profile: dict, run: Path, log):
        self.C, self.T, self.B, self.A, self.profile, self.run, self.log = C, T, B, A, profile, run, log
        self.tr = tv.Traverser(C)                         # reference scanner (same rules as T)
        self.by_q = {}                                    # qualified id -> node
        self.order = {}                                   # (reg, node key) -> position in the document
        for (reg, role), ns in C.by_reg.items():
            if role == "base":
                for i, n in enumerate(ns):
                    self.by_q[qid(n)] = n; self.order[(reg, n.key)] = i
        self.owner = {}                                   # (reg, node key) -> T item id
        for it in T["items"]:
            for nk in it["nodes"]:
                if (it["regulation"], "base", nk) not in C.index:
                    raise ValueError(f"traversal node {it['regulation']} {nk} not in the corpus")
                self.owner[(it["regulation"], nk)] = it["id"]
        self.items = {it["id"]: it for it in T["items"]}

    def node(self, reg, key):
        return self.C.index[(reg, "base", key)]

    def targets(self, n) -> list[tuple]:
        """(reg, doc, label, ref) references of one node, resolved like T does."""
        return list(self.tr.refs(n.reg, n))

    def ref_nodes(self, n, limit=REF_NODES):
        """Nodes each reference of n points to: [(ref text, [nodes], issue or None)]."""
        out = []
        for reg, doc, label, ref in self.targets(n):
            ns = self.tr.subtree(reg, doc, label)
            if not ns:
                out.append((ref, [], f"unresolved reference {reg} {label or doc}")); continue
            issue = None if all((reg, x.key) in self.owner for x in ns[:limit]) else "partly outside the traversal"
            if len(ns) > limit:
                issue = (issue + "; " if issue else "") + f"cut to {limit} of {len(ns)} nodes"
            out.append((ref, ns[:limit], issue))
        return out

    # -------------------------------------------------------------------------------------------- C1 jobs ------
    def jobs(self) -> list[dict]:
        jobs = []
        for reg in sorted({it["regulation"] for it in self.T["items"]}):
            keys = sorted({nk for it in self.T["items"] if it["regulation"] == reg and it["mode"] == "extract" for nk in it["nodes"]},
                          key=lambda k: self.order[(reg, k)])
            if not keys:
                continue
            dicts = [{**asdict(self.node(reg, k)), "id": qid(self.node(reg, k))} for k in keys]
            for name, part in reqagent.chunks(dicts):
                prim = {d["id"] for d in part}
                ctx, ctx_items = [], set()
                for d in part:
                    for _, ns, _ in self.ref_nodes(self.by_q[d["id"]]):
                        for x in ns:
                            q = qid(x)
                            if q not in prim and q not in ctx and (x.reg, x.key) in self.owner and len(ctx) < CONTEXT_PER_JOB:
                                ctx.append(q); ctx_items.add(self.owner[(x.reg, x.key)])
                jobs.append({"id": f"J-{len(jobs) + 1:03d}", "regulation": reg, "name": name, "primary": [d["id"] for d in part],
                             "context": ctx, "items": sorted({self.owner[(reg, self.by_q[q].key)] for q in prim}),
                             "context_items": sorted(ctx_items)})
        return jobs

    def job_prompt(self, job, examples) -> str:
        reg = job["regulation"]
        b = self.B[reg]
        product = {"product_summary": self.A["product_summary"], "profile": self.profile,
                   "regulation_mapping": {k: b[k] for k in ("how_it_applies", "part_mapping", "open_options")}}
        prim = [self.by_q[q] for q in job["primary"]]
        toc = reqagent.toc([{**asdict(n), "id": ""} for n in self.C.by_reg[(reg, "base")]])
        return (f"## Product profile\n{json.dumps(product, indent=1, ensure_ascii=False)}\n\n"
                f"## Style examples\n{examples}\n\n## Regulation\n{reg}, version {prim[0].version}\n\n## Table of contents\n{toc}\n\n"
                f"## Section {job['name']}\n\n" + "\n\n".join(render(n, reg) for n in prim) +
                ("\n\n## Referenced paragraphs (context)\n\n" + "\n\n".join(render(self.by_q[q], reg) for q in job["context"])
                 if job["context"] else ""))

    def c1(self, jobs, workers=8) -> tuple[list[dict], list[dict]]:
        examples = H.style_examples()

        def one(job):
            res = rp.call_json(reqagent.SYSTEM + H.C_EXTRA, [{"type": "text", "text": self.job_prompt(job, examples)}],
                               H.C_SCHEMA, max_tokens=64000, tag=f"BB_C1_{job['regulation'].replace(' ', '_')}_{job['id']}")
            return job, res["requirements"]

        with ThreadPoolExecutor(workers) as pool:
            res = list(pool.map(one, jobs))
        cands, rejected = [], []
        for job, reqs in res:
            for q in reqs:
                n = self.by_q.get(q["node_id"])
                if n is None or q["node_id"] not in job["primary"] + job["context"]:
                    rejected.append({**q, "job": job["id"], "reason": "unknown node id"}); continue
                para = tv.label_of(n) + (f". Table {q['table']}" if q.get("table") else "")
                cands.append({"id": f"C-{len(cands) + 1:04d}", "job": job["id"], "text": q["text"].strip(),
                              "test_case": q.get("test_case"), "table": q.get("table"), "applies_because": q.get("applies_because"),
                              "node_id": q["node_id"], "cited_role": "primary" if q["node_id"] in job["primary"] else "supporting",
                              "item": self.owner.get((n.reg, n.key)),
                              "source": {"regulation": n.reg, "version": n.version, "page": n.page_start, "paragraph": para}})
        return cands, rejected

    # -------------------------------------------------------------------------------------------- bundles ------
    def definitions(self, reg) -> list[tuple[str, object]]:
        out = []
        for n in self.C.by_reg[(reg, "base")]:
            if n.doc == "main" and (n.label == "2" or n.label.startswith("2.")):
                m = re.match(r'^\W*"\s*_?\s*([^"_]{2,80}?)\s*_?\s*"\s*(?:,[^,]{0,40},\s*)?(?:means|is|are|shall mean)\b', n.text)
                if m:
                    out.append((m.group(1).strip().lower(), n))
        return out

    def bundle(self, cand) -> tuple[list[dict], list[str]]:
        n = self.by_q[cand["node_id"]]
        recs, issues, seen = [], [], set()

        def add(x, rel, prio):
            if qid(x) not in seen:
                seen.add(qid(x))
                recs.append((prio, {"id": qid(x), "text": x.text, "regulation": x.reg, "version": x.version,
                                    "paragraph": tv.label_of(x), "page": x.page_start, "relation": rel,
                                    "traversal_item": self.owner.get((x.reg, x.key))}))

        add(n, "cited", 0)
        chain, p = [], n
        while p.parent:                                   # parent paragraphs: lead-ins and inherited scope
            par = next((x for x in self.C.by_reg[(p.reg, "base")] if x.doc == p.doc and x.label == p.parent), None)
            if par is None:
                break
            chain.append(par); p = par
        for i, par in enumerate(chain[:4]):
            add(par, "parent", 1 + i * 0.1)
        for src, prio in [(n, 2)] + [(par, 3) for par in chain[:1]]:
            for ref, ns, issue in self.ref_nodes(src):
                if issue:
                    issues.append(f"{ref!r} from {tv.label_of(src)}: {issue}")
                for x in ns:
                    add(x, f"referenced by {tv.label_of(src)} ({ref[:60]})", prio)
        text = (cand["text"] + " " + n.text).lower()
        defs = [d for t, d in self.definitions(n.reg) if re.search(r"\b" + re.escape(t) + r"\b", text)][:MAX_DEFS]
        for d in defs:
            add(d, "definition", 4)
        recs.sort(key=lambda r: r[0])
        out, size = [], 0
        for prio, r in recs:
            if out and size + len(r["text"]) > CONTEXT_CHARS:
                issues.append(f"left out for size: {r['id']} ({r['relation'][:40]})"); continue
            if not r["text"].strip():
                continue
            out.append(r); size += len(r["text"])
        it = self.items.get(cand["item"] or "")
        if it and it.get("truncated"):
            issues.append(f"traversal item {it['id']} was truncated")
        return out, issues


# ------------------------------------------------------------------------------------------------ stage D2 -------
D2_SYSTEM = """You review candidate engineering requirements extracted from ONE UN Regulation for a specific product,
and predict which of them an expert would put in their requirement list for this product. The candidates are grouped
by the section of the regulation they come from; each section line gives its title, its kind (requirement, test,
conformity, ...) and whether it was selected directly or reached through a cross-reference.

Score every candidate from 0 to 1: the probability that an expert engineer lists it as its own requirement for this
product. Think about the expert's list as a whole:
- They list obligations an engineer designs, builds or tests the product against: construction, performance,
  photometry/colour, durability tests with their severities and pass criteria, markings, installation conditions
  that bind this product's own functions.
- They leave out: whole sections about functions, devices or signals the product does not provide or that the
  inputs give no sign of (score every candidate of such a section low), procedure details of the laboratory,
  conformity-of-production administration, approval-mark examples and arrangements, general statements that only
  repeat or announce other paragraphs, and provisions that bind the vehicle manufacturer or another device.
- A conditional requirement for an open option of the product (one of the listed open_options) is likely listed;
  a conditional for a feature nothing in the inputs suggests is unlikely.
- When two candidates state the same obligation, mark the weaker one duplicate_of the better one.
Scores must spread: use the whole range, do not give everything 0.7."""

D2_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["sections", "candidates"],
             "properties": {
                 "sections": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                     "required": ["section", "expert_lists_it", "reason"],
                     "properties": {"section": {"type": "string"}, "expert_lists_it": {"type": "number"}, "reason": {"type": "string"}}}},
                 "candidates": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                     "required": ["id", "score", "duplicate_of", "reason"],
                     "properties": {"id": {"type": "string"}, "score": {"type": "number"},
                                    "duplicate_of": {"type": ["string", "null"]}, "reason": {"type": "string"}}}}}}


def prefilter(cands, items) -> tuple[list, list]:
    """Code filter: candidates from referenced sections (hop >= 1) or administrative sections are dropped."""
    keep, drop = [], []
    for c in cands:
        it = items.get(c["item"] or "", {})
        if it.get("hop", 0) >= 1:
            drop.append({**c, "drop_reason": "from a referenced section (hop >= 1)"})
        elif it.get("kind") == "administrative":
            drop.append({**c, "drop_reason": "administrative section"})
        else:
            keep.append(c)
    return keep, drop


def stage_d2(A, B, cands, items, workers=6) -> dict:
    """One call per regulation; returns {candidate id: {score, duplicate_of, reason, section_score}}."""
    by_reg = {}
    for c in cands:
        by_reg.setdefault(c["source"]["regulation"], []).append(c)

    def one(reg):
        b, groups = B[reg], {}
        for c in by_reg[reg]:
            groups.setdefault(c["item"] or "?", []).append(c)
        lines = []
        for iid, cs in groups.items():
            it = items.get(iid, {})
            lines.append(f"\n### Section {it.get('key', iid)}: {it.get('title') or ''} [kind: {it.get('kind', '?')}, "
                         f"{'selected' if it.get('hop', 0) == 0 else 'reached by reference'}]")
            lines += [f"{c['id']} [§{c['source']['paragraph']}] {c['text']}  (extractor's note: {c.get('applies_because') or ''})" for c in cs]
        user = ("## Product\n" + A["product_summary"] + "\n\n## How this regulation applies (earlier analysis)\n" +
                json.dumps({k: b[k] for k in ("how_it_applies", "part_mapping", "open_options")}, indent=1, ensure_ascii=False) +
                f"\n\n## Candidates from {reg} ({len(by_reg[reg])}), by section\n" + "\n".join(lines) +
                "\n\nScore every section and every candidate id.")
        res = rp.call_json(D2_SYSTEM, [{"type": "text", "text": user}], D2_SCHEMA, max_tokens=64000, tag=f"BB_D2_{reg.replace(' ', '_')}")
        sec = {s["section"]: s["expert_lists_it"] for s in res["sections"]}
        out = {}
        for d in res["candidates"]:
            out[d["id"]] = {"score": d["score"], "duplicate_of": d["duplicate_of"], "reason": d["reason"]}
        for c in by_reg[reg]:
            k = items.get(c["item"] or "", {}).get("key", "")
            out.setdefault(c["id"], {"score": 0.5, "duplicate_of": None, "reason": "not scored"})
            out[c["id"]]["section_score"] = next((v for s, v in sec.items() if s == k or s.endswith(" " + k) or k in s), None)
        return out

    res = {}
    with ThreadPoolExecutor(workers) as pool:
        for r in pool.map(one, sorted(by_reg)):
            res.update(r)
    return res


def select_d2(cands, scores, keep_share=None, threshold=None):
    """Drop duplicates, then keep by threshold or by the top share per regulation (share carries over to new cases)."""
    nodup = [c for c in cands if not scores[c["id"]]["duplicate_of"]]
    if threshold is not None:
        return [c for c in nodup if scores[c["id"]]["score"] >= threshold]
    out, by_reg = [], {}
    for c in nodup:
        by_reg.setdefault(c["source"]["regulation"], []).append(c)
    for reg, cs in by_reg.items():
        cs = sorted(cs, key=lambda c: -scores[c["id"]]["score"])
        out += cs[:max(1, round(len(cs) * keep_share))]
    keep = {c["id"] for c in out}
    return [c for c in cands if c["id"] in keep]


# ------------------------------------------------------------------------------------------------ product --------
def product_evidence(case: Path, profile: dict) -> dict:
    spec = next(iter(sorted(case.glob("*.md"))), None)
    passages, section = [], ""
    if spec:
        for line in spec.read_text().splitlines():
            s = line.strip()
            if not s or re.fullmatch(r"[|\-\s:]+", s):
                continue
            if s.startswith("#"):
                section = s.lstrip("# ").strip(); continue
            if s.startswith("|") and re.search(r"typical material|component", s, re.I) and passages and passages[-1]["text"].endswith(":"):
                continue
            text = re.sub(r"\s*\|\s*", " | ", s.strip("| ").strip()) if s.startswith("|") else s.lstrip("-* ").strip()
            passages.append({"id": f"S-{len(passages) + 1:02d}", "text": f"{section}: {text}" if section else text,
                             "source": spec.name, "section": section})
    for p in profile.get("parts", []):
        t = "; ".join(f"{k}: {p[k]}" for k in ("function", "regulated_function", "material", "evidence_spec", "evidence_drawing") if p.get(k))
        passages.append({"id": p["part_id"], "text": f"Part {p['part_id']} {p['name']}: {t}", "source": "profile (spec + exploded view)"})
    for i, o in enumerate(profile.get("drawing_observations", []) or []):
        passages.append({"id": f"D-{i + 1:02d}", "text": f"Exploded view: {o if isinstance(o, str) else json.dumps(o, ensure_ascii=False)}",
                         "source": "profile (exploded view)"})
    spec_ids = [p["id"] for p in passages if p["id"].startswith("S-")]

    def fact(field, value, fallback_ids):
        hits = [p["id"] for p in passages if p["id"] in spec_ids and str(value).lower() in p["text"].lower()]
        return {"id": f"F-{len(facts) + 1:02d}", "field": field, "value": value, "unit": None,
                "status": "explicit" if hits else "inferred", "evidence_ids": hits or fallback_ids}

    facts = []
    for field in ("product", "vehicle_category", "mounting"):
        if profile.get(field):
            facts.append(fact(field, profile[field], spec_ids[:1]))
    for p in profile.get("parts", []):
        if p.get("regulated_function"):
            facts.append(fact(f"part.{p['part_id']}.regulated_function", p["regulated_function"], [p["part_id"]]))
    if "has_electronics" in profile:
        facts.append({**fact("has_electronics", profile["has_electronics"], spec_ids[:1]), "status": "inferred"})
    facts = [f for f in facts if f["evidence_ids"]]
    return {"passages": passages, "facts": facts}


# ------------------------------------------------------------------------------------------------ blackbox -------
def fact_citing_llm(product):
    """pipeline_llm, but a cited fact id (F-09) is replaced by that fact's evidence passages: the model sees facts and
    cites them, blackbox accepts passage ids only."""
    ev = {f["id"]: f["evidence_ids"] for f in product["facts"]}

    def fix(ids):
        out = []
        for i in ids:
            for j in ev.get(i, [i]):
                if j not in out:
                    out.append(j)
        return out

    def call(system, payload, schema):
        r = bb.pipeline_llm(system, payload, schema)
        r = {**r, "product_evidence_ids": fix(r["product_evidence_ids"]),
             "condition_results": [{**c, "product_evidence_ids": fix(c["product_evidence_ids"])} for c in r["condition_results"]]}
        return r
    return call


def evaluate(cands, bundles, product, variants, workers=8):
    llm = fact_citing_llm(product)
    engines = {v: bb.BlackBox(VARIANTS[v], llm=llm) for v in variants}
    inputs, out = [], {v: [] for v in variants}
    for c in cands:
        ctx, issues = bundles[c["id"]]
        req = {"id": c["id"], "text": c["text"], "test_case": c["test_case"], "source": c["source"],
               "context_issues": issues,
               "conditions_note": "No conditions extracted in this experiment; judge applicability conditions from the regulatory context."}
        inputs.append({"requirement": req, "context": ctx, "conditions": []})

    def one(args):
        v, inp = args
        try:
            r = engines[v].evaluate(inp["requirement"], inp["context"], product, [])
            return v, {k: r[k] for k in ("requirement_id", "decision", "reason", "regulatory_evidence_ids", "product_evidence_ids",
                                         "missing_information", "trace")}
        except Exception as e:                            # failed evaluation: kept for review, never not_applicable
            return v, {"requirement_id": inp["requirement"]["id"], "decision": "evaluation_failed",
                       "reason": f"{type(e).__name__}: {e}", "trace": {"error": traceback.format_exc(limit=2)}}

    with ThreadPoolExecutor(workers) as pool:
        for v, r in pool.map(one, [(v, i) for v in variants for i in inputs]):
            out[v].append(r)
    return inputs, out


def submission(cands, decisions, keep: set, team, case_id) -> tuple[dict, dict]:
    dec = {d["requirement_id"]: d["decision"] for d in decisions}
    reqs, mapping = [], {}
    for c in cands:
        if dec.get(c["id"]) in keep:
            rid = f"R-{len(reqs) + 1:04d}"
            item = {"id": rid, "text": c["text"], "source": c["source"]}
            if c.get("test_case"):
                item["test_case"] = c["test_case"]
            reqs.append(item); mapping[rid] = c["id"]
    return {"team_id": team, "case_id": case_id, "requirements": reqs}, mapping


# ------------------------------------------------------------------------------------------------ run ------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("case_dir")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--team", default="team-05")
    ap.add_argument("--selection", help="reuse stage A/B from a run's selection.json")
    ap.add_argument("--profile", help="saved profile JSON; default: build it with vision")
    ap.add_argument("--hops", type=int, default=tv.N_HOPS)
    ap.add_argument("--variants", default="retrieval_rules,direct")
    ap.add_argument("--limit", type=int, help="evaluate only the first N candidates (smoke test)")
    ap.add_argument("--stop-after", choices=["t", "c1", "bundles"], help="stop after this step")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--keep-share", type=float, default=0.8, help="D2: share of candidates kept per regulation")
    ap.add_argument("--threshold", type=float, help="D2: keep score >= threshold instead of a share")
    ap.add_argument("--final", choices=["d", "d2", "blackbox"], default="blackbox", help="what follows C1: stage D or the blackbox")
    a = ap.parse_args(argv)
    load_env()
    case = Path(a.case_dir).resolve()
    run = reqagent.REPO / "runs" / a.run_name
    run.mkdir(parents=True, exist_ok=True)
    lines = []
    log = lambda s: (print(s, flush=True), lines.append(s))
    dump = lambda name, obj: (run / name).write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str))

    base = json.loads(Path(a.profile).read_text()) if a.profile else None
    if base is None:
        import profile as prof
        base = prof.build_profile(case, use_model=True); dump("profile.json", base)
    profile = base.get("model") or base["deterministic"]
    C = Corpus(reqagent.REPO / "data/pdf")
    if a.selection:
        sel = json.loads(Path(a.selection).read_text())
        A = {"C": C, "product_summary": sel["stageA"]["product_summary"]}
        B = sel["stageB"]
        log(f"A/B: reused {a.selection}")
    else:
        import hybrid7
        from extend_profile import product_blocks
        blocks = product_blocks(case, base)
        A = H.stage_a(blocks, profile, 0.3, log)
        B = H.stage_b(A, blocks, 0.6, log)
        hybrid7.collapse_nested(B)
        dump("selection.json", {"stageA": {k: A[k] for k in ("product_summary", "regulations", "candidates", "bm25")}, "stageB": B})

    T = tv.traverse(C, B, a.hops)
    dump("traversal.json", T)
    log("T: " + ", ".join(f"{r} {s['items']} items ({s['extract']} extract)" for r, s in T["summary"].items()))
    files = sorted({it["regulation"] for it in T["items"]})
    fp = {r: hashlib.sha256((reqagent.REPO / "data/pdf" / f"{r.replace(' ', '_')}_base.json").read_bytes()).hexdigest()[:16] for r in files}
    manifest = {"case": case.name, "run": a.run_name, "model": rp.MODEL, "selection": a.selection, "traversal": T["params"],
                "corpus_sha256": fp, "variants": a.variants, "conditions": "none (first experiment)",
                "policies": {"submission_applicable.json": ["applicable"],
                             "submission_keep.json": ["applicable", "unresolved", "evaluation_failed"]}}
    dump("run_manifest.json", manifest)
    if a.stop_after == "t":
        return
    R = Run(C, T, B, A, profile, run, log)
    jobs = R.jobs()
    dump("c1_jobs.json", jobs)
    log(f"C1: {len(jobs)} jobs")
    cands, rejected = R.c1(jobs, a.workers)
    dump("c1_candidates.json", {"candidates": cands, "rejected": rejected})
    log(f"C1: {len(cands)} candidates ({len(rejected)} rejected: unknown node id)")
    if a.final == "d2":                                  # C1 -> code filter -> D2 scores -> keep share -> submission
        kept0, dropped0 = prefilter(cands, R.items)
        log(f"prefilter: kept {len(kept0)} of {len(cands)}")
        scores = stage_d2(A, B, kept0, R.items)
        dump("d2_scores.json", {"prefilter_dropped": dropped0, "scores": scores})
        kept = select_d2(kept0, scores, keep_share=a.keep_share) if a.threshold is None else select_d2(kept0, scores, threshold=a.threshold)
        sub, mapping = submission(kept, [{"requirement_id": c["id"], "decision": "applicable"} for c in kept], {"applicable"}, a.team, case.name)
        dump("submission_d2.json", sub); dump("submission_mapping_d2.json", mapping)
        problems = rp.validate(sub)
        log(f"D2: {len(sub['requirements'])} requirements (keep share {a.keep_share}, threshold {a.threshold}), "
            f"validation: {'OK' if not problems else problems[:3]}")
        (run / "log.txt").write_text("\n".join(lines) + "\n")
        return
    if a.final == "d":                                   # C1 -> stage D (hybrid7 reviewer) -> submission, no blackbox
        import hybrid7
        seen, uniq = set(), []
        for c in cands:                                   # exact duplicates only (same paragraph, same text)
            k = (c["source"]["regulation"], c["source"]["paragraph"], rp._norm(c["text"]))
            if k not in seen:
                seen.add(k); uniq.append(c)
        log(f"exact duplicates removed: {len(cands) - len(uniq)}")
        kept, dropped = hybrid7.stage_d(A, B, uniq, log)
        dump("dropped_d.json", dropped)
        sub, mapping = submission(kept, [{"requirement_id": c["id"], "decision": "applicable"} for c in kept], {"applicable"}, a.team, case.name)
        for r, c in zip(sub["requirements"], kept):
            r["text"] = c["text"]                         # stage D may have merged text
        dump("submission.json", sub); dump("submission_mapping.json", mapping)
        problems = rp.validate(sub)
        log(f"final: {len(sub['requirements'])} requirements, validation: {'OK' if not problems else problems[:3]}")
        expected = case / "expected.json"
        if expected.exists():
            import judge
            (run / "judge_final").mkdir(exist_ok=True)
            J = judge.judge(sub, json.loads(expected.read_text()), run / "judge_final")["summary"]
            log(f"final: submitted {J['submitted']}, matched {J['matched']}, P {J['precision']:.3f}, R {J['recall']:.3f}, F2 {J['f2']:.3f}")
        (run / "log.txt").write_text("\n".join(lines) + "\n")
        return
    if a.stop_after == "c1":
        expected = case / "expected.json"
        if expected.exists():                             # score C1 alone (evaluation only)
            import judge
            sub, _ = submission(cands, [{"requirement_id": c["id"], "decision": "applicable"} for c in cands], {"applicable"}, a.team, case.name)
            dump("submission_c1_only.json", sub)
            (run / "judge_c1_only").mkdir(exist_ok=True)
            J = judge.judge(sub, json.loads(expected.read_text()), run / "judge_c1_only")["summary"]
            log(f"C1 only: submitted {J['submitted']}, matched {J['matched']}, P {J['precision']:.3f}, R {J['recall']:.3f}, F2 {J['f2']:.3f}")
        (run / "log.txt").write_text("\n".join(lines) + "\n")
        return
    bundles = {c["id"]: R.bundle(c) for c in cands}
    dump("context_bundles.json", {k: {"context": v[0], "issues": v[1]} for k, v in bundles.items()})
    product = product_evidence(case, profile)
    dump("product_evidence.json", product)
    log(f"bundles: {sum(len(v[0]) for v in bundles.values())} records, {sum(bool(v[1]) for v in bundles.values())} candidates with issues; "
        f"product: {len(product['passages'])} passages, {len(product['facts'])} facts")
    if a.stop_after == "bundles":
        return
    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    ev = cands[:a.limit] if a.limit else cands
    inputs, decisions = evaluate(ev, bundles, product, variants, a.workers)
    (run / "blackbox_inputs.jsonl").write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in inputs) + "\n")
    for v in variants:                                    # decisions on disk before any scoring call can fail
        (run / f"applicability_{v}.jsonl").write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in decisions[v]) + "\n")
    expected = case / "expected.json"
    exp = json.loads(expected.read_text()) if expected.exists() else None
    report = []
    if exp:
        import judge
        base_sub, _ = submission(ev, [{"requirement_id": c["id"], "decision": "applicable"} for c in ev], {"applicable"}, a.team, case.name)
        dump("submission_c1_only.json", base_sub)
        (run / "judge_c1_only").mkdir(exist_ok=True)
        J = judge.judge(base_sub, exp, run / "judge_c1_only")["summary"]
        report.append(f"| C1 only (no blackbox) | {len(base_sub['requirements'])} | {J['matched']} | {J['precision']:.3f} | {J['recall']:.3f} | {J['f2']:.3f} |")
    for v in variants:
        (run / f"applicability_{v}.jsonl").write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in decisions[v]) + "\n")
        counts = {k: sum(d["decision"] == k for d in decisions[v]) for k in ("applicable", "not_applicable", "unresolved", "evaluation_failed")}
        log(f"BB {v}: {counts}")
        dump(f"unresolved_{v}.json", [d for d in decisions[v] if d["decision"] in ("unresolved", "evaluation_failed")])
        for name, keep in manifest["policies"].items():
            sub, mapping = submission(ev, decisions[v], set(keep), a.team, case.name)
            stem = name.replace(".json", f"_{v}")
            dump(stem + ".json", sub); dump(stem.replace("submission", "submission_mapping") + ".json", mapping)
            problems = rp.validate(sub)
            if problems:
                log(f"validation {stem}: {problems[:3]}")
            if exp:
                import judge
                (run / f"judge_{stem}").mkdir(exist_ok=True)
                J = judge.judge(sub, exp, run / f"judge_{stem}")["summary"]
                report.append(f"| {v} / {name.split('_')[1].split('.')[0]} | {len(sub['requirements'])} | {J['matched']} | "
                              f"{J['precision']:.3f} | {J['recall']:.3f} | {J['f2']:.3f} |")
    if report:
        md = "| run | submitted | matched | precision | recall | F2 |\n|---|---|---|---|---|---|\n" + "\n".join(report)
        (run / "summary.md").write_text(f"# {a.run_name}\n\n{md}\n")
        log("\n" + md)
    (run / "log.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
