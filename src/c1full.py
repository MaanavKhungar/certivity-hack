"""Full-PDF C1 extraction, then traversal for context (artifacts/new-c1-traverse-integrate.md).

    python src/c1full.py "UN R148" --role base --run-name r148-c1full --plan-only     # batches + sizes, no calls
    python src/c1full.py "UN R148" --role base --run-name r148-c1full

Batch   every parsed node of ONE regulation/version in document order (main body per top-level section, annexes per
        annex/part), ~CHUNK_CHARS of primary text per batch; big sections are split at deeper labels, the ancestors of a
        split part are repeated as context-only scope; a single node above the limit is sent whole and marked oversized
C1      one structured call per batch, many requirements per response; source-local prompt (no product, no
        applicability filter); requirements cite the primary node(s) they are stated in
Check   citations of primary nodes are valid; citations of scope-only or unknown nodes are kept as flagged candidates;
        nothing is deduplicated, merged or truncated (exact repeats are only marked possible_duplicate_of)
T       code only, after extraction: per requirement its source nodes, their parent chain, references from the source
        nodes (and their direct parent) followed for --hops rounds (default 2), definitions of terms used. Traversal
        output is CONTEXT ONLY: it never creates requirements. Another regulation's text is attached, not scanned further.
Out     extraction_manifest.json, c1_candidates.json, requirements_with_context.json, context_nodes.json,
        context_diagnostics.json. Not a product-filtered submission; expected.json is not read.
"""
from __future__ import annotations
import argparse, hashlib, json, re, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline as rp
import reqagent
import traverse as tv
from regparse import Corpus

CHUNK_CHARS = 40_000          # primary text per batch (heuristic, not a token guarantee)
N_HOPS = 2                    # reference rounds from a requirement's source node(s)
MAX_TARGET_NODES = 300        # nodes one reference target may contribute (rest reported, not silently dropped)
MAX_CONTEXT_NODES = 1_000     # context nodes per requirement
MAX_DEFS = 8
RETRIES = 2                   # extra attempts per batch after a failed call

SYSTEM = """You are a vehicle type-approval engineer. You read part of a UN Regulation (UNECE, 1958 Agreement) and
list every engineering requirement it states, so that an engineer can later decide which of them bind a given product
and design and test against them. There is no product here: do not leave out requirements because of a product type,
class, colour or variant. Keep each requirement's own conditions in its text instead.

You receive the regulation's table of contents, the PRIMARY paragraphs to extract from, and possibly SCOPE paragraphs:
headings and lead-ins of the section the primary paragraphs sit in. Scope paragraphs are context only: use them to
understand conditions inherited from a heading, never extract from them or cite them. Each paragraph is shown as
[node_id] p.<pages> <label> followed by its text. Tables appear as markdown grids inside their paragraph.

What counts as a requirement:
- An obligation on the regulated device, component or vehicle: its design, construction, materials, dimensions,
  photometric/colorimetric or other performance, durability, markings, installation, and the results it must achieve
  in the prescribed tests.
- For prescribed tests, keep what the item must withstand and achieve: the exposure or stress applied (atmosphere or
  solution, temperature, duration, levels), the conditions that define what is measured, and the pass criteria.
- Conditional requirements and design options ("if the device has / is marked / is combined with X, it shall ...",
  "in the case of category X ..."): keep them, each with its condition.
- Approval marking on the product: content, placement, size, legibility, format.
- Permissions, exemptions and relaxations ("X may be ...", "notwithstanding paragraph N", "does not apply if ..."),
  written with their condition.
- An obligation stated by reference ("shall meet the requirements of Annex 6", "shall comply with paragraph 6.2."):
  write that obligation as stated, keeping the reference. Do not import or invent the referenced content.

What does not count:
- Obligations on the approval authority, Contracting Parties or technical service; application paperwork,
  communication forms, numbering of approvals, transitional provisions, definitions, and the scope statement.
- How the laboratory runs a test: apparatus, instruments, measuring distances, stray light, sample allocation and the
  technician's handling steps. Do not turn a test step into an obligation; a test paragraph yields a requirement only
  where it states a condition of the kind listed above or a pass criterion.
- Conformity-of-production statistics (sampling plans, sample sizes, deviation thresholds per sample group); keep only
  a limit on how far a production part's performance may deviate.
- Figures showing example layouts of approval marks; recommendations ("should"); descriptions without an obligation.

How to write them:
- One obligation per requirement. Each "shall"/"must" clause is its own requirement, also when two clauses share a
  sentence. A paragraph can hold several requirements or none; do not merge distinct requirements because they sit in
  the same paragraph.
- SPLIT when values or conditions differ: each test level, severity, band or pulse; each supply voltage or category;
  each direction; each different property; each separate signal or indicator. A limit constant over one range and
  varying over another is two requirements.
- KEEP TOGETHER: several attributes of one property in one clause; the checks of one test at several stages; an
  obligation with its own exception stated in the same paragraph; both sides of one "and/or".
- When a paragraph sets a limit by reference to a table: the paragraph's own obligation is one requirement (table
  null), then the table's values (table set): a single limit over a measuring grid is ONE requirement listing the
  values with their header relationships; rows that are DIFFERENT tests or conditions are ONE requirement PER ROW.
- If the requirement comes from a table, set "table" to the table name (e.g. "3" or "A6-1"), else null.
- Stay close to the regulation's wording and numbers; make each requirement understandable on its own (name the
  subject instead of "it", keep "when tested according to ...").
- source_node_ids: the PRIMARY node id(s) of the paragraph(s) the obligation is stated in, deepest first. Never cite
  a scope paragraph, never invent ids.
- If the primary paragraphs state no requirement, return an empty list."""

GRANULARITY_V1 = """- SPLIT when values or conditions differ: each test level, severity, band or pulse; each supply voltage or category;
  each direction; each different property; each separate signal or indicator. A limit constant over one range and
  varying over another is two requirements.
- KEEP TOGETHER: several attributes of one property in one clause; the checks of one test at several stages; an
  obligation with its own exception stated in the same paragraph; both sides of one "and/or".
"""

# v2: granularity rules from the expert lists' consistent patterns (generic, no paragraph numbers or wording)
GRANULARITY_V2 = """- SPLIT into separate requirements:
  - each different test, test pulse, test profile, severity class or table row that has its own levels, duration or
    pass criterion;
  - each item of a list of things the product must bear, carry, provide or show (e.g. the marks a component must
    bear: name or mark, identification code, rated voltage, each its own requirement); when the same items are
    required on two different objects (e.g. on a component and on the lamp), write them separately for each object;
  - two different constraints under one "shall" joined by "and" (e.g. "can only be fitted in the correct position
    and can only be removed with tools" is two requirements), each repeating the subject and condition;
  - an additional conditional obligation stated inside another one ("..., and where several X are used, the code
    shall be followed by ..."): its own requirement, with its condition;
  - each direction, each separate signal or indicator, each different property of the same object.
- KEEP TOGETHER in one requirement:
  - one obligation whose value is given per voltage system, per frequency band or per category in the same clause
    (e.g. "a supply voltage of 6.75 V (6 V systems), 13.5 V (12 V systems) or 28 V (24 V systems)", "limit A in
    band 1 and limit B in band 2", "level X (12 V system) or level Y (24 V system)"): one requirement with all the
    values, not one per voltage, band or category;
  - an obligation and the exception, allowance or relaxation that modifies it ("except that ...", "however, ... shall
    be permitted ...", "but not less than ...", "this does not apply to ..."), also when the exception is the next
    sentence of the same paragraph: one requirement stating both. A permission that does not modify a limit stated
    in the same paragraph is its own requirement;
  - several attributes of one property in one clause ("clearly legible and indelible"); the checks of one test at
    several stages; both sides of one "and/or".
"""
# v3: v2 with the over-reach it showed on R148/R10 taken back (lists of tests, one constraint on several objects,
# constant vs varying limits, separate test profiles per voltage, separate tables)
GRANULARITY_V3 = """- SPLIT into separate requirements:
  - each different test, test pulse, test profile, severity class or table row that has its own levels, duration or
    pass criterion; a separate test or test profile for each voltage system is a separate requirement too;
  - each table the paragraph refers to for a DIFFERENT criterion (e.g. one table for the severity over most of a band,
    another for a minimum level over the whole band): one requirement per table, besides the method;
  - each mark or inscription in a list of marks a product or component must bear (name or mark, identification code,
    rated voltage, ...); when the same marks are required on two different objects (e.g. on a component and on the
    lamp), write them separately for each object. This applies to marks; a list of tests or of aspects the product
    must comply with in one sentence ("shall comply with the radiated and conducted emission and immunity
    requirements") stays one requirement;
  - two different design features or actions under one "shall" joined by "and" (e.g. "can only be fitted in the
    correct position and can only be removed with tools" is two requirements), each repeating subject and condition.
    One constraint applied to several objects ("not interchangeable with A or with B") and one general performance
    statement ("shall operate satisfactorily and continue to show the prescribed characteristics") stay one
    requirement;
  - an additional conditional obligation stated inside another one ("..., and where several X are used, the code
    shall be followed by ..."): its own requirement, with its condition;
  - each direction, each separate signal or indicator, each different property of the same object.
- KEEP TOGETHER in one requirement:
  - one obligation whose value is given per voltage system or per category in the same clause (e.g. "a supply
    voltage of 6.75 V (6 V systems), 13.5 V (12 V systems) or 28 V (24 V systems)", "level X (12 V system) or level Y
    (24 V system)"): one requirement with all the values, not one per voltage or category;
  - a limit given over frequency or angle ranges: the ranges where the limit VARIES (e.g. decreasing or increasing
    with frequency) form one requirement together; a range where the limit is CONSTANT is its own requirement;
  - an obligation and the exception, allowance or relaxation that modifies it ("except that ...", "however, ... shall
    be permitted ...", "but not less than ...", "this does not apply to ..."), also when the exception is the next
    sentence of the same paragraph: one requirement stating both. A permission that does not modify a limit stated
    in the same paragraph is its own requirement;
  - several attributes of one property in one clause ("clearly legible and indelible"); the checks of one test at
    several stages; both sides of one "and/or".
"""
SYSTEMS = {"v1": SYSTEM, "v2": SYSTEM.replace(GRANULARITY_V1, GRANULARITY_V2),
           "v3": SYSTEM.replace(GRANULARITY_V1, GRANULARITY_V3)}
assert SYSTEMS["v2"] != SYSTEM != SYSTEMS["v3"]

TEST_CASE_NOTE = """
- test_case: one short sentence on how a test engineer would verify the requirement, or null."""


def schema(test_case: bool) -> dict:
    props = {"source_node_ids": {"type": "array", "items": {"type": "string"}},
             "table": {"type": ["string", "null"]}, "text": {"type": "string"}}
    if test_case:
        props["test_case"] = {"type": ["string", "null"]}
    return {"type": "object", "additionalProperties": False, "required": ["requirements"],
            "properties": {"requirements": {"type": "array", "items": {
                "type": "object", "additionalProperties": False, "required": list(props), "properties": props}}}}


def short(reg: str) -> str:
    return reg.replace("UN ", "").replace(" ", "")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


# ------------------------------------------------------------------------------------------------ corpus view ------
class Source:
    """Nodes of every regulation with qualified ids "R148|base|Annex 5 1.2" (a suffix #2.. where a document repeats a
    label; regparse.Corpus.index would keep only the first of those)."""

    def __init__(self, C: Corpus, pdf_dir: Path):
        self.C, self.pdf_dir = C, pdf_dir
        self.qid, self.by_q, self.parent_of, self.fp = {}, {}, {}, {}
        for (reg, role), ns in C.by_reg.items():
            seen, pos = {}, {}
            for n in ns:
                base = f"{short(reg)}|{role}|{tv.label_of(n)}"
                seen[base] = seen.get(base, 0) + 1
                q = base if seen[base] == 1 else f"{base}#{seen[base]}"
                self.qid[id(n)] = q; self.by_q[q] = n
                if n.parent:                                  # nearest earlier node of the same doc with the parent label
                    self.parent_of[id(n)] = pos.get((n.doc, n.parent))
                pos[(n.doc, n.label)] = n
        self.duplicate_labels = sum("#" in q for q in self.by_q)

    def q(self, n) -> str:
        return self.qid[id(n)]

    def fingerprint(self, reg: str, role: str) -> str | None:
        if (reg, role) not in self.fp:
            p = self.pdf_dir / f"{reg.replace(' ', '_')}_{role}.json"
            self.fp[(reg, role)] = sha(p) if p.exists() else None
        return self.fp[(reg, role)]

    def ancestors(self, n) -> list:
        out, p = [], self.parent_of.get(id(n))
        while p is not None:
            out.append(p); p = self.parent_of.get(id(p))
        return out


class RoleTraverser(tv.Traverser):
    """traverse.Traverser's reference scanner and resolvers, reading the chosen role of the own regulation (other
    regulations: base, the only version every PDF has)."""

    def __init__(self, C, own: str, role: str):
        super().__init__(C)
        self.own, self.role = own, role

    def nodes(self, reg):
        return self.C.by_reg.get((reg, self.role if reg == self.own else "base"), [])


# ------------------------------------------------------------------------------------------------ batches ----------
def size(ns) -> int:
    return sum(len(n.text) for n in ns)


def runs(ns, key) -> list[tuple[object, list]]:
    """Consecutive nodes with the same key (document order is kept even where a label prefix comes back later)."""
    out = []
    for n in ns:
        k = key(n)
        if out and out[-1][0] == k:
            out[-1][1].append(n)
        else:
            out.append((k, [n]))
    return out


def split(ns, depth: int, limit: int) -> list[tuple[list, bool]]:
    """Units of nodes in document order, each <= limit where the label structure allows; (nodes, oversized)."""
    if size(ns) <= limit:
        return [(ns, False)]
    if len(ns) == 1:
        return [(ns, True)]                               # one node above the limit: sent whole, never sliced
    groups = runs(ns, lambda n: ".".join(n.label.split(".")[:depth]))
    if len(groups) == 1:                                  # all nodes share this prefix: go one level deeper
        if depth > max(len(n.label.split(".")) for n in ns):
            return [([n], len(n.text) > limit) for n in ns]
        return split(ns, depth + 1, limit)
    out = []
    for _, g in groups:
        out += split(g, depth + 1, limit)
    return out


def batches(S: Source, reg: str, role: str, limit: int) -> list[dict]:
    ns = S.C.by_reg[(reg, role)]
    out = []
    for (doc, top), sec in runs(ns, lambda n: (n.doc, n.label.split(".")[0] if n.doc == "main" else "")):
        name = f"§{top}" if doc == "main" else doc
        units = split(sec, 2 if doc == "main" else 1, limit)
        packed, cur = [], []
        for u, big in units:                              # pack small units back together up to the limit
            if big:
                if cur: packed.append((cur, False)); cur = []
                packed.append((u, True)); continue
            if cur and size(cur) + size(u) > limit:
                packed.append((cur, False)); cur = []
            cur = cur + u
        if cur:
            packed.append((cur, False))
        for part, big in packed:
            prim = {S.q(n) for n in part}
            scope = []
            for n in part:                                # headings/lead-ins of a split part: context only
                for a in reversed(S.ancestors(n)):
                    if S.q(a) not in prim and S.q(a) not in scope:
                        scope.append(S.q(a))
            label = name if len(packed) == 1 else f"{name} ({tv.label_of(part[0])} – {tv.label_of(part[-1])})"
            b = {"name": label, "primary": [S.q(n) for n in part], "scope": scope, "primary_chars": size(part),
                 "oversized": big, "main": doc == "main", "whole": len(packed) == 1}
            prev = out[-1] if out else None               # neighbouring small whole sections share one call
            if prev and b["whole"] and prev["whole"] and not big and not prev["oversized"] and prev["main"] == b["main"] \
                    and prev["primary_chars"] + b["primary_chars"] <= limit:
                prev["name"] += ", " + label; prev["primary"] += b["primary"]; prev["primary_chars"] += b["primary_chars"]
            else:
                out.append(b)
    for i, b in enumerate(out):
        b.update({"id": f"B-{i + 1:03d}", "status": "planned"})
        del b["main"], b["whole"]
    return out


# ------------------------------------------------------------------------------------------------ C1 ---------------
def render(S: Source, n) -> str:
    pages = f"{n.page_start}" if n.page_start == n.page_end else f"{n.page_start}-{n.page_end}"
    return f"[{S.q(n)}] p.{pages} {tv.label_of(n)}\n{n.text}"


def prompt(S: Source, reg: str, role: str, b: dict, contents: str) -> str:
    prim = [S.by_q[q] for q in b["primary"]]
    txt = (f"## Regulation\n{reg}, version {prim[0].version} ({role})\n\n## Table of contents\n{contents}\n\n"
           f"## Batch {b['name']}\n")
    if b["scope"]:
        txt += "\n## SCOPE paragraphs (context only: do not extract, do not cite)\n\n" + \
               "\n\n".join(render(S, S.by_q[q]) for q in b["scope"]) + "\n"
    return txt + "\n## PRIMARY paragraphs (extract from these)\n\n" + "\n\n".join(render(S, n) for n in prim)


def cached(system, user, sch, tag) -> bool:
    """Whether pipeline.call_json will answer from its cache (same key as call_json computes)."""
    payload = json.dumps({"m": rp.MODEL, "s": system, "u": user, "sc": sch, "e": "high"}, sort_keys=True, ensure_ascii=False)
    return (rp.CACHE / f"{tag}_{hashlib.sha256(payload.encode()).hexdigest()[:24]}.json").exists()


def c1(S, reg, role, plan, test_case, workers, log, prompt_version="v3") -> dict:
    system = SYSTEMS[prompt_version] + (TEST_CASE_NOTE if test_case else "")
    sch = schema(test_case)
    contents = reqagent.toc([{"depth": n.depth, "title": n.title, "text": n.text, "page_start": n.page_start,
                              "doc": n.doc, "label": n.label} for n in S.C.by_reg[(reg, role)]])

    def one(b):
        user = [{"type": "text", "text": prompt(S, reg, role, b, contents)}]
        tag = f"C1F_{short(reg)}_{role}_{b['id']}"
        b["cache_hit"] = cached(system, user, sch, tag)
        b["attempts"], b["errors"] = 0, []
        for attempt in range(RETRIES + 1):
            b["attempts"] += 1
            try:
                t = time.time()
                res = rp.call_json(system, user, sch, max_tokens=64000, tag=tag)
                b["status"] = "ok"; b["seconds"] = round(time.time() - t, 1)
                b["n_requirements"] = len(res["requirements"])
                if not res["requirements"]:
                    b["status"] = "empty_output"              # not proof that the batch states no requirement
                log(f"  {b['id']} {b['name'][:50]:<50} {len(b['primary']):>4} nodes -> {len(res['requirements']):>3} reqs"
                    f"{' (cache)' if b['cache_hit'] else ''}")
                return b, res
            except Exception as e:
                b["errors"].append(f"{type(e).__name__}: {e}")
                if attempt == RETRIES or "max_tokens" in str(e):   # a truncated answer would only repeat
                    break
                time.sleep(5 * (attempt + 1))
        b["status"] = "failed"
        log(f"  {b['id']} {b['name'][:50]:<50} FAILED: {b['errors'][-1][:120]}")
        return b, None

    with ThreadPoolExecutor(workers) as pool:
        return dict((b["id"], (b, res)) for b, res in pool.map(one, plan))


def validate(S, reg, role, plan, results, test_case) -> tuple[list, list]:
    """-> (valid candidates, flagged candidates); everything the model returned is kept in one of the two."""
    valid, flagged, seen = [], [], {}
    for b in plan:
        res = results.get(b["id"], (b, None))[1]
        if res is None:
            continue
        prim, scope = set(b["primary"]), set(b["scope"])
        for i, r in enumerate(res["requirements"]):
            ids = list(dict.fromkeys(r["source_node_ids"]))
            ok = [q for q in ids if q in prim]
            other = [q for q in ids if q not in prim]
            cand = {"batch": b["id"], "index_in_response": i, "text": r["text"], "table": r.get("table"),
                    "cited_ids": ids, "raw": r}
            if test_case:
                cand["test_case"] = r.get("test_case")
            if not ok:
                reason = "cites only scope (context-only) nodes" if other and all(q in scope for q in other) else \
                         "cites no primary node of its batch" if not other or any(q in S.by_q for q in other) else "unknown node id"
                flagged.append({**cand, "id": f"F-{len(flagged) + 1:04d}", "validation": "flagged", "reason": reason})
                continue
            n = S.by_q[ok[0]]
            para = tv.label_of(n) + (f". Table {r['table']}" if r.get("table") else "")
            rid = f"{short(reg)}-{role}-{len(valid) + 1:04d}"
            norm = (tuple(ok), rp._norm(r["text"]))
            cand.update({"id": rid, "source_node_ids": ok, "validation": "valid",
                         "non_primary_ids_ignored": other,
                         "source": {"regulation": reg, "version": n.version, "role": role, "page": n.page_start,
                                    "page_end": max(S.by_q[q].page_end for q in ok), "paragraph": para},
                         "possible_duplicate_of": seen.get(norm)})
            seen.setdefault(norm, rid)
            valid.append(cand)
    return valid, flagged


# ------------------------------------------------------------------------------------------------ traversal --------
class Context:
    """Code-only context per requirement. Reference resolution is cached per source node and per target, so many
    requirements citing the same paragraph or annex cost one expansion."""

    def __init__(self, S: Source, reg: str, role: str, hops: int):
        self.S, self.reg, self.role, self.hops = S, reg, role, hops
        self.tr = RoleTraverser(S.C, reg, role)
        self._refs, self._targets = {}, {}
        self.nodes: dict[str, object] = {}                    # every context node used by any requirement
        self.defs = self._definitions()

    def _definitions(self) -> list[tuple[str, object]]:
        out = []
        for n in self.S.C.by_reg[(self.reg, self.role)]:
            if n.doc == "main" and (n.label == "2" or n.label.startswith("2.")):
                m = re.match(r'^\W*"\s*_?\s*([^"_]{2,80}?)\s*_?\s*"\s*(?:,[^,]{0,40},\s*)?(?:means|is|are|shall mean)\b', n.text)
                if m:
                    out.append((m.group(1).strip().lower(), n))
        return out

    def target(self, reg, doc, label) -> tuple[list, dict | None]:
        k = (reg, doc, label)
        if k not in self._targets:
            ns = self.tr.subtree(reg, doc, label)
            issue = None
            if not ns:
                issue = {"type": "unresolved", "target": f"{reg} {label or doc}",
                         "in_corpus": reg == self.reg or reg in self.tr.regs}
            elif len(ns) > MAX_TARGET_NODES:
                issue = {"type": "target_truncated", "target": f"{reg} {label or doc}", "kept": MAX_TARGET_NODES, "of": len(ns)}
                ns = ns[:MAX_TARGET_NODES]
            self._targets[k] = (ns, issue)
        return self._targets[k]

    def refs(self, n) -> list[tuple]:
        """[(ref text, target key, [nodes], issue, cross)] for one node; cross-PDF targets are never scanned further."""
        q = self.S.q(n)
        if q not in self._refs:
            out = []
            for reg, doc, label, ref in self.tr.refs(n.reg, n):
                ns, issue = self.target(reg, doc, label)
                cross = reg != self.reg
                if cross and re.search(r"series\s+of\s+amendments", ref, re.I):
                    issue = issue or {"type": "version_mismatch", "target": f"{reg} {label or doc}",
                                      "note": f"reference names a series of amendments; resolved to the corpus base "
                                              f"version {ns[0].version if ns else '?'}"}
                out.append((ref, f"{reg} {label or doc}", ns, issue, cross))
            self._refs[q] = out
        return self._refs[q]

    def for_requirement(self, req) -> dict:
        S = self.S
        ids, edges, diags = [], [], []

        def add(n):
            q = S.q(n)
            if q in ids:
                return True
            if len(ids) >= MAX_CONTEXT_NODES:
                return False
            ids.append(q); self.nodes[q] = n
            return True

        roots = [S.by_q[q] for q in req["source_node_ids"]]
        for n in roots:
            add(n)
        for n in roots:                                       # parent chain: lead-ins and inherited scope
            child = n
            for p in S.ancestors(n):
                add(p); edges.append({"from": S.q(child), "to": [S.q(p)], "relation": "parent", "hop": 0})
                child = p
        frontier = list(roots) + [a for n in roots for a in S.ancestors(n)[:1]]
        scanned, budget_hit = set(), False
        for hop in range(1, self.hops + 1):
            nxt = []
            for n in frontier:
                if S.q(n) in scanned:
                    continue                                  # cycles and repeats
                scanned.add(S.q(n))
                for ref, key, ns, issue, cross in self.refs(n):
                    if issue:
                        diags.append({**issue, "from": S.q(n), "ref": ref, "hop": hop})
                    if not ns:
                        continue
                    edges.append({"from": S.q(n), "to": [S.q(x) for x in ns], "target": key, "relation": "reference",
                                  "ref": ref, "hop": hop, "cross_regulation": cross})
                    for x in ns:
                        new = S.q(x) not in ids
                        if not add(x):
                            budget_hit = True; break
                        if new and not cross:
                            nxt.append(x)
            frontier = nxt
        if frontier and self.hops:
            diags.append({"type": "hop_limit", "note": f"{len({S.q(x) for x in frontier})} nodes reached at hop {self.hops} "
                                                       f"were not scanned for further references"})
        if budget_hit:
            diags.append({"type": "context_budget", "note": f"stopped at {MAX_CONTEXT_NODES} context nodes"})
        text = (req["text"] + " " + " ".join(n.text for n in roots)).lower()
        for term, d in [(t, d) for t, d in self.defs if re.search(r"\b" + re.escape(t) + r"\b", text)][:MAX_DEFS]:
            if add(d):
                edges.append({"from": req["source_node_ids"][0], "to": [S.q(d)], "relation": "definition", "term": term, "hop": 0})
        return {"context_node_ids": ids, "edges": edges, "diagnostics": diags,
                "context_status": "incomplete" if any(x["type"] in ("context_budget", "target_truncated", "unresolved")
                                                      for x in diags) else "resolved_within_policy"}


def context_record(S: Source, q: str, n) -> dict:
    return {"id": q, "regulation": n.reg, "version": n.version, "role": n.role, "doc": n.doc, "label": n.label,
            "node_key": n.key, "paragraph": tv.label_of(n), "page_start": n.page_start, "page_end": n.page_end,
            "title": n.title, "heading": n.heading, "tables": n.tables, "text": n.text,
            "corpus_sha256": S.fingerprint(n.reg, n.role)}


# ------------------------------------------------------------------------------------------------ run --------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("regulation", help='e.g. "UN R148"')
    ap.add_argument("--role", default="base", help="base | latest (which parsed version of the PDF)")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--hops", type=int, default=N_HOPS)
    ap.add_argument("--chunk-chars", type=int, default=CHUNK_CHARS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--prompt", choices=sorted(SYSTEMS), default="v3", help="C1 system prompt version")
    ap.add_argument("--test-case", action="store_true", help="also ask for a test_case per requirement (same call)")
    ap.add_argument("--max-batches", type=int, help="smoke test: run only the first N batches")
    ap.add_argument("--plan-only", action="store_true", help="write the batch plan and sizes, make no calls")
    a = ap.parse_args(argv)
    reqagent.load_env()
    pdf_dir = reqagent.REPO / "data/pdf"
    run = reqagent.REPO / "runs" / a.run_name
    run.mkdir(parents=True, exist_ok=True)
    lines = []
    log = lambda s: (print(s, flush=True), lines.append(s))
    dump = lambda name, obj: (run / name).write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str))

    S = Source(Corpus(pdf_dir), pdf_dir)
    reg, role = a.regulation, a.role
    if (reg, role) not in S.C.by_reg:
        sys.exit(f"{reg} {role} is not in {pdf_dir}")
    nodes = S.C.by_reg[(reg, role)]
    plan = batches(S, reg, role, a.chunk_chars)
    todo = plan[:a.max_batches] if a.max_batches else plan
    manifest = {"regulation": reg, "role": role, "version": nodes[0].version,
                "input_file": f"{reg.replace(' ', '_')}_{role}.json", "corpus_sha256": S.fingerprint(reg, role),
                "model": rp.MODEL, "effort": "high", "prompt": a.prompt, "prompt_sha256": hashlib.sha256(SYSTEMS[a.prompt].encode()).hexdigest()[:16],
                "test_case": a.test_case,
                "settings": {"chunk_chars": a.chunk_chars, "hops": a.hops, "max_target_nodes": MAX_TARGET_NODES,
                             "max_context_nodes": MAX_CONTEXT_NODES, "retries": RETRIES},
                "nodes": len(nodes), "chars": size(nodes), "batches": plan,
                "limitations": ["traversal context is complete only within the declared hop/size policy",
                                "definitions are found by a regex over main-body section 2 only",
                                "references into another regulation resolve to that regulation's base version",
                                "an empty batch output is not proof that the batch states no requirement"]}
    log(f"{reg} {role} ({nodes[0].version}): {len(nodes)} nodes, {size(nodes):,} chars -> {len(plan)} batches "
        f"({sum(b['oversized'] for b in plan)} oversized), running {len(todo)}")
    if a.plan_only:
        for b in plan:
            log(f"  {b['id']} {b['name'][:60]:<60} {len(b['primary']):>4} nodes {b['primary_chars']:>7} chars"
                f"{' + ' + str(len(b['scope'])) + ' scope' if b['scope'] else ''}{'  OVERSIZED' if b['oversized'] else ''}")
        dump("extraction_manifest.json", manifest)
        (run / "log.txt").write_text("\n".join(lines) + "\n")
        return

    results = c1(S, reg, role, todo, a.test_case, a.workers, log, a.prompt)
    valid, flagged = validate(S, reg, role, plan, results, a.test_case)
    status = {}
    for b in plan:
        for q in b["primary"]:
            status[q] = {"ok": "processed", "empty_output": "processed", "failed": "failed"}.get(b["status"], "not_processed")
    manifest["coverage"] = {k: sum(v == k for v in status.values()) for k in ("processed", "failed", "not_processed")}
    manifest["node_status"] = status
    manifest["calls"] = {"model_calls": sum(b.get("attempts", 0) for b in todo if not b.get("cache_hit")),
                         "cache_hits": sum(bool(b.get("cache_hit")) for b in todo),
                         "failed_batches": [b["id"] for b in todo if b["status"] == "failed"],
                         "empty_batches": [b["id"] for b in todo if b["status"] == "empty_output"]}
    dump("extraction_manifest.json", manifest)
    dump("c1_candidates.json", {"candidates": valid, "flagged": flagged})
    log(f"C1: {len(valid)} requirements, {len(flagged)} flagged; coverage {manifest['coverage']}; "
        f"calls {manifest['calls']['model_calls']} (+{manifest['calls']['cache_hits']} cached)")

    T = Context(S, reg, role, a.hops)                       # code only from here: no model calls
    out, diags = [], []
    for r in valid:
        ctx = T.for_requirement(r)
        out.append({"id": r["id"], "text": r["text"], "table": r["table"], **({"test_case": r.get("test_case")} if a.test_case else {}),
                    "regulation": reg, "version": r["source"]["version"], "role": role, "source_node_ids": r["source_node_ids"],
                    "source": r["source"], "batch": r["batch"], "validation": r["validation"],
                    "possible_duplicate_of": r["possible_duplicate_of"], **ctx})
        diags += [{"requirement": r["id"], **d} for d in ctx["diagnostics"]]
    dump("requirements_with_context.json", {"regulation": reg, "role": role, "hops": a.hops, "requirements": out})
    dump("context_nodes.json", {q: context_record(S, q, n) for q, n in T.nodes.items()})
    hints = list({(h["regulation"], h["paragraph"]): h for h in T.tr.hints}.values())
    dump("context_diagnostics.json", {"per_requirement": diags, "regulation_hints": hints,
                                      "summary": {k: sum(d["type"] == k for d in diags) for k in sorted({d["type"] for d in diags})}})
    log(f"T ({a.hops} hops): {len(T.nodes)} shared context nodes, {sum(r['context_status'] == 'incomplete' for r in out)} "
        f"requirements with incomplete context, diagnostics {sorted({d['type'] for d in diags})}")
    (run / "log.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
