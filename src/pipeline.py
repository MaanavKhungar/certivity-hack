"""Requirements pipeline: product profile + parsed regulations -> submission JSON.

Stage A  regulation selection   (1 model call over scope + chapter titles of all base regulations, + BM25 safety net)
Stage B  section selection      (1 model call per kept regulation over its table of contents)
Expand   code only              (nodes of the selected sections + nodes they cross-reference)
Stage C  requirement writing    (1 model call per section, concurrent)
Post     code only              (page from the parsed node, dedupe, ids, schema validation, local score)
Every model call is cached on disk by content hash, and every stage writes its result under runs/<case_id>/.
"""
from __future__ import annotations

import hashlib, json, re, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from regparse import Corpus, Node, paragraph_key

MODEL = "claude-opus-5-5"
WORK = Path(__file__).resolve().parent
ROOT = WORK.parent
PKG = ROOT / "package"
CACHE = WORK / "cache"
RUNS = ROOT / "runs"


# ============================================================ model call with cache ============================
def _client():
    import anthropic
    return anthropic.Anthropic(max_retries=4)


def call_json(system: str, user_content, schema: dict, *, max_tokens: int = 16000, effort: str = "high",
              tag: str = "", use_cache: bool = True) -> dict | list:
    """One structured-output call. Cached by hash of (model, system, user_content, schema)."""
    CACHE.mkdir(exist_ok=True)
    payload = json.dumps({"m": MODEL, "s": system, "u": user_content, "sc": schema, "e": effort}, sort_keys=True, ensure_ascii=False)
    h = hashlib.sha256(payload.encode()).hexdigest()[:24]
    cp = CACHE / f"{tag}_{h}.json" if tag else CACHE / f"{h}.json"
    if use_cache and cp.exists():
        return json.loads(cp.read_text())["result"]
    client = _client()
    t = time.time()
    with client.messages.stream(
        model=MODEL, max_tokens=max_tokens,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
        messages=[{"role": "user", "content": user_content}],
    ) as stream:
        msg = stream.get_final_message()
    if msg.stop_reason == "refusal":
        raise RuntimeError(f"model refused ({tag}): {msg.stop_details}")
    if msg.stop_reason == "max_tokens":
        raise RuntimeError(f"output truncated at max_tokens ({tag}); raise max_tokens")
    text = next(b.text for b in msg.content if b.type == "text")
    result = json.loads(text)
    cp.write_text(json.dumps({"tag": tag, "model": msg.model, "seconds": round(time.time() - t, 1),
                              "usage": {"in": msg.usage.input_tokens, "out": msg.usage.output_tokens,
                                        "cache_read": msg.usage.cache_read_input_tokens},
                              "result": result}, ensure_ascii=False, indent=1))
    return result


# ============================================================ helpers over the corpus ===========================
def index_titles() -> dict[str, str]:
    return {e["regulation"]: e["title"] for e in json.loads((PKG / "regulations/index.json").read_text())}


def base_version(reg: str) -> str:
    for e in json.loads((PKG / "regulations/index.json").read_text()):
        if e["regulation"] == reg:
            return next(v["version_label"] for v in e["versions"] if v["role"] == "base")
    raise KeyError(reg)


def scope_text(C: Corpus, reg: str, limit: int = 700) -> str:
    nodes = C.by_reg[(reg, "base")]
    n = next((n for n in nodes if n.doc == "main" and n.label == "1"), None)
    if n is not None:
        txt = n.text
    else:                               # "1 Scope" without a dot lands inside the preamble node: slice from "Scope"
        pre = next((n for n in nodes if n.doc == "main" and n.label == "0"), nodes[0])
        m = re.search(r"\b(?:1\s+)?Scope\b", pre.text)
        txt = pre.text[m.start():] if m else re.sub(r"^.*?Role in the hackathon: \w+ version\.\s*", "", pre.text)
    # include the direct children of the scope paragraph (1.1, 1.2 ...) when they exist
    kids = [k for k in nodes if k.doc == "main" and k.parent == "1"]
    for k in kids[:6]:
        txt += " " + k.label + ". " + k.text
    return (txt[:limit] + " …") if len(txt) > limit else txt


def chapter_titles(C: Corpus, reg: str, max_depth: int = 2) -> list[str]:
    out = []
    for n in C.by_reg[(reg, "base")]:
        if n.doc != "main" or n.depth > max_depth or n.label == "0":
            continue
        t = n.title or (n.text[:70] if n.depth == 1 else None)
        if t:
            out.append(f"{n.label} {t}")
    # annexes: one line each
    seen = []
    for n in C.by_reg[(reg, "base")]:
        if n.doc != "main":
            a = n.doc.split(" Part")[0].split(" Appendix")[0]
            if a not in seen:
                seen.append(a)
    return out + [f"{a} (p{min(n.page_start for n in C.by_reg[(reg,'base')] if n.doc.startswith(a))})" for a in seen]


def toc_lines(C: Corpus, reg: str) -> list[str]:
    """Table of contents incl. annexes, as 'KEY | pages | title' lines. KEY is what stage B returns."""
    lines, nodes = [], C.by_reg[(reg, "base")]
    last_doc = None
    for n in nodes:
        if n.doc != last_doc:
            last_doc = n.doc
            if n.doc != "main":
                first = [m for m in nodes if m.doc == n.doc]
                lines.append(f"{n.doc} | p{first[0].page_start}-{first[-1].page_end} | (annex)")
        if n.depth <= 2 and (n.title or n.depth == 1) and n.label != "0":
            key = n.label if n.doc == "main" else f"{n.doc} {n.label}"
            sub = [m for m in nodes if m.doc == n.doc and (m.label == n.label or m.label.startswith(n.label + "."))]
            lines.append(f"{key} | p{sub[0].page_start}-{sub[-1].page_end} | {n.title or n.text[:80]}")
    return lines


def section_nodes(C: Corpus, reg: str, key: str) -> list[Node]:
    """Nodes of a section key: '6.10', 'Annex 3', 'Annex 4 Part 1', 'Annex 3 1.2'."""
    nodes = C.by_reg[(reg, "base")]
    k = paragraph_key(key)
    if k.startswith("annex"):
        m = re.match(r"^(annex \S+(?: (?:part|appendix) \S+)?)(?: (\S+))?$", k)
        doc, label = m.group(1), m.group(2)
        out = [n for n in nodes if paragraph_key(n.doc) == doc or paragraph_key(n.doc).startswith(doc + " ")]
        if label:
            out = [n for n in out if n.label == label or n.label.startswith(label + ".")]
        return out
    return [n for n in nodes if n.doc == "main" and (n.label == k or n.label.startswith(k + "."))]


REF_PARA = re.compile(r"paragraphs?\s+((?:\d+(?:\.\d+)+\.?)(?:\s*(?:,|and|to)\s*\d+(?:\.\d+)+\.?)*)", re.I)
REF_ANNEX = re.compile(r"Annex(?:es)?\s+(\d+[A-Z]?)(?:,?\s*Part\s+(\d+))?", re.I)
REF_TABLE = re.compile(r"Table\s+(A?\d+[A-Z]?(?:-\d+)?)\b")


def expand_refs(C: Corpus, reg: str, nodes: list[Node], max_annex_nodes: int = 120) -> list[Node]:
    """Nodes referenced from `nodes` (same regulation), excluding ones already present."""
    have = {(n.doc, n.label) for n in nodes}
    extra: dict[tuple, Node] = {}
    all_nodes = C.by_reg[(reg, "base")]
    for n in nodes:
        for m in REF_PARA.finditer(n.text):
            for lab in re.findall(r"\d+(?:\.\d+)+", m.group(1)):
                t = C.find(reg, lab)
                if t and (t.doc, t.label) not in have:
                    extra[(t.doc, t.label)] = t
        for m in REF_ANNEX.finditer(n.text):
            doc = f"Annex {m.group(1)}" + (f" Part {m.group(2)}" if m.group(2) else "")
            sub = [x for x in all_nodes if x.doc == doc or x.doc.startswith(doc + " ")]
            if 0 < len(sub) <= max_annex_nodes:
                for x in sub:
                    if (x.doc, x.label) not in have:
                        extra[(x.doc, x.label)] = x
        for m in REF_TABLE.finditer(n.text):
            for x in all_nodes:
                if m.group(1) in x.tables and (x.doc, x.label) not in have:
                    extra[(x.doc, x.label)] = x
    return list(extra.values())


def fmt_node(n: Node) -> str:
    key = n.label if n.doc == "main" else f"{n.doc} {n.label}"
    pages = f"p{n.page_start}" if n.page_start == n.page_end else f"p{n.page_start}-{n.page_end}"
    return f"[{pages} §{key}] {n.text}"


# ============================================================ BM25 safety net ==================================
def bm25_regulations(C: Corpus, terms: list[str], top: int = 10) -> list[tuple[str, float]]:
    from rank_bm25 import BM25Okapi
    regs = sorted({r for (r, role) in C.by_reg if role == "base"})
    docs = [" ".join(n.text for n in C.by_reg[(r, "base")]).lower() for r in regs]
    tok = [re.findall(r"[a-z0-9]+", d) for d in docs]
    bm = BM25Okapi(tok)
    q = re.findall(r"[a-z0-9]+", " ".join(terms).lower())
    scores = bm.get_scores(q)
    ranked = sorted(zip(regs, scores), key=lambda x: -x[1])
    return [(r, round(float(s), 1)) for r, s in ranked[:top]]


# ============================================================ Stage A =========================================
STAGE_A_SYSTEM = """You are a UNECE vehicle type-approval expert. Given a product profile and the list of UN Regulations
(each with its scope and chapter titles), decide for EVERY regulation whether it applies to this product.

Verdicts: "applies" (the product or one of its parts/functions is within scope and the regulation sets obligations on it),
"maybe" (plausible, depends on a detail not in the profile), "no".

Rules of thumb (follow them):
- Consolidated device regulations supersede the individual ones for new approvals: UN R148 (light-signalling devices)
  replaces R6, R7, R23, R38, R4, R50, R77, R87, R91; UN R149 (road illumination devices) replaces R19, R98, R112, R113,
  R123; UN R150 (retro-reflective devices) replaces R3, R27, R69, R70, R104. Mark the superseded ones "no" when the
  consolidated one applies.
- Installation regulations (UN R48 for M/N/O, UN R53 for L3) apply whenever a lamp or retro-reflector is on the product
  and the product is for that vehicle category.
- Any electronics (LED driver, PCB, control gear, sensors) make the product an electrical/electronic sub-assembly (ESA):
  UN R10 applies.
- Light-source regulations (R37 filament, R99 gas-discharge, R128 LED) apply only to REPLACEABLE light sources.
- Respect the vehicle category: regulations scoped to other categories (L, M2/M3 buses, T tractors ...) are "no".
- Whole-vehicle regulations (crash tests, emissions, braking, steering, noise ...) are "no" unless the product itself is
  the subject of the regulation (e.g. a bumper for R42 / R26 / pedestrian safety, a safety-belt for R16).
Err towards "maybe" rather than "no" when the scope could cover the product: a missed regulation costs far more than an
extra one, which later stages will drop if nothing in it binds the product."""

STAGE_A_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["verdicts"],
    "properties": {"verdicts": {"type": "array", "items": {"type": "object", "additionalProperties": False,
        "required": ["regulation", "verdict", "reason"],
        "properties": {"regulation": {"type": "string"}, "verdict": {"type": "string", "enum": ["applies", "maybe", "no"]},
                       "reason": {"type": "string"}}}}}}


SUPERSEDED = {  # consolidated regulation -> the individual ones it replaces for new approvals
    "UN R148": ["UN R4", "UN R6", "UN R7", "UN R23", "UN R38", "UN R50", "UN R77", "UN R87", "UN R91"],
    "UN R149": ["UN R19", "UN R98", "UN R112", "UN R113", "UN R123"],
    "UN R150": ["UN R3", "UN R27", "UN R69", "UN R70", "UN R104"],
}


def stage_a(C: Corpus, profile: dict, run_dir: Path) -> dict:
    titles = index_titles()
    regs = sorted({r for (r, role) in C.by_reg if role == "base"}, key=lambda r: (len(r), r))
    listing = []
    for r in regs:
        listing.append(f"### {r} — {titles.get(r, '')}\nScope: {scope_text(C, r)}\nChapters: " + "; ".join(chapter_titles(C, r)[:40]))
    user = [{"type": "text", "text": "## PRODUCT PROFILE\n" + json.dumps(profile, indent=1, ensure_ascii=False)
             + "\n\n## REGULATIONS (" + str(len(regs)) + ")\n\n" + "\n\n".join(listing)
             + "\n\nReturn one verdict for every regulation listed, using the exact regulation strings above."}]
    res = call_json(STAGE_A_SYSTEM, user, STAGE_A_SCHEMA, max_tokens=16000, tag="A")
    verdicts = {v["regulation"]: v for v in res["verdicts"]}
    # BM25 safety net: top lexical hits the model said "no" to become "maybe" (unless superseded by a kept consolidated reg)
    bm = bm25_regulations(C, profile.get("search_terms", []) + profile.get("regulated_functions", []), top=8)
    kept_now = {r for r, v in verdicts.items() if v["verdict"] in ("applies", "maybe")}
    superseded = {old for new, olds in SUPERSEDED.items() if new in kept_now for old in olds}
    for r, s in bm:
        if r in verdicts and verdicts[r]["verdict"] == "no" and r not in superseded:
            verdicts[r] = {**verdicts[r], "verdict": "maybe", "reason": verdicts[r]["reason"] + f" [BM25 safety net: lexical score {s}]"}
    kept = [r for r in regs if verdicts.get(r, {}).get("verdict") in ("applies", "maybe")]
    out = {"kept": kept, "verdicts": verdicts, "bm25_top": bm, "prompt_chars": sum(len(x) for x in listing)}
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "stageA.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return out


# ============================================================ Stage B =========================================
STAGE_B_SYSTEM = """You are a UNECE vehicle type-approval expert. You get a product profile and the table of contents of ONE
UN Regulation that applies (or may apply) to the product. Select every section whose paragraphs contain obligations the
PRODUCT must fulfil. Return section KEYS exactly as printed before the first '|' (e.g. "5", "6.10", "Annex 3",
"Annex 4 Part 1").

Include:
- marking / approval-mark provisions that the device must carry (usually "3.3" or "4" in device regulations)
- general technical requirements / general specifications
- the specific-requirement sections for every function or part the product HAS (and only those)
- conformity-of-production requirements (tolerances on measured values)
- test-procedure annexes referenced by the selected sections (photometry grids, stability tests, environmental tests)
Exclude:
- scope, definitions, application-for-approval procedure, approval/communication forms, modifications and extensions,
  penalties, production discontinued, names of technical services, transitional provisions, approval-mark drawings
- sections for functions or vehicle categories the product does not have
Prefer selecting a whole chapter (e.g. "5") over listing its sub-sections, unless only some sub-sections apply.
Err towards including: a missed section loses all its requirements; an extra one is cheap."""

STAGE_B_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["selected", "excluded"],
    "properties": {"selected": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                "required": ["key", "reason"], "properties": {"key": {"type": "string"}, "reason": {"type": "string"}}}},
                   "excluded": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                "required": ["key", "reason"], "properties": {"key": {"type": "string"}, "reason": {"type": "string"}}}}}}


def stage_b_one(C: Corpus, profile: dict, reg: str, verdict: dict) -> dict:
    toc = toc_lines(C, reg)
    user = [{"type": "text", "text": "## PRODUCT PROFILE\n" + json.dumps(profile, indent=1, ensure_ascii=False)
             + f"\n\n## REGULATION {reg} — {index_titles().get(reg,'')}\nWhy it was selected: {verdict.get('reason','')}"
             + "\n\n## TABLE OF CONTENTS (key | pages | title)\n" + "\n".join(toc)}]
    res = call_json(STAGE_B_SYSTEM, user, STAGE_B_SCHEMA, max_tokens=8000, tag=f"B_{reg.replace(' ','_')}")
    # keep only keys that resolve to nodes
    sel = []
    for s in res["selected"]:
        n = section_nodes(C, reg, s["key"])
        if n:
            sel.append({**s, "n_nodes": len(n), "pages": [n[0].page_start, n[-1].page_end]})
        else:
            sel.append({**s, "n_nodes": 0, "error": "key not found in parsed structure"})
    return {"regulation": reg, "selected": sel, "excluded": res["excluded"], "toc_lines": len(toc)}


def stage_b(C: Corpus, profile: dict, a: dict, run_dir: Path, workers: int = 6) -> dict:
    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(lambda r: stage_b_one(C, profile, r, a["verdicts"][r]), a["kept"]))
    out = {r["regulation"]: r for r in results}
    (run_dir / "stageB.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return out


# ============================================================ Stage C =========================================
STAGE_C_SYSTEM = """You are a regulatory engineer at a company that manufactures vehicle components. You turn the text of a
UN Regulation into the list of ENGINEERING REQUIREMENTS a specific product must fulfil, one obligation per requirement,
each citing the paragraph it comes from. A requirement is matched one-to-one against an expert-written golden list that
states the same obligation (same subject, condition and constraint), so granularity and completeness both matter.

How to write requirements:
- One obligation per requirement. If a paragraph contains several obligations (several sentences with "shall", a list
  (a)/(b)/(c), several limits, several conditions), write one requirement per obligation. Never merge.
- Tables: write out the values that apply to this product (e.g. the row for the product's class/colour), with their
  conditions (angles, categories). One requirement per row or per coherent group of values of the same kind.
- Phrase it as the product's obligation in the regulation's own words: "The rear position lamp shall ...",
  "The ESA shall ...", "The markings shall ...". Keep the condition in the sentence ("Where optional lamps are installed, ...",
  "When measured according to Annex 3, ...", "For M1 and N1 vehicles, ...").
- Keep the regulation's numbers, units and tolerances exactly.
- Include: marking/approval-mark provisions the device must carry; design/construction rules; photometric, colorimetric,
  electrical, EMC, mechanical, environmental limits; installation rules (position, number, visibility, electrical connections,
  tell-tales) for functions present; conformity-of-production tolerances; test conditions that constrain the product.
- Exclude: definitions; what the applicant / manufacturer / authority / technical service must DO (submit samples, documents,
  grant approval, notify); descriptions of test set-ups that do not state a pass/fail criterion for the product; provisions
  for functions, categories, classes or vehicle categories the product does not have; "may" permissions without an obligation.
- Use the paragraph label exactly as printed in the [§...] tag of the text you cite. For a table, cite the paragraph that
  introduces it. Cite the paragraph the text is IN, not a paragraph it merely references.
- test_case: one short sentence on how a test engineer would verify it (measurement, inspection, test), optional.
Err towards including a borderline product obligation; the golden list is thorough."""

STAGE_C_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["requirements"],
    "properties": {"requirements": {"type": "array", "items": {"type": "object", "additionalProperties": False,
        "required": ["paragraph", "text", "test_case", "why"],
        "properties": {"paragraph": {"type": "string", "description": "exactly as in the [§...] tag, e.g. '6.10.4.2' or 'Annex 3 1.2'"},
                       "text": {"type": "string"}, "test_case": {"type": ["string", "null"]},
                       "why": {"type": "string", "description": "why this binds the product (one line)"}}}}}}


def golden_examples(k: int = 14) -> str:
    """Style examples from the worked example (dev-rear-lamp expected.json), spread over its regulations."""
    exp = json.loads((PKG / "cases/dev-rear-lamp/expected.json").read_text())["requirements"]
    by = {}
    for r in exp:
        by.setdefault(r["source"]["regulation"], []).append(r)
    out, i = [], 0
    while len(out) < k:
        for reg, lst in by.items():
            if i < len(lst) and len(out) < k:
                r = lst[(i * 7) % len(lst)]
                out.append(f"- [{reg} §{r['source']['paragraph']}] {r['text']}")
        i += 1
    return "\n".join(out)


def stage_c_one(C: Corpus, profile: dict, reg: str, sec: dict, examples: str) -> dict:
    nodes = section_nodes(C, reg, sec["key"])
    refs = expand_refs(C, reg, nodes)
    body = "\n".join(fmt_node(n) for n in nodes)
    ref_txt = "\n".join(fmt_node(n) for n in refs[:150])
    user = [{"type": "text", "text":
             "## PRODUCT PROFILE\n" + json.dumps(profile, indent=1, ensure_ascii=False)
             + "\n\n## STYLE EXAMPLES (from an expert-written list for a different product; match this granularity and phrasing)\n" + examples
             + f"\n\n## REGULATION {reg} — section {sec['key']} ({sec.get('reason','')})\n"
             + "Write the requirements for THIS section only. Each line below is one paragraph: [page §label] text.\n\n" + body
             + ("\n\n## REFERENCED PARAGRAPHS (context only; cite them only if the obligation is stated there and binds the product)\n" + ref_txt if ref_txt else "")}]
    res = call_json(STAGE_C_SYSTEM, user, STAGE_C_SCHEMA, max_tokens=32000, tag=f"C_{reg.replace(' ','_')}_{paragraph_key(sec['key']).replace(' ','-')}")
    return {"regulation": reg, "key": sec["key"], "n_nodes": len(nodes), "n_refs": len(refs),
            "prompt_chars": len(user[0]["text"]), "requirements": res["requirements"]}


def stage_c(C: Corpus, profile: dict, b: dict, run_dir: Path, workers: int = 8) -> list[dict]:
    jobs = [(reg, s) for reg, r in b.items() for s in r["selected"] if s.get("n_nodes", 0) > 0]
    ex_txt = golden_examples()
    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(lambda j: stage_c_one(C, profile, j[0], j[1], ex_txt), jobs))
    (run_dir / "stageC.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    return results


# ============================================================ Post-processing ================================
def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", t.lower())


def assemble(C: Corpus, results: list[dict], case_id: str, team_id: str, run_dir: Path) -> dict:
    reqs, dropped, seen = [], [], {}
    for r in results:
        reg = r["regulation"]
        for q in r["requirements"]:
            n = C.find(reg, q["paragraph"])
            if n is None:
                dropped.append({**q, "regulation": reg, "reason": "paragraph not found in parsed structure"}); continue
            key = (reg, n.key)
            norm = _norm(q["text"])
            dup = next((s for s in seen.get(key, []) if _similar(s, norm)), None)
            if dup:
                dropped.append({**q, "regulation": reg, "reason": "near-duplicate of an earlier requirement"}); continue
            seen.setdefault(key, []).append(norm)
            para = n.label if n.doc == "main" else f"{n.doc} {n.label}"
            item = {"id": f"R-{len(reqs)+1:04d}", "text": q["text"].strip()[:2000],
                    "source": {"regulation": reg, "version": base_version(reg), "page": n.page_start, "paragraph": para},
                    "why": q.get("why", ""), "section": r["key"]}
            if q.get("test_case"):
                item["test_case"] = q["test_case"]
            reqs.append(item)
    sub = {"team_id": team_id, "case_id": case_id, "requirements": reqs}
    (run_dir / "submission.json").write_text(json.dumps(sub, indent=1, ensure_ascii=False))
    (run_dir / "dropped.json").write_text(json.dumps(dropped, indent=1, ensure_ascii=False))
    return sub


def _similar(a: str, b: str, thr: float = 0.92) -> bool:
    if a == b:
        return True
    import difflib
    return difflib.SequenceMatcher(None, a, b).ratio() >= thr


def validate(sub: dict) -> list[str]:
    """Same checks as tools/validate.py: schema + regulation/version in index + unique ids."""
    import jsonschema
    schema = json.loads((PKG / "schemas/submission_main.schema.json").read_text())
    problems = [f"{'/'.join(map(str, e.absolute_path))}: {e.message}" for e in jsonschema.Draft202012Validator(schema).iter_errors(sub)]
    index = json.loads((PKG / "regulations/index.json").read_text())
    versions = {e["regulation"]: {v["version_label"] for v in e["versions"]} for e in index}
    ids = set()
    for i, r in enumerate(sub["requirements"]):
        if r["id"] in ids:
            problems.append(f"requirements[{i}].id duplicate {r['id']}")
        ids.add(r["id"])
        s = r["source"]
        if s["regulation"] not in versions:
            problems.append(f"requirements[{i}].source.regulation {s['regulation']!r} not in index")
        elif s["version"] not in versions[s["regulation"]]:
            problems.append(f"requirements[{i}].source.version {s['version']!r} not a version of {s['regulation']}")
    return problems


def score_local(sub: dict, expected: dict) -> dict:
    """Regulation- and page-level P/R/F1, same logic as tools/score_local.py."""
    def cited(doc): return [(r["source"]["regulation"], r["source"].get("version", ""), r["source"]["page"]) for r in doc["requirements"]]
    def paras(doc):
        on_page, anywhere = {}, {}
        for r in doc["requirements"]:
            s, k = r["source"], paragraph_key(r["source"].get("paragraph"))
            if k:
                on_page.setdefault((s["regulation"], s.get("version", ""), s["page"]), set()).add(k)
                anywhere.setdefault((s["regulation"], s.get("version", "")), set()).add(k)
        return on_page, anywhere
    def prf(m, ns, ne):
        p, r = (m / ns if ns else 0), (m / ne if ne else 0)
        return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / (p + r), 3) if p + r else 0, "submitted": ns, "expected": ne, "matched": m}
    c, w = cited(sub), cited(expected)
    sr, gr = {r for r, _, _ in c}, {r for r, _, _ in w}
    sp, gp = set(c), set(w)
    sop, sany = paras(sub); gop, gany = paras(expected)
    correct = {p for p in sp if p in gp or sop.get(p, set()) & gany.get(p[:2], set())}
    found = {p for p in gp if p in sp or gop.get(p, set()) & sany.get(p[:2], set())}
    pl = prf(len(correct), len(sp), len(gp)); pl["recall"] = round(len(found) / len(gp), 3) if gp else 0
    return {"regulation_level": prf(len(sr & gr), len(sr), len(gr)), "page_level": pl,
            "regulations_submitted": sorted(sr), "regulations_expected": sorted(gr)}


# ============================================================ driver ==========================================
def run(case_dir: Path, profile: dict, team_id: str = "team-00", parsed_dir: Path = WORK / "parsed") -> dict:
    C = Corpus(parsed_dir)
    case_id = Path(case_dir).name
    run_dir = RUNS / case_id
    a = stage_a(C, profile, run_dir)
    b = stage_b(C, profile, a, run_dir)
    c = stage_c(C, profile, b, run_dir)
    sub = assemble(C, c, case_id, team_id, run_dir)
    return {"stageA": a, "stageB": b, "stageC": c, "submission": sub, "problems": validate(sub)}
