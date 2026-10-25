"""Sub-goal: which requirements derived from an old version of a UN Regulation must change in the new version, and why
(generated from the notebook).

1. align   : parse both PDFs, split paragraphs into sentences and match every sentence to its best counterpart in the other
             version by content, so moved / renumbered text is not mistaken for a change
2. modified: one model call per old paragraph whose requirements may be hit (its text changed, or the new paragraphs it
             moved to gained text) -> unchanged / modified / removed per requirement, with the old and new source quotes
3. added   : new sentences with no old counterpart, inside the chapters the old requirement set covers -> one call per new
             paragraph extracts the new obligations, in the style of the old requirement set
4. reasons : written in code from a word diff of the quotes, so they say exactly what changed in the regulation
"""
from __future__ import annotations
import difflib, json, random, re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from rapidfuzz import fuzz, process
from regparse import parse_pdf, extract_pages, paragraph_key
from pipeline import call_json, RUNS

SAME = 97      # a sentence matched at >= SAME is unchanged (only editorial differences)
NOVEL = 90     # a new sentence whose best old match is below NOVEL is new text
LINK = 60      # below LINK a match is not a counterpart at all
NORMATIVE_RE = re.compile(r"\b(?:shall|must|may)\b|\b(?:is|are) not required\b", re.I)


# ============================================================ align ===========================================
SPLIT_RE = re.compile(r"(?<=[.;:])\s+(?=(?:\([a-z0-9]{1,4}\)\s*)?[A-Z(\"“])")


def sentences(text: str) -> list[str]:
    out = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        out += [s.strip() for s in (SPLIT_RE.split(line) if not line.startswith("|") else [line]) if s.strip()]
    return out


def norm(s: str) -> str:
    s = s.casefold().replace("%", " per cent").replace("“", '"').replace("”", '"').replace("’", "'")
    return " ".join(s.split()).rstrip(".;:, ")


ORDER_SENSITIVE = {
    "above", "after", "before", "behind", "below", "between", "except",
    "fewer", "following", "front", "greater", "less", "more", "not",
    "only", "preceding", "unless", "until", "without",
}
FUNCTION_WORDS = {
    "a", "an", "and", "as", "at", "be", "both", "by", "during", "for",
    "from", "in", "is", "of", "on", "or", "shall", "should", "than", "the",
    "to", "when", "where", "which", "while", "with",
}


def _words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", norm(s))


def pure_reordering_equivalent(a: str, b: str) -> bool:
    """Conservatively identify editorial word movement without hiding changes.

    Both sides must have exactly the same normalized words and duplicate counts.
    Order-sensitive regulatory language is excluded, and most content words must
    retain their relative order. Ambiguous reorderings therefore continue to the
    existing review path instead of being silently marked unchanged.
    """
    wa, wb = _words(a), _words(b)
    if wa == wb or sorted(wa) != sorted(wb) or ORDER_SENSITIVE.intersection(wa):
        return wa == wb
    ca = [w for w in wa if w not in FUNCTION_WORDS]
    cb = [w for w in wb if w not in FUNCTION_WORDS]
    if not ca:
        return False
    common = sum(i2 - i1 for tag, i1, i2, _j1, _j2 in difflib.SequenceMatcher(
        None, ca, cb, autojunk=False
    ).get_opcodes() if tag == "equal")
    return common / max(len(ca), len(cb)) >= 0.8


def _sent_table(nodes) -> list[dict]:
    return [{"key": n.key, "i": i, "text": s, "norm": norm(s)} for n in nodes for i, s in enumerate(sentences(n.text))]


def _best(rows: list[dict], other: list[dict]):
    """For every row, the best match among `other` rows: (score, other row) — exact hits first, fuzzy for the rest."""
    exact = {}
    for o in other:
        exact.setdefault(o["norm"], o)
    choices = [o["norm"] for o in other]
    for r in rows:
        if r["norm"] in exact:
            r["score"], r["match"] = 100.0, exact[r["norm"]]
            continue
        hit = process.extractOne(r["norm"], choices, scorer=fuzz.ratio, processor=None) if choices else None
        r["score"], r["match"] = (hit[1], other[hit[2]]) if hit else (0.0, None)


def align(old_pdf: Path, new_pdf: Path) -> dict:
    old, new = parse_pdf(old_pdf, "base"), parse_pdf(new_pdf, "latest")
    so, sn = _sent_table(old), _sent_table(new)
    _best(so, sn); _best(sn, so)
    # an old sentence repeated in an extra place (a step reused by a new procedure, boilerplate of a new section) is new
    # text there. When a sentence occurs k more times than before, the k copies in the most-new surroundings are the new ones.
    parent = {n.key: (n.key.rsplit(".", 1)[0] if "." in n.key else n.key) for n in new}
    fresh = Counter(parent[r["key"]] for r in sn if r["score"] < NOVEL)
    size = Counter(parent[r["key"]] for r in sn)
    n_old, by_norm = Counter(r["norm"] for r in so), {}
    for r in sn:
        r["copy"] = False
        if r["score"] == 100 and len(r["norm"]) > 30 and not r["text"].startswith("|"):
            by_norm.setdefault(r["norm"], []).append(r)
    for nm, rows in by_norm.items():
        extra = len(rows) - n_old[nm]
        ranked = sorted(rows, key=lambda r: -fresh[parent[r["key"]]] / size[parent[r["key"]]])
        for r in ranked[:max(extra, 0)]:
            r["copy"] = fresh[parent[r["key"]]] > 0
    for r in sn:
        r["new"] = r["score"] < NOVEL or r["copy"]
    return {"old": {n.key: n for n in old}, "new": {n.key: n for n in new}, "so": so, "sn": sn,
            "reg": old[0].reg, "v_old": old[0].version, "v_new": new[0].version,
            "pages_new": extract_pages(new_pdf)[2]}


def page_of(A: dict, key: str, quote: str | None) -> int:
    """Page of the new file where the quote starts (falls back to the paragraph's first page)."""
    node = A["new"][key]
    if quote:
        probe = norm(quote)[:60]
        for p, lines in A["pages_new"]:
            if node.page_start <= p <= node.page_end and probe and probe in norm(" ".join(lines)):
                return p
    return node.page_start


def chapter(key: str) -> str:
    """'5.1.4' -> '5', 'annex 3 1.2' -> 'annex 3', 'annex 4 part 1 1.2' -> 'annex 4 part 1'."""
    parts = key.split()
    return " ".join(parts[:-1]) if len(parts) > 1 else key.split(".")[0]


# ============================================================ prompts =========================================
MOD_SYSTEM = """You maintain a list of engineering requirements that were derived from an OLD version of a UN Regulation.
A NEW version of the regulation was published. You get ONE old paragraph, the requirements derived from it, and the new
paragraph(s) its text now lives in (a code-made sentence alignment tells you where each old sentence went).

For EVERY requirement decide:
- "unchanged": the obligation it states is still in the new version with the same subject, condition and constraint.
  Judge the sentence(s) the requirement comes from, not their surroundings: if those words are the same in substance, it
  is unchanged even when the paragraph was renumbered, moved, split, got a new heading, or now sits in a narrower
  procedure next to new text. Moving words or phrases without changing who must do what, under which conditions, is also
  unchanged. Purely editorial differences do NOT count either: "%" written as "per cent", a cross-reference renumbered,
  punctuation, capitalisation.
- "modified": the words of its source sentence(s) changed: a value, a condition, the subject, an added or removed
  condition/list item, a changed procedure step, or even one qualifying word added or dropped ("the target" -> "the
  relevant target"). First split the old requirement conceptually into independent semantic obligations or constraints.
  Write new_text for the obligation(s) whose meaning actually changed. Omit an independent sibling constraint when it is
  unchanged and can be removed without changing the subject, trigger, applicability or meaning of the changed obligation.
  For example, if one sentence independently specifies both test direction and intervention distance, and only the
  intervention-distance rule changes, new_text should state the changed intervention-distance rule and need not repeat
  the unchanged direction rule. Do NOT omit shared conditions or context needed to make the changed obligation accurate.
  Within the retained obligation, make a MINIMAL EDIT of the old requirement: copy its wording and change only the words
  the regulation changed (replace changed words, insert added words or list items, delete removed ones).
  Never add content the old requirement did not contain
  (if the old requirement was cut off mid-list or mid-sentence, end the new one at the last complete item and drop
  the incomplete fragment). new_text has a hard limit of 2000 characters. Only if the edited text would be longer,
  make room by dropping the sub-items of the LONGEST fully unchanged sub-list first (then the next longest, only as
  many as needed), keeping its lead-in line and ending it with ";" instead of "i.e.:" (e.g. "(c) There are no external
  influences affecting the physical sensing capabilities;"). Every changed or added part stays in. Give new_paragraph
  (the label of the new paragraph it now comes from, exactly as listed).
- "removed": the provision it comes from no longer exists in the new version.
Also give old_quote: the exact text of the OLD paragraph the requirement covers (ALL of it, e.g. the whole list when
the requirement contains a list), and new_quote: the corresponding exact text of the NEW paragraph (null when removed).
Quote verbatim from the paragraphs shown: code diffs the two quotes to write the reason, so every change must be inside
them. In note, state only material changes to the obligation; do not report harmless grammar, pluralisation, reordered
wording, or context that was deliberately omitted as an independent unchanged sibling constraint."""

MOD_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["decisions"], "properties": {
    "decisions": {"type": "array", "items": {"type": "object", "additionalProperties": False,
        "required": ["requirement_id", "decision", "new_paragraph", "new_text", "old_quote", "new_quote", "note"],
        "properties": {"requirement_id": {"type": "string"},
                       "decision": {"type": "string", "enum": ["unchanged", "modified", "removed"]},
                       "new_paragraph": {"type": ["string", "null"]}, "new_text": {"type": ["string", "null"]},
                       "old_quote": {"type": "string"}, "new_quote": {"type": ["string", "null"]},
                       "note": {"type": "string"}}}}}}

ADD_SYSTEM = """You maintain a list of engineering requirements derived from a UN Regulation. A new version of the
regulation was published. You get ONE paragraph of the new version in which sentences marked «NEW» have no counterpart in
the old version (or repeat an old sentence in a new place, e.g. a step reused by a new procedure: that is new here), plus the requirements that already cover this paragraph.

Write the requirements the NEW sentences add, exactly as the existing requirement set would have written them:
- Preserve the source proposition before improving style. Keep every actor, action, condition, exception, negation,
  comparator, number, unit, direction and cross-reference. Do not strengthen or weaken deontic force: in particular,
  do not change "is/are" into "shall", "shall" into "may", or an exception into a positive obligation.
- Prefer a minimal grammatical transformation of the source sentence. Do not add scene-setting phrases such as
  "during the test procedure" unless they are required to resolve an otherwise ambiguous subject or condition.
- A passive source with no explicit actor must remain passive; never invent "the system", "the test system", the
  manufacturer or the Technical Service as the actor merely to make the requirement active-voice.
- one obligation per requirement. A sentence is normally ONE requirement, even when it lists alternatives, cases
  (forwards / rearwards), exemption conditions (a) (b), or several attributes of the same thing. Split only (1) the
  steps of a procedure "(a) Position..., (b) Hold..., (c) Release...", one requirement per step, (2) enumerated items
  that each set their own value or target ("as close to: (a) the maximum creeping speed forwards, (b) 4 km/h
  rearwards" -> one requirement per item, each repeating the shared sentence), and (3) a sentence with two different
  actions ("shall be deactivated ..., and may be combined ..."). Alternative conditions of ONE rule ("is not required
  when: (a) ...; or (b) ...") stay together.
- keep conditions ("When...", "While...", "If...") in the requirement
- make each requirement self-contained (resolve "it", "the vehicle", the procedure name from the paragraph heading)
- an exemption or permission that bounds an obligation ("is not required when", "may be combined") counts too
- follow the STYLE EXAMPLES and the list of paragraphs that got NO requirements: if the new text is of a kind the
  existing set never turns into requirements (headings, pure definitions, notes, administrative text), return nothing
- do not repeat what the ALREADY COVERED requirements state
Give source_quote: the exact NEW sentence(s) each requirement comes from (verbatim)."""

ADD_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["added"], "properties": {
    "added": {"type": "array", "items": {"type": "object", "additionalProperties": False,
        "required": ["new_text", "source_quote"],
        "properties": {"new_text": {"type": "string"}, "source_quote": {"type": "string"}}}}}}


# ============================================================ stage 2: old requirements ======================
def _old_key(req: dict) -> str:
    return paragraph_key(req["source"]["paragraph"])


def affected_groups(A: dict, reqs: list[dict]) -> list[dict]:
    """Old paragraphs cited by requirements whose text changed, or whose new home gained text."""
    novel_keys = {s["key"] for s in A["sn"] if s["score"] < NOVEL}   # copies do not hit old requirements
    by_key: dict[str, list[dict]] = {}
    for r in reqs:
        by_key.setdefault(_old_key(r), []).append(r)
    groups = []
    for key, rs in by_key.items():
        if key not in A["old"]:
            continue
        ss = [s for s in A["so"] if s["key"] == key]
        partners = [s["match"]["key"] for s in ss if s["match"] and s["score"] >= LINK]
        if key in A["new"]:
            partners.append(key)
        partners = list(dict.fromkeys(partners))
        changed = any(s["score"] < SAME for s in ss) or any(p in novel_keys for p in partners)
        if changed:
            groups.append({"key": key, "reqs": rs, "sents": ss, "partners": partners})
    return groups


def _fmt_new(A: dict, key: str) -> str:
    n = A["new"][key]
    lines = []
    for s in (s for s in A["sn"] if s["key"] == key):
        tag = ("«NEW, repeated from old §" + s["match"]["key"] + "» ") if s.get("copy") else ("«NEW» " if s["score"] < NOVEL else "")
        lines.append(f"  {tag}{s['text']}")
    return f"§{n.label if n.doc == 'main' else n.key} (page {n.page_start}) [{n.heading}]\n" + "\n".join(lines)


def modify_one(A: dict, g: dict) -> dict:
    o = A["old"][g["key"]]
    align_lines = []
    for s in g["sents"]:
        m = s["match"]
        if m is None or s["score"] < LINK:
            align_lines.append(f"- '{s['text']}'  ->  NO COUNTERPART (deleted)")
        elif s["score"] >= SAME:
            align_lines.append(f"- '{s['text']}'  ->  unchanged in §{m['key']}")
        else:
            align_lines.append(f"- '{s['text']}'  ->  changed ({s['score']:.0f}%), now §{m['key']}: '{m['text']}'")
    user = [{"type": "text", "text":
             f"## REGULATION {A['reg']}: old version {A['v_old']} -> new version {A['v_new']}\n\n"
             f"## OLD PARAGRAPH §{g['key']} (page {o.page_start}) [{o.heading}]\n{o.text}\n\n"
             f"## REQUIREMENTS DERIVED FROM IT\n" + "\n".join(f"{r['id']}: {r['text']}" for r in g["reqs"]) +
             f"\n\n## SENTENCE ALIGNMENT (old sentence -> new version)\n" + "\n".join(align_lines) +
             f"\n\n## NEW PARAGRAPHS (sentences marked «NEW» have no counterpart in the old version)\n" +
             "\n\n".join(_fmt_new(A, p) for p in g["partners"])}]
    res = call_json(MOD_SYSTEM, user, MOD_SCHEMA, max_tokens=16000, tag=f"SG_mod_{g['key'].replace(' ', '_')}")
    return {"key": g["key"], "partners": g["partners"], "decisions": res["decisions"]}


# ============================================================ stage 3: added =================================
def style_examples(A: dict, reqs: list[dict], k: int = 18, seed: int = 0) -> tuple[str, str]:
    """(sample of old requirements, sample of in-scope old paragraphs that got no requirement)."""
    rnd = random.Random(seed)
    ex = rnd.sample(reqs, min(k, len(reqs)))
    cited = {_old_key(r) for r in reqs}
    scope = {chapter(k) for k in cited}
    none = [n for k, n in A["old"].items() if k not in cited and chapter(k) in scope and len(n.text) > 40]
    none = rnd.sample(none, min(10, len(none)))
    return ("\n".join(f"- [§{r['source']['paragraph']}] {r['text']}" for r in ex),
            "\n".join(f"- §{n.key}: {n.text[:220]}" for n in none))


def add_one(A: dict, key: str, covered: list[str], examples: tuple[str, str]) -> dict:
    user = [{"type": "text", "text":
             f"## REGULATION {A['reg']}: new version {A['v_new']}\n\n## STYLE EXAMPLES (existing requirements)\n{examples[0]}\n\n"
             f"## OLD PARAGRAPHS THAT GOT NO REQUIREMENT (do not extract this kind of text)\n{examples[1]}\n\n"
             f"## PARAGRAPH\n{_fmt_new(A, key)}\n\n## ALREADY COVERED\n" + ("\n".join(f"- {c}" for c in covered) or "(none)")}]
    res = call_json(ADD_SYSTEM, user, ADD_SCHEMA, max_tokens=12000, tag=f"SG_add_{key.replace(' ', '_')}")
    return {"key": key, "added": res["added"]}


def added_keys(A: dict, reqs: list[dict]) -> list[str]:
    """New paragraphs with new text, inside the chapters the old requirement set covers (mapped to the new numbering)."""
    cited = {_old_key(r) for r in reqs}
    scope = {chapter(k) for k in cited}
    scope |= {chapter(s["match"]["key"]) for s in A["so"] if s["key"] in cited and s["match"] and s["score"] >= LINK}
    keys = [s["key"] for s in A["sn"] if s["new"] and chapter(s["key"]) in scope]
    return list(dict.fromkeys(keys))


# ============================================================ stage 4: reasons + assembly ====================
def diff_ops(a: str, b: str) -> list[str]:
    a, b = a.split(), b.split()
    ops = []
    for t, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        x, y = " ".join(a[i1:i2]), " ".join(b[j1:j2])
        if t == "replace":
            ops.append(f"'{x}' became '{y}'")
        elif t == "insert":
            ops.append(f"'{y}' was added")
        elif t == "delete":
            ops.append(f"'{x}' was removed")
    return ops


def _label(A: dict, key: str, side: str = "new") -> str:
    n = A[side].get(key)
    return (n.label if n.doc == "main" else n.key.title().replace("Annex", "Annex")) if n else key


def _short(s: str, n: int) -> str:
    s = " ".join(s.replace("|", " ").split())
    return s if len(s) <= n else s[:n] + "…"


def build_changes(A: dict, mods: list[dict], adds: list[dict]) -> list[dict]:
    reg, v = A["reg"], A["v_new"]
    out = []
    modified_quotes: dict[str, list[str]] = {}
    for g in mods:
        for d in g["decisions"]:
            if d["decision"] == "unchanged":
                continue
            if (d["decision"] == "modified" and d.get("new_quote") and
                    pure_reordering_equivalent(d["old_quote"], d["new_quote"])):
                # Suppress only a conservative reorder-only false positive.
                continue
            old_lbl = _label(A, g["key"], "old")
            if d["decision"] == "removed":
                out.append({"requirement_id": d["requirement_id"], "change": "removed",
                            "reason": f"Paragraph {old_lbl} of {reg} was amended in version {v}; the provision "
                                      f"'{_short(d['old_quote'], 100)}' no longer appears."})
                continue
            nk = paragraph_key(d["new_paragraph"] or "")
            if nk not in A["new"]:
                nk = g["partners"][0] if g["partners"] else g["key"]
            new_lbl = _label(A, nk)
            ops = diff_ops(d["old_quote"], d["new_quote"] or "")
            where = new_lbl if nk == g["key"] else f"{new_lbl} (formerly {old_lbl})"
            reason = f"Paragraph {where} of {reg} was amended in version {v}: " + ("; ".join(ops) if ops else d["note"]) + "."
            out.append({"requirement_id": d["requirement_id"], "change": "modified", "new_text": d["new_text"][:2000],
                        "reason": reason, "source": {"page": page_of(A, nk, d["new_quote"]), "paragraph": new_lbl}})
            modified_quotes.setdefault(nk, []).append(d["new_quote"] or "")
    for a in adds:
        lbl = _label(A, a["key"])
        for x in a["added"]:
            quote = x["source_quote"]
            # Procedure steps may inherit "shall" from a lead-in. Block only
            # an unsupported upgrade from "is/are to be" to invented "shall".
            if (re.search(r"\b(?:is|are) to be\b", quote, re.I) and
                    not NORMATIVE_RE.search(quote) and re.search(r"\bshall\b", x["new_text"], re.I)):
                continue
            # Avoid emitting an added duplicate already covered by a modified
            # requirement from the same paragraph.
            enumerated_items = re.findall(r"\([a-z0-9ivx]{1,4}\)", quote, re.I)
            if (len(enumerated_items) >= 2 and any(
                    fuzz.partial_token_set_ratio(quote, q) == 100
                    for q in modified_quotes.get(a["key"], []))):
                continue
            out.append({"requirement_id": None, "change": "added", "new_text": x["new_text"][:2000],
                        "reason": f"Paragraph {lbl} of {reg} is new in version {v}: '{_short(quote, 100)}'.",
                        "source": {"page": page_of(A, a["key"], quote), "paragraph": lbl}})
    return out


def validate_subgoal(sub: dict, old_ids: set[str]) -> list[str]:
    import jsonschema
    from pipeline import PKG
    schema = json.loads((PKG / "schemas/submission_subgoal.schema.json").read_text())
    errs = [f"{'/'.join(map(str, e.path))}: {e.message}" for e in jsonschema.Draft202012Validator(schema).iter_errors(sub)]
    for i, c in enumerate(sub["changes"]):
        if c["change"] in ("modified", "removed") and c.get("requirement_id") not in old_ids:
            errs.append(f"changes/{i}: unknown requirement_id {c.get('requirement_id')}")
    return errs


def run(case_dir: Path | None = None, team_id: str = "team-00", workers: int = 8, *, old_pdf: Path | None = None,
        new_pdf: Path | None = None, reqs: list[dict] | None = None, case_id: str | None = None) -> dict:
    """A sub-goal case folder (old.pdf, new.pdf, requirements_old.json), or explicit paths + requirements for testing."""
    if case_dir is not None:
        case_dir = Path(case_dir)
        case_id = case_id or (case_dir.name if case_dir.name.startswith("subgoal") else f"subgoal-{case_dir.name}")
        old_pdf, new_pdf = old_pdf or case_dir / "old.pdf", new_pdf or case_dir / "new.pdf"
        if reqs is None:
            reqs = json.loads((case_dir / "requirements_old.json").read_text())
    reqs = reqs["requirements"] if isinstance(reqs, dict) else reqs
    run_dir = RUNS / case_id; run_dir.mkdir(parents=True, exist_ok=True)
    A = align(old_pdf, new_pdf)

    groups = affected_groups(A, reqs)
    with ThreadPoolExecutor(workers) as ex:
        mods = list(ex.map(lambda g: modify_one(A, g), groups))

    by_id = {r["id"]: r for r in reqs}
    covered: dict[str, list[str]] = {}
    for g in mods:                      # what already covers each new paragraph: modified texts + untouched requirements
        for d in g["decisions"]:
            if d["decision"] == "removed":
                continue
            k = paragraph_key(d["new_paragraph"] or "") if d["decision"] == "modified" else None
            for p in ([k] if k else g["partners"]):
                covered.setdefault(p, []).append(d["new_text"] if d["decision"] == "modified" else by_id[d["requirement_id"]]["text"])
    for r in reqs:
        k = _old_key(r)
        if k in A["new"] and all(g["key"] != k for g in groups):
            covered.setdefault(k, []).append(r["text"])
    examples = style_examples(A, reqs)
    akeys = added_keys(A, reqs)
    with ThreadPoolExecutor(workers) as ex:
        adds = list(ex.map(lambda k: add_one(A, k, covered.get(k, []), examples), akeys))

    sub = {"team_id": team_id, "case_id": case_id, "changes": build_changes(A, mods, adds)}
    errs = validate_subgoal(sub, set(by_id))
    (run_dir / "stage_mod.json").write_text(json.dumps(mods, indent=1, ensure_ascii=False))
    (run_dir / "stage_add.json").write_text(json.dumps(adds, indent=1, ensure_ascii=False))
    (run_dir / "changes.json").write_text(json.dumps(sub, indent=1, ensure_ascii=False))
    return {"submission": sub, "errors": errs, "groups": [g["key"] for g in groups], "added_keys": akeys, "run_dir": run_dir}


# ============================================================ local judge ====================================
JUDGE_SYSTEM = """You grade a submission for this task: given requirements derived from an old version of a UN Regulation
and the new version, list which requirements must change (modified / removed, by requirement id) and which new
requirements are added. You get the EXPECTED changes (from regulatory experts) and the SUBMITTED changes.

For every expected change find the submitted change that corresponds to it: for modified/removed the one with the same
requirement_id; for added the submitted added requirement stating the same obligation (each submitted item used once).
Then say whether it is CORRECT: same change type, and for modified/added the new_text states the same obligation (same
subject, condition, constraint/value) as the expected new_text. Paraphrase is fine."""

JUDGE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["items"], "properties": {
    "items": {"type": "array", "items": {"type": "object", "additionalProperties": False,
        "required": ["expected_index", "submitted_index", "correct", "note"],
        "properties": {"expected_index": {"type": "integer"}, "submitted_index": {"type": ["integer", "null"]},
                       "correct": {"type": "boolean"}, "note": {"type": "string"}}}}}}


def judge_subgoal(sub: dict, exp: dict, run_dir: Path) -> dict:
    fmt = lambda i, c: f"[{i}] {c['change']} {c.get('requirement_id')} §{(c.get('source') or {}).get('paragraph', '-')}: {c.get('new_text') or ''}"
    user = [{"type": "text", "text": "## EXPECTED\n" + "\n".join(fmt(i, c) for i, c in enumerate(exp["changes"]))
             + "\n\n## SUBMITTED\n" + "\n".join(fmt(i, c) for i, c in enumerate(sub["changes"]))}]
    res = call_json(JUDGE_SYSTEM, user, JUDGE_SCHEMA, max_tokens=16000, tag="SG_judge")
    seen, items = set(), []
    for it in res["items"]:
        s = it["submitted_index"]
        if s is not None and (s in seen or not 0 <= s < len(sub["changes"])):
            s = None
        if s is not None:
            seen.add(s)
        items.append({**it, "submitted_index": s, "correct": it["correct"] and s is not None})
    n_e, found = len(exp["changes"]), sum(i["submitted_index"] is not None for i in items)
    correct = sum(i["correct"] for i in items)
    summary = {"expected": n_e, "submitted": len(sub["changes"]), "flag_recall": round(found / n_e, 3),
               "accuracy_of_found": round(correct / found, 3) if found else 0, "correct_recall": round(correct / n_e, 3),
               "precision": round(len(seen) / len(sub["changes"]), 3) if sub["changes"] else 0}
    (run_dir / "judge.json").write_text(json.dumps({"summary": summary, "items": items}, indent=1, ensure_ascii=False))
    return {"summary": summary, "items": items}
