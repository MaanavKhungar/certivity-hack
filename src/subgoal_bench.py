"""Silver benchmark for the sub-goal: a deterministic expected answer for any base -> latest pair, independent of the
pipeline's alignment (order-preserving sequence diff instead of best-match fuzzy alignment, no model calls).

Old requirement set: every 'shall' sentence of the old version, verbatim (definitions and table rows excluded).
Expected changes, following what the dev expected file does:
  - a requirement whose sentence is still there (same place or moved), up to formatting   -> no change
  - its sentence is there with different words                                            -> modified, new_text = new sentence
  - its sentence is gone                                                                  -> removed
  - a new 'shall' sentence (or an extra copy of an old one), in a chapter the old set covers -> added
"""
from __future__ import annotations
import difflib, json, re
from collections import Counter
from pathlib import Path
from rapidfuzz import fuzz
from regparse import parse_pdf, paragraph_key
import subgoal as sg

PAIR = 60      # two sentences in the same changed block are the same provision from this similarity on
MOVED = 85     # a removed and an added sentence anywhere in the document are one moved+edited provision from here on


def clean(s: str) -> str:
    """Comparison form: drop extraction noise, then the subgoal normalisation."""
    s = s.replace("|", " ")
    s = re.sub(r"\s+\d+(?:\.\d+)+\.?\s*$", "", s)            # next paragraph's number glued to the end
    s = re.sub(r"(?<=[a-z\)])\d{1,2}(?=[\s,.;:]|$)", "", s)  # footnote markers "approval,3"
    s = re.sub(r"\*+|_+", "", s)
    s = re.sub(r"\s+([,.;:])", r"\1", s)
    for a, b in ((" seconds", " s"), (" second", " s"), (" minutes", " min"), (" hours", " h"), (" metres", " m")):
        s = s.replace(a, b)
    return sg.norm(s)


REF = re.compile(r"\b\d+(?:\.\d+)+\.?|\b\d{1,2}\b")
NORMATIVE = re.compile(r"\bshall\b|\bmust\b|\bis not required\b|\bare not required\b")


def _letters(s: str) -> str:
    """Letters and digits only, British spelling: hyphenation, lost spaces, punctuation and quotes do not count."""
    s = re.sub(r"iz(e|ed|es|ing|ation)", r"is\1", s)
    return re.sub(r"[^a-z0-9]", "", s)


def editorial(a: str, b: str) -> bool:
    """Same provision up to formatting, a heading merged in front, renumbered cross-references or footnote numbers."""
    la, lb = _letters(a), _letters(b)
    if la == lb or lb.endswith(la) or la.endswith(lb):
        return True
    return _letters(REF.sub("#", a)) == _letters(REF.sub("#", b))


def _rows(nodes, keep_defs: bool):
    out = []
    for n in nodes:
        if n.section_kind == "definition" and not keep_defs:
            continue
        lbl = n.label if n.doc == "main" else n.key.title()
        for s in sg.sentences(n.text):
            if len(s) > 25 and not s.startswith("|"):
                out.append({"key": n.key, "label": lbl, "page": n.page_start, "text": s, "c": clean(s),
                            "shall": bool(re.search(r"\bshall\b", s)), "kind": n.section_kind})
    return out


def requirements_old(rows) -> list[dict]:
    reqs = []
    for r in rows:
        if r["shall"]:
            r["id"] = f"R-{len(reqs) + 1:04d}"
            reqs.append({"id": r["id"], "text": r["text"], "source": {"regulation": None, "page": r["page"], "paragraph": r["label"]}})
    return reqs


def _diff_reason(a: str, b: str) -> str:
    return "; ".join(sg.diff_ops(a, b)) or "wording changed"


def build(old_pdf: Path, new_pdf: Path) -> tuple[list[dict], dict]:
    old, new = parse_pdf(old_pdf, "base"), parse_pdf(new_pdf, "latest")
    reg, v = old[0].reg, new[0].version
    O, N = _rows(old, False), _rows(new, False)
    reqs = requirements_old(O)
    for q in reqs:
        q["source"]["regulation"] = reg
    sm = difflib.SequenceMatcher(None, [r["c"] for r in O], [r["c"] for r in N], autojunk=False)
    status = {}                          # old row index -> ("same"|"mod", new row index) ; unpaired -> removed candidates
    new_used = set()
    for t, i1, i2, j1, j2 in sm.get_opcodes():
        if t == "equal":
            for k in range(i2 - i1):
                status[i1 + k] = ("same", j1 + k); new_used.add(j1 + k)
        elif t == "replace":             # pair inside the block, order-preserving, greedy by similarity
            j = j1
            for i in range(i1, i2):
                best = max(range(j, j2), key=lambda jj: fuzz.ratio(O[i]["c"], N[jj]["c"]), default=None)
                if best is not None and fuzz.ratio(O[i]["c"], N[best]["c"]) >= PAIR:
                    status[i] = ("same" if editorial(O[i]["c"], N[best]["c"]) else "mod", best); new_used.add(best); j = best + 1
    # moved: an unpaired old sentence identical to an unpaired new one elsewhere -> unchanged; similar -> modified
    free_new = [j for j in range(len(N)) if j not in new_used]
    by_c = {}
    for j in free_new:
        by_c.setdefault(_letters(N[j]["c"]), []).append(j)
    for i in range(len(O)):
        if i in status:
            continue
        if by_c.get(_letters(O[i]["c"])):
            j = by_c[_letters(O[i]["c"])].pop(0); status[i] = ("same", j); new_used.add(j)
    free_new = [j for j in range(len(N)) if j not in new_used]
    for i in range(len(O)):
        if i in status or not O[i]["shall"]:
            continue
        cand = max(free_new, key=lambda j: fuzz.ratio(O[i]["c"], N[j]["c"]), default=None)
        if cand is not None and fuzz.ratio(O[i]["c"], N[cand]["c"]) >= MOVED:
            status[i] = ("same" if editorial(O[i]["c"], N[cand]["c"]) else "mod", cand); new_used.add(cand); free_new.remove(cand)

    changes = []
    for i, r in enumerate(O):
        if not r["shall"]:
            continue
        st = status.get(i)
        if st and st[0] == "same":
            continue
        if st is None:
            changes.append({"requirement_id": r["id"], "change": "removed",
                            "reason": f"Paragraph {r['label']} of {reg} was amended in version {v}; the provision '{sg._short(r['text'], 100)}' no longer appears."})
            continue
        n = N[st[1]]
        where = n["label"] if n["key"] == r["key"] else f"{n['label']} (formerly {r['label']})"
        changes.append({"requirement_id": r["id"], "change": "modified", "new_text": n["text"][:2000],
                        "reason": f"Paragraph {where} of {reg} was amended in version {v}: {_diff_reason(r['text'], n['text'])}.",
                        "source": {"page": n["page"], "paragraph": n["label"]}})
    scope = {sg.chapter(r["key"]) for r in O if r["shall"]}
    scope |= {sg.chapter(N[st[1]]["key"]) for i, st in status.items() if O[i]["shall"]}
    free = sorted(set(range(len(N))) - new_used)
    lead = {}                            # list item row -> its lead-in sentence ("... shall be as follows:")
    for j in range(len(N)):
        if N[j]["text"].rstrip().endswith(":") and NORMATIVE.search(N[j]["text"]):
            k = j + 1
            while k < len(N) and N[k]["key"] == N[j]["key"] and re.match(r"^\(?[a-z0-9ivx]{1,4}\)", N[k]["text"]):
                lead[k] = j; k += 1
    def add(n, text, quote):
        changes.append({"requirement_id": None, "change": "added", "new_text": text[:2000],
                        "reason": f"Paragraph {n['label']} of {reg} is new in version {v}: '{sg._short(quote, 100)}'.",
                        "source": {"page": n["page"], "paragraph": n["label"]}})
    for j in free:
        n = N[j]
        if sg.chapter(n["key"]) not in scope:
            continue
        if j in lead:                                    # a new step of a procedure: lead-in + step
            add(n, N[lead[j]]["text"].rstrip(": ") + ": " + n["text"], n["text"])
        elif NORMATIVE.search(n["text"]) and not (n["text"].rstrip().endswith(":") and any(lead.get(k) == j for k in free)):
            add(n, n["text"], n["text"])
    return reqs, {"team_id": "silver", "case_id": f"subgoal-bench-{reg.replace(' ', '_')}", "changes": changes}


# ============================================================ scoring =========================================
def score(sub: dict, exp: dict, run_dir: Path, use_model: bool = True) -> dict:
    """Code part: id-level recall/precision of modified+removed flags and change-type agreement.
    Model part (judge_subgoal): one-to-one matching incl. added and new_text accuracy."""
    E = {c["requirement_id"]: c for c in exp["changes"] if c["requirement_id"]}
    S = {c["requirement_id"]: c for c in sub["changes"] if c.get("requirement_id")}
    hit = set(E) & set(S)
    out = {"exp_flags": len(E), "sub_flags": len(S), "flag_recall": round(len(hit) / len(E), 3) if E else None,
           "flag_precision": round(len(hit) / len(S), 3) if S else None,
           "type_agree": round(sum(E[k]["change"] == S[k]["change"] for k in hit) / len(hit), 3) if hit else None,
           "exp_added": sum(c["change"] == "added" for c in exp["changes"]), "sub_added": sum(c["change"] == "added" for c in sub["changes"])}
    if use_model and exp["changes"]:
        out["judge"] = sg.judge_subgoal(sub, exp, run_dir)["summary"]
    return out
