"""Stage T (traversal): stage B's selected sections -> the full reading list for stage C.

    python src/traverse.py runs/rearlamp-v7b/selection.json --hops 3      # offline, no model calls

Start (hop 0): every section B selected for an included regulation (main-body section or annex, with all its
sub-paragraphs). Then, for up to N_HOPS rounds, the text of everything collected in the last round is scanned for
references and the targets are added as new items (hop k+1):
  same PDF   "paragraph 5.3.2.", "paragraphs 5.1. to 5.4.", "Annex 4", "Annex 6, Part 2", "Appendix 1 to Annex 9",
             "Table 3"; inside an annex a bare paragraph number means the annex's own paragraph unless the text says
             "of this Regulation"
  other PDF  "paragraph 5.5.2. of UN Regulation No. 48", "Annex 14 to UN Regulation No. 149", "according to the
             01 series of amendments to UN Regulation No. 3": the target is added (base version) and NOT scanned
             further; a regulation named without paragraph/annex, or not in our parsed corpus, becomes a hint only
Every item keeps how it got there (hop, via = the referring paragraph and the reference text) so recall and noise
can be traced back per item later. A paragraph reached twice is one item (smallest hop, all paths in via).
Limits (referenced material only; B's sections are always complete): one reference adds at most MAX_ITEM_NODES nodes / MAX_ITEM_CHARS characters (rest marked truncated); one
regulation's reading list grows by at most MAX_ADDED_CHARS through references (lower hops first, rest skipped).
"""
from __future__ import annotations
import argparse, json, re
from dataclasses import dataclass, field
from pathlib import Path

from regparse import Corpus, Node, paragraph_key

N_HOPS = 3                  # same-PDF reference hops; references into another PDF always stop after 1
MAX_ITEM_NODES = 150        # nodes one reference may add (a paragraph with its sub-paragraphs, or an annex)
MAX_ITEM_CHARS = 60_000
MAX_ADDED_CHARS = 250_000   # per regulation, everything added by references
EXTRACT_HOPS = 0            # mode rule: same-PDF items up to this hop are extracted, deeper ones are context
                            # (0: referenced sections are context only; on the rear lamp no expected requirement came
                            #  from them and their candidates were all noise)
CROSS_MODE = "context"      # mode of items from another PDF
CONTEXT_KINDS = {"administrative", "definition"}   # items of these kinds are context only, also when B selected them

LBL = r"\d+(?:\.\d+)+\.?|\d+\.(?!\d)"
REF_PARA = re.compile(rf"\bparagraphs?\s+((?:{LBL})(?:\s*(?:\([a-z0-9]+\)\s*)?(?:,|and|or|to|-|–)\s*(?:{LBL}))*)", re.I)
REF_ANNEX = re.compile(r"\bAnnex(?:es)?\s+(\d+[A-Z]?)(?:\s*,?\s*Part\s+(\d+[A-Z]?))?", re.I)
REF_APPX = re.compile(r"\bAppendix\s+(\d+[A-Z]?)(?:\s+(?:to|of)\s+Annex\s+(\d+[A-Z]?))?", re.I)
REF_TABLE = re.compile(r"\bTable\s+(A?\d+[A-Z]?(?:-\d+)?)\b")
REG_NO = r"(?:UN\s+)?(?:ECE\s+)?Regulations?\s+Nos?\.?\s*(\d{1,3})(?!\d)"
# text between a paragraph/annex reference and the regulation it belongs to
TAIL = re.compile(rf"^\s*\.?\s*(?:\([a-z0-9]+\)\s*)?,?\s*(?:of|in|to|under)\s+(?:(?:Annex\s+(\d+[A-Z]?))\s+(?:of|to)\s+)?"
                  rf"(?:the\s+[^;:]{{0,90}}?series\s+of\s+amendments\s+to\s+)?{REG_NO}", re.I)
SELF_TAIL = re.compile(r"^\s*\.?\s*(?:of|in)\s+this\s+(Regulation|annex|appendix|part)", re.I)
REG_MENTION = re.compile(REG_NO + r"((?:\s*,\s*\d{1,3}(?!\d))*(?:\s*(?:,|or|and)\s*\d{1,3}(?!\d))?)", re.I)


@dataclass
class Item:
    regulation: str
    key: str                    # section key as B writes it: "5", "6.4.2", "Annex 4", "Annex 6 Part 2"
    doc: str                    # "main" | "annex"
    origin: str                 # "selected" (stage B) | "reference" (found here)
    hop: int
    cross_pdf: bool
    kind: str                   # section_kind of the first node
    mode: str = "extract"       # "extract" | "context"
    via: list = field(default_factory=list)          # [{regulation, paragraph, ref}]
    nodes: list = field(default_factory=list)        # node keys, in document order
    pages: list = field(default_factory=list)        # [first, last]
    chars: int = 0
    truncated: bool = False
    title: str = ""

    def out(self):
        d = dict(self.__dict__)
        d["id"] = f"T-{self.regulation.replace('UN ', '').replace(' ', '')}-{self.key.replace(' ', '_')}"
        return d


def label_of(n: Node) -> str:
    return n.label if n.doc == "main" else f"{n.doc} {n.label}"


class Traverser:
    def __init__(self, C: Corpus, n_hops: int = N_HOPS, **limits):
        self.C, self.n_hops = C, n_hops
        self.max_item_nodes = limits.get("max_item_nodes", MAX_ITEM_NODES)
        self.max_item_chars = limits.get("max_item_chars", MAX_ITEM_CHARS)
        self.max_added_chars = limits.get("max_added_chars", MAX_ADDED_CHARS)
        self.extract_hops = limits.get("extract_hops", EXTRACT_HOPS)
        self.cross_mode = limits.get("cross_mode", CROSS_MODE)
        self.regs = {r for (r, role) in C.by_reg if role == "base"}
        self.items: dict[tuple, Item] = {}            # (reg, key) -> item
        self.owner: dict[tuple, tuple] = {}           # (reg, node key) -> item id it belongs to
        self.added_chars: dict[str, int] = {}
        self.hints, self.unresolved, self.skipped = [], [], []

    # ---------------------------------------------------------------------------------------- resolving --------
    def nodes(self, reg):
        return self.C.by_reg.get((reg, "base"), [])

    def subtree(self, reg: str, doc: str, label: str | None) -> list[Node]:
        """A paragraph with all its sub-paragraphs (label given), or a whole annex/part/appendix (label None)."""
        ns = self.nodes(reg)
        if label is None:
            d = paragraph_key(doc)
            return [n for n in ns if paragraph_key(n.doc) == d or paragraph_key(n.doc).startswith(d + " ")]
        k = paragraph_key(label)
        return [n for n in ns if n.doc == doc and (paragraph_key(n.label) == k or paragraph_key(n.label).startswith(k + "."))]

    def has_label(self, reg, doc, label) -> bool:
        k = paragraph_key(label)
        return any(n.doc == doc and paragraph_key(n.label) == k for n in self.nodes(reg))

    def para_targets(self, reg: str, here: Node, labels: list[str], tail: str, cross: str | None):
        """(reg, doc, label) for each paragraph label of one reference."""
        if cross:
            m = TAIL.match(tail)
            annex = m.group(1) if m else None
            doc = f"Annex {annex}" if annex else "main"
            return [(cross, doc, l) for l in labels]
        own_main = bool(SELF_TAIL.match(tail)) and SELF_TAIL.match(tail).group(1).lower() == "regulation"
        out = []
        for l in labels:
            if here.doc != "main" and not own_main and self.has_label(reg, here.doc, l):
                out.append((reg, here.doc, l))
            else:
                out.append((reg, "main", l))
        return out

    @staticmethod
    def expand_labels(s: str) -> list[str]:
        """'5.1. to 5.4.' -> 5.1, 5.2, 5.3, 5.4 (same parent); '5.1., 5.3. and 5.7.' -> three labels."""
        parts = re.split(r"\s*(,|and|or|to|-|–)\s*", re.sub(r"\([a-z0-9]+\)", "", s))
        labels, out, i = [], [], 0
        toks = [p for p in parts if p]
        while i < len(toks):
            t = toks[i].rstrip(".")
            if re.fullmatch(r"\d+(?:\.\d+)*", t):
                if out and i >= 2 and toks[i - 1] in ("to", "-", "–"):
                    a, b = out[-1].split("."), t.split(".")
                    if len(a) == len(b) and a[:-1] == b[:-1] and int(a[-1]) < int(b[-1]) <= int(a[-1]) + 30:
                        out += [".".join(a[:-1] + [str(x)]) for x in range(int(a[-1]) + 1, int(b[-1]) + 1)]
                        i += 1; continue
                out.append(t)
            i += 1
        return out

    def cross_reg(self, own: str, tail: str) -> str | None:
        m = TAIL.match(tail)
        if not m:
            return None
        r = f"UN R{m.group(2)}"
        return None if r == own else r

    def refs(self, reg: str, n: Node):
        """Yield (target reg, doc, label or None, reference text) for every reference in one node."""
        text = n.text
        for m in REF_PARA.finditer(text):
            tail = text[m.end():m.end() + 160]
            cross = self.cross_reg(reg, tail)
            for t in self.para_targets(reg, n, self.expand_labels(m.group(1)), tail, cross):
                yield (*t, m.group(0) + (tail[:TAIL.match(tail).end()] if cross else ""))
        for m in REF_ANNEX.finditer(text):
            tail = text[m.end():m.end() + 160]
            if re.match(r"^\s*\.?\s*(?:of|to)\s+(?:this|the)\s+", tail) and not TAIL.match(tail):
                if not SELF_TAIL.match(tail):
                    continue                              # "Annex 1 to the 1958 Agreement" and the like
            cross = self.cross_reg(reg, tail)
            doc = f"Annex {m.group(1)}" + (f" Part {m.group(2)}" if m.group(2) else "")
            yield (cross or reg, doc, None, m.group(0) + (tail[:TAIL.match(tail).end()] if cross else ""))
        for m in REF_APPX.finditer(text):
            annex = m.group(2) or (re.match(r"Annex (\d+[A-Z]?)", n.doc).group(1) if n.doc.startswith("Annex") else None)
            if annex:
                yield (reg, f"Annex {annex} Appendix {m.group(1)}", None, m.group(0))
        for m in REF_TABLE.finditer(text):
            for x in self.nodes(reg):
                if m.group(1) in x.tables:
                    yield (reg, x.doc, x.label, m.group(0)); break
        # regulations named without a paragraph or annex: hints only
        for m in REG_MENTION.finditer(text):
            for num in [m.group(1)] + re.findall(r"\d{1,3}", m.group(2) or ""):
                r = f"UN R{num}"
                if r != reg:
                    self.hints.append({"regulation": r, "in_corpus": r in self.regs, "from": reg,
                                       "paragraph": label_of(n), "text": text[max(0, m.start() - 80):m.end() + 40]})

    def note_unresolved(self, reg, key, via):
        """A reference whose target is not in the parsed corpus (or not parsed as its own node): one entry per target."""
        for u in self.unresolved:
            if (u["regulation"], u["key"]) == (reg, key):
                if via and len(u["via"]) < 5:
                    u["via"].append(via)
                return
        self.unresolved.append({"regulation": reg, "key": key, "in_corpus": reg in self.regs, "via": [via] if via else []})

    # ---------------------------------------------------------------------------------------- items -----------
    def add(self, reg, doc, label, *, origin, hop, cross, via) -> Item | None:
        ns = self.subtree(reg, doc, label)
        key = (label_of(ns[0]) if label else doc) if ns else (label if doc == "main" else f"{doc} {label or ''}".strip())
        if not ns:
            self.note_unresolved(reg, key, via)
            return None
        if origin == "selected":                          # as before: B's sections are read without their definitions
            ns = [n for n in ns if n.section_kind != "definition"] or ns
        new = [n for n in ns if (reg, n.key) not in self.owner]
        if not new:                                       # already read: record the extra path on its owner
            it = self.items.get(self.owner[(reg, ns[0].key)])
            if it and via and via not in it.via:
                it.via.append(via)
            return None
        truncated, kept, size = False, [], 0
        for n in new:                                     # limits apply to referenced material only, never to B's sections
            if origin == "reference" and (len(kept) >= self.max_item_nodes or size + len(n.text) > self.max_item_chars):
                truncated = True; break
            kept.append(n); size += len(n.text)
        if origin == "reference":
            if self.added_chars.get(reg, 0) + size > self.max_added_chars:
                self.skipped.append({"regulation": reg, "key": key, "hop": hop, "chars": size, "via": via})
                return None
            self.added_chars[reg] = self.added_chars.get(reg, 0) + size
        kind = kept[0].section_kind or "requirement"
        if origin == "selected":
            mode = "context" if kind in CONTEXT_KINDS else "extract"
        elif cross:
            mode = self.cross_mode
        else:
            mode = "extract" if hop <= self.extract_hops and kind not in CONTEXT_KINDS else "context"
        it = Item(regulation=reg, key=key, doc="main" if doc == "main" else "annex", origin=origin, hop=hop,
                  cross_pdf=cross, kind=kind, mode=mode, via=[via] if via else [], nodes=[n.key for n in kept],
                  pages=[kept[0].page_start, kept[-1].page_end], chars=size, truncated=truncated,
                  title=kept[0].title or kept[0].doc_title or "")
        self.items[(reg, key)] = it
        for n in kept:
            self.owner[(reg, n.key)] = (reg, key)
        return it

    def run(self, B: dict) -> dict:
        frontier = []
        for reg, b in B.items():
            if not b.get("include"):
                continue
            for s in b["selected_sections"]:
                k = s["key"].strip()
                m = re.match(r"^(Annex \S+(?: (?:Part|Appendix) \S+)?)(?: (\S+))?$", k, re.I)
                doc, label = (m.group(1), m.group(2)) if m else ("main", k)
                it = self.add(reg, doc, label, origin="selected", hop=0, cross=False, via=None)
                if it:
                    frontier.append(it)
        for hop in range(1, self.n_hops + 1):
            nxt = []
            for it in frontier:
                if it.cross_pdf:
                    continue                              # never traverse inside another PDF
                for nk in it.nodes:
                    n = self.C.index[(it.regulation, "base", nk)]
                    for reg, doc, label, ref in self.refs(it.regulation, n):
                        cross = reg != it.regulation
                        if cross and reg not in self.regs:
                            self.note_unresolved(reg, label or doc, {"regulation": it.regulation, "paragraph": label_of(n), "ref": ref})
                            continue
                        new = self.add(reg, doc, label, origin="reference", hop=hop, cross=cross,
                                       via={"regulation": it.regulation, "paragraph": label_of(n), "ref": ref})
                        if new:
                            nxt.append(new)
            frontier = nxt
            if not frontier:
                break
        return self.result()

    def result(self) -> dict:
        items = sorted(self.items.values(), key=lambda i: (i.regulation, i.hop, i.cross_pdf, i.key))
        seen, hints = set(), []
        for h in self.hints:                              # one hint per (target, from paragraph)
            k = (h["regulation"], h["from"], h["paragraph"])
            if k not in seen:
                seen.add(k); hints.append(h)
        by_reg = {}
        for i in items:
            s = by_reg.setdefault(i.regulation, {"items": 0, "extract": 0, "context": 0, "nodes": 0, "chars": 0, "by_hop": {}})
            s["items"] += 1; s[i.mode] += 1; s["nodes"] += len(i.nodes); s["chars"] += i.chars
            s["by_hop"][i.hop] = s["by_hop"].get(i.hop, 0) + 1
        return {"params": {"n_hops": self.n_hops, "max_item_nodes": self.max_item_nodes, "max_item_chars": self.max_item_chars,
                           "max_added_chars": self.max_added_chars, "extract_hops": self.extract_hops, "cross_mode": self.cross_mode},
                "summary": by_reg, "items": [i.out() for i in items], "regulation_hints": hints,
                "unresolved": self.unresolved, "skipped": self.skipped}


def traverse(C: Corpus, B: dict, n_hops: int = N_HOPS, **limits) -> dict:
    return Traverser(C, n_hops, **limits).run(B)


def trace(T: dict, expected: dict) -> dict:
    """Where each expected requirement's paragraph sits in the reading list: which item, how it got there (selected /
    reference, hop, mode), or missed. Plus, per item, how many expected requirements it holds (0 = noise candidate)."""
    owner = {}
    for it in T["items"]:
        for nk in it["nodes"]:
            owner[(it["regulation"], nk)] = it
    per_item = {it["id"]: 0 for it in T["items"]}
    rows = []
    for e in expected["requirements"]:
        reg, k = e["source"]["regulation"], paragraph_key(re.sub(r"\.?\s*Table\s+\S+$", "", str(e["source"]["paragraph"])))
        it = owner.get((reg, k))
        while it is None and "." in k:                    # an expected label deeper than our nodes: walk up
            k = k.rsplit(".", 1)[0]; it = owner.get((reg, k))
        if it:
            per_item[it["id"]] += 1
        rows.append({"expected_id": e.get("id"), "regulation": reg, "paragraph": e["source"]["paragraph"],
                     "item": it and it["id"], "origin": it and it["origin"], "hop": it and it["hop"], "mode": it and it["mode"]})
    found = [r for r in rows if r["item"]]
    summary = {"expected": len(rows), "covered": len(found), "covered_extract": sum(r["mode"] == "extract" for r in found),
               "by_hop": {h: sum(r["hop"] == h for r in found) for h in sorted({r["hop"] for r in found})},
               "items_without_expected": sum(v == 0 for v in per_item.values()), "items": len(per_item)}
    return {"summary": summary, "per_item": per_item, "rows": rows}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("selection", help="a run's selection.json (stageB inside) or a bare stage B dict")
    ap.add_argument("--hops", type=int, default=N_HOPS)
    ap.add_argument("--out", help="default: traversal.json next to the selection")
    ap.add_argument("--expected", help="expected.json of the case: also write trace.json (coverage per item / hop)")
    a = ap.parse_args(argv)
    sel = json.loads(Path(a.selection).read_text())
    B = sel.get("stageB", sel)
    C = Corpus(Path(__file__).resolve().parent.parent / "data/pdf")
    T = traverse(C, B, a.hops)
    out = Path(a.out) if a.out else Path(a.selection).with_name("traversal.json")
    out.write_text(json.dumps(T, indent=1, ensure_ascii=False))
    for reg, s in T["summary"].items():
        print(f"{reg:<9} items {s['items']:>3} (extract {s['extract']}, context {s['context']})  nodes {s['nodes']:>4}  "
              f"chars {s['chars']:>7}  by hop {s['by_hop']}")
    print(f"hints {len(T['regulation_hints'])}, unresolved {len(T['unresolved'])}, skipped {len(T['skipped'])} -> {out}")
    if a.expected:
        tr = trace(T, json.loads(Path(a.expected).read_text()))
        out.with_name("trace.json").write_text(json.dumps(tr, indent=1, ensure_ascii=False))
        print("trace:", tr["summary"])


if __name__ == "__main__":
    main()
