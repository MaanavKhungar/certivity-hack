"""Parse a Certivity-rendered UN Regulation PDF into a flat list of paragraph nodes (generated from the notebook)."""
from __future__ import annotations
import json, re, warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
warnings.filterwarnings("ignore")
import pymupdf, pymupdf4llm

MD_CACHE = Path(__file__).resolve().parent / "md"   # per-file cache of the PyMuPDF4LLM markdown

HEADER_RE   = re.compile(r"^(UN (?:R\d+[A-Z]?|GTR \d+)), version (\S+) \| page (\d+) of (\d+)\s*$")
LABEL_RE    = re.compile(r"^(\d+(?:\.\d+)*)(\.)?\s+([A-Z(\"“'].*)$")      # "6.10.4.2. Text" / "5.7 Grouped"
ANNEX_RE    = re.compile(r"^Annex\s+(\d+[A-Z]?)\s*(?:[-–:]\s*)?(.*)$", re.I)
PART_RE     = re.compile(r"^(Part|Appendix)\s+(\d+[A-Z]?)\b\s*(?:[-–:]\s*)?(.*)$", re.I)
TABLE_RE    = re.compile(r"^Table\s+([A-Z]?\d+[A-Z]?(?:-\d+)?)\b")


def paragraph_key(label: str) -> str:
    """Same normalisation as tools/score_local.py so our keys match the scorer's."""
    text = str(label or "").casefold().replace("§", " ")
    text = re.sub(r"[,;:()\[\]]", " ", text)
    text = re.sub(r"(?<![0-9a-z])\.|\.(?![0-9a-z])", " ", text)
    text = " ".join(text.split())
    return re.sub(r"^(paragraphs?|para|par|section|clause)\s+", "", text)


@dataclass
class Node:
    reg: str; version: str; role: str
    doc: str            # "main" | "Annex 3" | "Annex 4 Part 1" | "Annex 9 Appendix 1"
    label: str          # "6.10.4.2"
    key: str            # normalised full address: "annex 4 part 1 1.2" / "6.10.4.2"
    depth: int
    parent: str | None
    page_start: int; page_end: int
    heading: str        # titles of the enclosing sections, outermost first ("Individual specifications > Rear position lamp")
    title: str | None   # this node's own title if it is a heading
    text: str
    tables: list[str] = field(default_factory=list)
    has_shall: bool = False


def full_label(doc: str, label: str) -> str:
    return label if doc == "main" else f"{doc} {label}"


def _clean_md_line(line: str) -> str:
    """Strip the markdown decoration PyMuPDF4LLM adds (heading marks, bold/italic at line start,
    <sup>/<br> tags) so a paragraph label is again at the very start of the line."""
    line = line.rstrip()
    if line.lstrip().startswith("|"):                                   # table row: keep the grid, tidy the cells
        line = re.sub(r"<sup>(.*?)</sup>", r"^\1", line)
        line = line.replace("<br>", " ").replace("**", "").replace("_", "")
        return re.sub(r"\s+", " ", line).strip()
    line = re.sub(r"^\s*#+\s*", "", line)
    line = re.sub(r"^\s*[-•*]\s+", "", line)                              # bullet that PyMuPDF4LLM put in front of a label
    line = re.sub(r"^\s*[*_]{1,3}(.*?)[*_]{1,3}(?=\s|$)", r"\1", line)
    line = re.sub(r"<sup>(.*?)</sup>", r"\1", line).replace("<br>", " ")
    return line.strip()


def page_markdown(pdf_path):
    """[(page_no, markdown)] for the file, from PyMuPDF4LLM, cached on disk (the conversion takes ~8 s per 100 pages)."""
    pdf_path = Path(pdf_path)
    MD_CACHE.mkdir(parents=True, exist_ok=True)
    cache = MD_CACHE / (pdf_path.parent.parent.name + "_" + pdf_path.parent.name + ".json")
    if cache.exists():
        return [tuple(x) for x in json.loads(cache.read_text())]
    chunks = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True, show_progress=False)
    out = [(i + 1, c["text"]) for i, c in enumerate(chunks)]
    cache.write_text(json.dumps(out, ensure_ascii=False))
    return out


def extract_pages(pdf_path):
    """-> (regulation, version, [(page_no, cleaned_lines_without_header), ...])"""
    reg = version = None; pages = []
    for pdf_page_no, md in page_markdown(pdf_path):
        lines, page_no = [], pdf_page_no
        for raw in md.splitlines():
            s = raw.strip()
            if (m := HEADER_RE.match(s)):
                reg, version, page_no = m.group(1), m.group(2), int(m.group(3)); continue
            s = _clean_md_line(raw)
            if s and s != "-----":
                lines.append(s)
        pages.append((page_no, lines))
    if reg is None:
        raise ValueError(f"{pdf_path}: no Certivity page header found")
    return reg, version, pages


def _is_annex_heading(line: str, first_on_page: bool):
    m = ANNEX_RE.match(line)
    if not m or len(line) > 170 or line.endswith("."):
        return None
    title = (m.group(2) or "").strip()
    if title and not re.match(r"^[A-Z(\"“]", title) and not title.lower().startswith("appendix"):
        return None                                   # "Annex 4 to this Regulation" -> a reference, not a heading
    if (not title or title.startswith("(")) and not first_on_page:
        return None                                   # bare "Annex 5" or "Annex 10 (values in" inside a table
    return f"Annex {m.group(1)}", title


def parse_pdf(pdf_path, role: str) -> list[Node]:
    reg, version, pages = extract_pages(pdf_path)
    nodes: list[Node] = []
    doc, annex = "main", None
    cur: Node | None = None
    buf: list[str] = []
    headings: dict[int, str] = {}                     # depth -> title within the current doc
    in_doc: set[str] = set()                          # labels already created in the current doc
    top_max = 0                                       # highest depth-1 number seen in the current doc

    def flush():
        nonlocal cur, buf
        if cur is None:
            return
        text = ""
        for s in (s.strip() for s in buf if s.strip()):          # table rows keep their own line
            text += ("\n" if (s.startswith("|") or text.endswith("|")) else " ") + s if text else s
        cur.text = text
        cur.has_shall = bool(re.search(r"\bshall\b", text))
        cur.tables = sorted({m.group(1) for s in buf if (m := TABLE_RE.match(s.strip()))})
        if cur.depth <= 3 and len(text) <= 90 and not text.endswith((".", ";", ":")) and not cur.has_shall:
            cur.title = text
            headings[cur.depth] = text
            for d in [d for d in headings if d > cur.depth]:
                del headings[d]
        cur.heading = " > ".join(headings[d] for d in sorted(headings) if d < cur.depth)
        nodes.append(cur); cur, buf = None, []

    def switch(new_doc: str, title: str):
        nonlocal doc, headings, in_doc, top_max
        flush(); doc = new_doc; headings = {0: f"{new_doc} {title}".strip()}; in_doc = set(); top_max = 0

    for page_no, lines in pages:
        first = True
        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            if (ah := _is_annex_heading(line, first)):
                annex = ah[0]
                am = re.match(r"^(.*?)\s*(?:[-–:]\s*)?(Appendix\s+\d+[A-Z]?)\s*(?:[-–:]\s*)?(.*)$", ah[1], re.I) if "appendix" in ah[1].lower() else None
                if am:   # "Annex 9 - Appendix 1 BCI test"
                    switch(f"{annex} {am.group(2).title()}", am.group(3))
                else:
                    switch(annex, ah[1])
                first = False; continue
            pm = PART_RE.match(line)
            if pm and annex and len(line) <= 170 and not line.endswith(".") and (not pm.group(3) or re.match(r"^[A-Z(\"“]", pm.group(3))):
                switch(f"{annex} {pm.group(1).title()} {pm.group(2)}", pm.group(3)); first = False; continue
            first = False
            lm = LABEL_RE.match(line)
            if lm:
                label, rest = lm.group(1), lm.group(3)
                parts = label.split("."); depth = len(parts)
                parent = ".".join(parts[:-1]) or None
                grand  = ".".join(parts[:-2]) or None
                ok = (depth == 1 and lm.group(2) == "." and int(label) in (top_max + 1, top_max + 2, 0, 1) and label not in in_doc) \
                     or (depth >= 2 and (parent in in_doc or (grand and grand in in_doc)) and label not in in_doc) \
                     or (cur is None and not nodes) \
                     or (not in_doc and lm.group(2) == ".")               # first label of a new numbering space, any depth
                if ok:
                    flush()
                    cur = Node(reg=reg, version=version, role=role, doc=doc, label=label,
                               key=paragraph_key(full_label(doc, label)), depth=depth, parent=parent,
                               page_start=page_no, page_end=page_no, heading="", title=None, text="")
                    buf = [rest]; in_doc.add(label)
                    if depth == 1: top_max = max(top_max, int(label))
                    continue
            if cur is None:                           # preamble / text before the first label of this doc
                cur = Node(reg=reg, version=version, role=role, doc=doc, label="0", key=paragraph_key(full_label(doc, "0")),
                           depth=1, parent=None, page_start=page_no, page_end=page_no, heading="", title=None, text="")
                buf = []
            buf.append(line); cur.page_end = page_no
    flush()
    return nodes


def save(nodes, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps([asdict(n) for n in nodes], ensure_ascii=False, indent=1))

def load(path):
    return [Node(**d) for d in json.loads(Path(path).read_text())]


class Corpus:
    """All parsed regulations in memory. find / toc / read_section are the future agent tools."""
    def __init__(self, parsed_dir):
        self.by_reg, self.index = {}, {}
        for p in sorted(Path(parsed_dir).glob("*.json")):
            nodes = load(p)
            if nodes:
                self.by_reg[(nodes[0].reg, nodes[0].role)] = nodes
                for n in nodes:
                    self.index.setdefault((n.reg, n.role, n.key), n)

    def find(self, reg, label, role="base"):
        key = paragraph_key(label)
        n = self.index.get((reg, role, key))
        if n is None:                                  # golden labels may append a table: "5.1.4.4.1. Table 3"
            n = self.index.get((reg, role, re.sub(r"\s+table\s+\S+$", "", key)))
        return n

    def toc(self, reg, role="base", max_depth=2):
        return [n for n in self.by_reg[(reg, role)] if n.depth <= max_depth and (n.title or n.depth == 1)]

    def read_section(self, reg, label, role="base"):
        key = paragraph_key(label)
        return [n for n in self.by_reg[(reg, role)] if n.key == key or n.key.startswith(key + ".")]
