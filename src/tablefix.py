"""Repair the structure of tables in selected sections by checking them against the PDF page image.

PyMuPDF4LLM sometimes flattens a table: header cells split mid-word ("Immunit level" | "y test"), a header spanning
two rows loses its parent, or a cell absorbs its neighbour's condition. The numbers usually survive but their
associations do not. For every table node in the sections stage B selected, the page(s) are rendered and Claude
returns each markdown table rebuilt with full header paths, one value per cell.

Guards: the number of tables must match, and a repaired table must keep every number of the original (a repair may
add header words, never lose values); otherwise the original stays. Results are cached (pipeline.call_json) and
written to data/table_fixes/<reg>.json for audit and the demo.
"""
from __future__ import annotations
import base64, json, re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pipeline as rp

FIXES = Path(__file__).resolve().parent.parent / "data/table_fixes"
MAX_PAGES = 3

SYSTEM = """You check tables extracted from a UN Regulation PDF against images of the PDF pages. The extracted text shows
each table as a markdown grid, but the extraction can break structure: header cells split mid-word, a header spanning
several rows or columns losing its parent, a cell absorbing a neighbouring cell or a condition, rows run together.

For every markdown table in the extracted text, in order, return the table as it is on the page:
- one header row in which every column header spells out its full path, e.g. "Immunity test level / 12 V system";
- one data value per cell; when a cell carries a value with a condition (e.g. "C (for other ESA)"), keep both in that
  cell or split into rows so the condition stays attached to its value; repeat a spanned value in every row it covers;
- keep every number, unit, sign, footnote marker and symbol exactly as on the page; do not add data the page does not show;
- if the extracted table is already correct, return it unchanged with changed=false."""

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["tables"],
          "properties": {"tables": {"type": "array", "items": {
              "type": "object", "additionalProperties": False, "required": ["index", "changed", "markdown", "note"],
              "properties": {"index": {"type": "integer"}, "changed": {"type": "boolean"},
                             "markdown": {"type": "string"}, "note": {"type": "string"}}}}}}

NUM = re.compile(r"[-+±]?\d+(?:[.,]\d+)?")


def table_blocks(text: str) -> list[tuple[int, int]]:
    """(start, end) line index ranges of consecutive '|' lines."""
    lines, out, start = text.split("\n"), [], None
    for i, l in enumerate(lines + [""]):
        if l.startswith("|") and start is None:
            start = i
        elif not l.startswith("|") and start is not None:
            out.append((start, i)); start = None
    return out


def page_images(pdf: Path, first: int, last: int) -> list[dict]:
    import pymupdf
    doc = pymupdf.open(pdf)
    out = []
    for p in range(first, min(last, first + MAX_PAGES - 1) + 1):
        if 1 <= p <= doc.page_count:
            png = doc[p - 1].get_pixmap(dpi=130).tobytes("png")
            out.append({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                     "data": base64.standard_b64encode(png).decode()}})
    return out


def fix_node(reg: str, node) -> dict | None:
    blocks = table_blocks(node.text)
    if not blocks:
        return None
    slug = reg.replace(" ", "_")
    pdf = rp.PKG / "regulations/pdf" / slug / "base" / f"{slug}_base.pdf"
    imgs = page_images(pdf, node.page_start, node.page_end)
    if not imgs:
        return None
    label = node.label if node.doc == "main" else f"{node.doc} {node.label}"
    user = imgs + [{"type": "text", "text": f"{reg}, paragraph {label}, pages {node.page_start}-{node.page_end}.\n\n"
                    f"Extracted text ({len(blocks)} table(s)):\n\n{node.text}"}]
    res = rp.call_json(SYSTEM, user, SCHEMA, max_tokens=32000, tag=f"T_{slug}_{label.replace(' ', '-')}")
    tabs = sorted(res["tables"], key=lambda t: t["index"])
    lines = node.text.split("\n")
    record = {"regulation": reg, "paragraph": label, "pages": [node.page_start, node.page_end], "tables": []}
    if len(tabs) != len(blocks):
        record["status"] = f"rejected: {len(tabs)} tables returned for {len(blocks)}"
        return record
    new_lines, prev = [], 0
    for (s, e), t in zip(blocks, tabs):
        old = "\n".join(lines[s:e])
        keep_old = not t["changed"] or not set(NUM.findall(old)) <= set(NUM.findall(t["markdown"]))
        record["tables"].append({"changed": t["changed"], "applied": not keep_old, "note": t["note"],
                                 "before": old, "after": t["markdown"]})
        new_lines += lines[prev:s] + (lines[s:e] if keep_old else t["markdown"].strip().split("\n"))
        prev = e
    new_lines += lines[prev:]
    record["status"] = "applied" if any(t["applied"] for t in record["tables"]) else "unchanged"
    record["text"] = "\n".join(new_lines)
    return record


def fix_selected(C, B, log) -> list[dict]:
    """Repair, in place, the text of table nodes in the sections stage B selected."""
    jobs = []
    for reg, b in B.items():
        if not b["include"]:
            continue
        keys = {(n.doc, n.label) for s in b["selected_sections"] for n in rp.section_nodes(C, reg, s["key"])}
        jobs += [(reg, n) for n in C.by_reg[(reg, "base")] if (n.doc, n.label) in keys and "\n|" in n.text]
    with ThreadPoolExecutor(8) as pool:
        records = list(pool.map(lambda j: (j, fix_node(*j)), jobs))
    out = []
    for (reg, n), rec in records:
        if rec is None:
            continue
        if rec["status"] == "applied":
            n.text = rec["text"]
        out.append(rec)
    FIXES.mkdir(parents=True, exist_ok=True)
    for reg in {r["regulation"] for r in out}:
        (FIXES / f"{reg.replace(' ', '_')}.json").write_text(
            json.dumps([{k: v for k, v in r.items() if k != "text"} for r in out if r["regulation"] == reg], indent=1, ensure_ascii=False))
    n_app = sum(r["status"] == "applied" for r in out)
    n_rej = sum(r["status"].startswith("rejected") for r in out)
    n_guard = sum(t["changed"] and not t["applied"] for r in out for t in r["tables"])
    log(f"table fix: {len(out)} table nodes checked, {n_app} repaired, {n_rej} rejected (table count), "
        f"{n_guard} table repairs refused by the number guard")
    return out
