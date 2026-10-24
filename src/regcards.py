"""Stage A: one applicability card per regulation (product-independent, cached in data/regcards/).

    python src/regcards.py            # all regulations in the index, skips cards that already exist
    python src/regcards.py "UN R48"   # one regulation (rebuilds it)

A card is built from the regulation's table of contents and its opening sections (scope, definitions,
classification), and says what the regulation applies to, its defined terms and classes, and which sections deal with
which subject. Stage B (extend_profile.py) matches a product against these cards.
"""
from __future__ import annotations
import argparse, json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reqagent import MODEL, PKG, REPO, full_label, load_env, load_nodes

CARDS = REPO / "data/regcards"
FRONT_CHARS = 60_000           # opening text sent per regulation (R48's definitions alone are ~43k)

SYSTEM = """You are a vehicle type-approval engineer building a reference card for one UN Regulation (UNECE, 1958
Agreement). The card is later used to decide, for any given vehicle component or product, whether this regulation
applies and which parts of it are relevant. You receive the regulation's table of contents and its opening sections.

Be factual and quote the regulation where asked. In particular:
- subject_kind: "component" if the regulation approves a device/component type (e.g. a lamp, a retro-reflector, a
  tyre); "vehicle_installation" if it approves vehicles with regard to how devices are installed on them; "vehicle_system"
  if it approves a vehicle system or vehicle behaviour (e.g. braking, emissions, EMC of the vehicle); "mixed" if it
  approves both vehicles and separate components (e.g. vehicles and their electrical/electronic sub-assemblies).
- relation_to_other_regulations: note if the text says it replaces, supersedes, is an alternative to, or refers to
  other UN Regulations (e.g. installation regulations it refers to, or older device regulations it replaces).
- defined_terms: the terms the regulation defines that describe products, parts, functions, classes or categories
  (skip pure procedural terms). Keep each definition to one sentence.
- sections: every top-level section and every annex, plus the second-level sections that address one specific
  device, function or class, each with the defined terms it is about."""

SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["regulation", "title", "subject_kind", "scope_summary", "scope_quote", "vehicle_categories",
                 "applies_to_components", "relation_to_other_regulations", "defined_terms", "classes_and_variants",
                 "sections", "applicability_hints"],
    "properties": {
        "regulation": {"type": "string"},
        "title": {"type": "string"},
        "subject_kind": {"type": "string", "enum": ["component", "vehicle_installation", "vehicle_system", "mixed"]},
        "scope_summary": {"type": "string", "description": "two sentences: what is approved and for which vehicles"},
        "scope_quote": {"type": "string", "description": "the key sentence(s) of the scope, verbatim"},
        "vehicle_categories": {"type": "array", "items": {"type": "string"}},
        "applies_to_components": {"type": "array", "items": {"type": "string"},
                                  "description": "components/devices/functions this regulation sets requirements for, in its own terms"},
        "relation_to_other_regulations": {"type": "array", "items": {"type": "string"}},
        "defined_terms": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["term", "label", "definition"],
            "properties": {"term": {"type": "string"}, "label": {"type": "string"}, "definition": {"type": "string"}}}},
        "classes_and_variants": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["name", "distinguished_by"],
            "properties": {"name": {"type": "string"}, "distinguished_by": {"type": "string"}}}},
        "sections": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["label", "title", "about"],
            "properties": {"label": {"type": "string", "description": "e.g. '6.10' or 'Annex 4' or 'Annex 6 Part 2'"},
                           "title": {"type": "string"},
                           "about": {"type": "array", "items": {"type": "string"}}}}},
        "applicability_hints": {"type": "array", "items": {"type": "string"},
                                "description": "non-obvious ways a component can fall under this regulation (e.g. 'any component with electronics is an ESA')"},
    },
}


def front_text(nodes: list[dict]) -> str:
    """Main-text sections up to and including the definitions / classification, capped at FRONT_CHARS."""
    out, size = [], 0
    for n in nodes:
        if n["doc"] != "main":
            break
        top = n["label"].split(".")[0]
        if top.isdigit() and int(top) > 4:
            break
        s = f"[p.{n['page_start']}] {full_label(n)} {n['text']}"
        if size + len(s) > FRONT_CHARS:
            out.append("[... truncated ...]"); break
        out.append(s); size += len(s)
    return "\n".join(out)


def full_toc(nodes: list[dict]) -> str:
    return "\n".join(f"p.{n['page_start']} {full_label(n)} {n['title'] or n['text'][:90]}"
                     for n in nodes if n["depth"] <= 2 and (n["title"] or n["depth"] == 1))


def build_card(client, reg: str, title: str) -> dict:
    nodes = load_nodes(reg)
    user = (f"# {reg} - {title}, version {nodes[0]['version']}\n\n## Table of contents\n{full_toc(nodes)}\n\n"
            f"## Opening sections\n{front_text(nodes)}")
    with client.beta.messages.stream(
        model=MODEL, max_tokens=64000, system=SYSTEM, thinking={"type": "adaptive"},
        output_config={"effort": "high", "format": {"type": "json_schema", "schema": SCHEMA}},
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        messages=[{"role": "user", "content": user}],
    ) as stream:
        msg = stream.get_final_message()
    if msg.stop_reason == "refusal":
        raise RuntimeError(f"{reg}: refused: {msg.stop_details}")
    card = json.loads(next(b.text for b in msg.content if b.type == "text"))
    card["version"] = nodes[0]["version"]
    return card


def main(argv=None):
    import anthropic
    ap = argparse.ArgumentParser()
    ap.add_argument("regulations", nargs="*")
    a = ap.parse_args(argv)
    load_env()
    client = anthropic.Anthropic()
    CARDS.mkdir(parents=True, exist_ok=True)
    index = {e["regulation"]: e["title"] for e in json.loads((PKG / "regulations/index.json").read_text())}
    todo = a.regulations or [r for r in index if not (CARDS / f"{r.replace(' ', '_')}.json").exists()]

    def one(reg):
        t = time.time()
        try:
            card = build_card(client, reg, index[reg])
        except Exception as e:
            print(f"  {reg}: FAILED {e!r}"); return
        (CARDS / f"{reg.replace(' ', '_')}.json").write_text(json.dumps(card, indent=1, ensure_ascii=False))
        print(f"  {reg:<10} {card['subject_kind']:<20} {len(card['defined_terms']):>3} terms "
              f"{len(card['sections']):>3} sections ({time.time()-t:.0f}s)")

    print(f"building {len(todo)} cards")
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(one, todo))


if __name__ == "__main__":
    main()
