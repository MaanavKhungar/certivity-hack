"""Stage B: product (spec.md + exploded view) x regulation cards -> extended product profile.

    python src/extend_profile.py ../industrial-ai-hackathon/cases/dev-rear-lamp

Writes data/product/<case>.extended.json (next to the existing product.json):
  1. base profile: spec.md parsed + Claude vision on the exploded view (src/profile.py)
  2. screening: every regulation card at once -> likelihood that it applies, with the reason
  3. mapping: for each regulation at or above --min-likelihood, the full card -> product parts mapped to the
     regulation's defined terms and classes, relevant / not relevant sections, and assumptions
A regulation is included when likelihood >= --threshold (0.3 by default: the F2 break-even, see README of the run).
"""
from __future__ import annotations
import argparse, json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reqagent import MODEL, REPO, load_env
from regcards import CARDS
import profile as prof

SCREEN_SYSTEM = """You are a vehicle type-approval engineer. A manufacturer gives you a product (a vehicle component or
assembly: its specification, an exploded-view drawing and a parsed profile). For every UN Regulation card you receive,
estimate the probability that the regulation sets requirements this product must fulfil, i.e. that a regulatory expert
listing the product's engineering requirements would cite it.

How regulations bind a component:
- directly, when the regulation approves this kind of device or component;
- through installation: a vehicle-installation regulation binds the component when it sets requirements on the
  position, visibility, orientation, switching or other features of devices of this kind, because the component must
  be designed so that it can be installed compliantly;
- through technology: e.g. electronics make it an electrical/electronic sub-assembly, a sensor makes it part of a
  system, a material or a mass makes it subject to a protection regulation.

Calibrate the probability. Use the regulation's scope, relations and applicability hints. When a newer regulation
replaces an older one for the same device (the cards' relations say so), the older one is unlikely to be cited.
Vehicle-level regulations about behaviour of the whole vehicle (e.g. emissions, braking performance) are unlikely
unless the product is part of that system. Return one entry for every card."""

SCREEN_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["product_summary", "regulations"],
    "properties": {
        "product_summary": {"type": "string"},
        "regulations": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["regulation", "likelihood", "binds_through", "reason"],
            "properties": {
                "regulation": {"type": "string"},
                "likelihood": {"type": "number", "description": "0.0 to 1.0"},
                "binds_through": {"type": "string", "enum": ["direct", "installation", "technology", "none"]},
                "reason": {"type": "string", "description": "one or two sentences, citing the product feature and the scope"}}}}},
}

MAP_SYSTEM = """You are a vehicle type-approval engineer. You receive a product (specification, exploded view, parsed
profile) and the reference card of one UN Regulation that may apply to it. Work out how the regulation applies:
- map each part or function of the product to the regulation's own defined terms, and give the class, category or
  variant it falls under, with evidence from the spec or the drawing;
- when the inputs do not settle a feature that changes which requirements apply (e.g. replaceable or non-replaceable
  light source, watertight or not, steady or variable intensity, with or without a logo), do NOT pick one: list it
  under open_options with every plausible option. The engineer will check which one holds, and the requirements for
  every plausible option must be kept;
- list the sections of the regulation that set requirements for this product (and why). In not_relevant_sections
  list only sections excluded by facts the spec or drawing confirms (other devices, other classes, other vehicle
  categories) or administrative content; never exclude a section because of an open option."""

MAP_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["applies", "how_it_applies", "part_mapping", "open_options", "relevant_sections", "not_relevant_sections"],
    "properties": {
        "applies": {"type": "boolean"},
        "how_it_applies": {"type": "string"},
        "part_mapping": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["part", "defined_term", "class_or_variant", "confirmed", "evidence"],
            "properties": {"part": {"type": "string"}, "defined_term": {"type": "string"},
                           "class_or_variant": {"type": ["string", "null"]},
                           "confirmed": {"type": "boolean", "description": "true if the spec or drawing settles it"},
                           "evidence": {"type": "string"}}}},
        "open_options": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["feature", "options", "affects_sections"],
            "properties": {"feature": {"type": "string", "description": "e.g. 'light source replaceability'"},
                           "options": {"type": "array", "items": {"type": "string"}},
                           "affects_sections": {"type": "array", "items": {"type": "string"}}}}},
        "relevant_sections": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["label", "why"],
            "properties": {"label": {"type": "string"}, "why": {"type": "string"}}}},
        "not_relevant_sections": {"type": "array", "items": {"type": "string"}},
    },
}

COMPACT = ["regulation", "title", "subject_kind", "scope_summary", "vehicle_categories", "applies_to_components",
           "relation_to_other_regulations", "applicability_hints"]


def call(client, system, content, schema):
    with client.beta.messages.stream(
        model=MODEL, max_tokens=64000, system=system, thinking={"type": "adaptive"},
        output_config={"effort": "high", "format": {"type": "json_schema", "schema": schema}},
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        messages=[{"role": "user", "content": content}],
    ) as stream:
        msg = stream.get_final_message()
    if msg.stop_reason == "refusal":
        raise RuntimeError(f"refused: {msg.stop_details}")
    return json.loads(next(b.text for b in msg.content if b.type == "text"))


def product_blocks(case_dir: Path, base: dict) -> list[dict]:
    spec = next(iter(sorted(case_dir.glob("*.md"))), None)
    images = sorted(p for p in case_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    blocks = [prof._image_block(p) for p in images]
    blocks.append({"type": "text", "text":
        ("The image(s) above are the exploded view(s) of the product.\n\n" if images else "No drawing was provided.\n\n") +
        "## Specification\n" + (spec.read_text() if spec else "(none)") +
        "\n\n## Parsed product profile\n" + json.dumps(base, indent=1, ensure_ascii=False)})
    return blocks


def extend(case_dir: Path, threshold: float, min_likelihood: float, reuse_base: bool = True) -> dict:
    import anthropic
    load_env()
    client = anthropic.Anthropic()
    out_path = REPO / "data/product" / f"{case_dir.name}.extended.json"

    # 1. base profile (vision); reuse the one from a previous run if present
    old = json.loads(out_path.read_text()) if out_path.exists() else {}
    if reuse_base and old.get("base_profile", {}).get("model"):
        base = old["base_profile"]; print("base profile: reused")
    else:
        t = time.time(); base = prof.build_profile(case_dir, use_model=True, client=client)
        print(f"base profile: built with vision ({time.time()-t:.0f}s)")
    blocks = product_blocks(case_dir, base)

    # 2. screening against all cards
    cards = {c["regulation"]: c for c in (json.loads(p.read_text()) for p in sorted(CARDS.glob("*.json")))}
    compact = [{k: c[k] for k in COMPACT} for c in cards.values()]
    t = time.time()
    screen = call(client, SCREEN_SYSTEM, blocks + [{"type": "text", "text":
                  f"## Regulation cards ({len(compact)})\n" + json.dumps(compact, indent=1, ensure_ascii=False)}], SCREEN_SCHEMA)
    regs = {r["regulation"]: r for r in screen["regulations"] if r["regulation"] in cards}
    print(f"screening: {len(regs)}/{len(cards)} regulations scored ({time.time()-t:.0f}s)")

    # 3. detailed mapping for the candidates
    cand = [r for r in regs.values() if r["likelihood"] >= min_likelihood]

    def one(r):
        card = cards[r["regulation"]]
        m = call(client, MAP_SYSTEM, blocks + [{"type": "text", "text":
                 "## Regulation card\n" + json.dumps(card, indent=1, ensure_ascii=False)}], MAP_SCHEMA)
        return r["regulation"], m

    with ThreadPoolExecutor(8) as pool:
        maps = dict(pool.map(one, cand))

    entries = []
    for name, r in sorted(regs.items(), key=lambda kv: -kv[1]["likelihood"]):
        e = {"regulation": name, "version": cards[name]["version"], "likelihood": r["likelihood"],
             "include": r["likelihood"] >= threshold, "binds_through": r["binds_through"], "reason": r["reason"]}
        if name in maps:
            e["mapping"] = maps[name]
        entries.append(e)
    ext = {"case_id": case_dir.name, "threshold": threshold, "product_summary": screen["product_summary"],
           "base_profile": base, "regulations": entries}
    out_path.write_text(json.dumps(ext, indent=1, ensure_ascii=False))
    print(f"\n{'regulation':<11}{'p':>5}  incl  through        reason")
    for e in entries:
        if e["likelihood"] >= 0.05:
            print(f"{e['regulation']:<11}{e['likelihood']:>5.2f}  {'yes ' if e['include'] else ' -  '}  {e['binds_through']:<13}  {e['reason'][:110]}")
    print(f"\n-> {out_path}")
    return ext


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("case_dir")
    ap.add_argument("--threshold", type=float, default=0.3, help="include a regulation at or above this likelihood")
    ap.add_argument("--min-likelihood", type=float, default=0.15, help="build the detailed mapping at or above this")
    ap.add_argument("--rebuild-base", action="store_true", help="re-run the vision profile even if one is saved")
    a = ap.parse_args(argv)
    extend(Path(a.case_dir), a.threshold, a.min_likelihood, reuse_base=not a.rebuild_base)


if __name__ == "__main__":
    main()
