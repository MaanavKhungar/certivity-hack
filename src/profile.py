"""Product profile: deterministic parse of spec.md + optional Claude vision enrichment (generated from the notebook)."""
from __future__ import annotations
import base64, json, re
from pathlib import Path

MODEL = "claude-opus-5-5"

# ------------------------------------------------------------------ 1. deterministic ------------------------------
def parse_spec_md(md: str) -> dict:
    """Split a spec markdown into its '## ' sections and pull the fields we rely on."""
    sections, cur = {}, None
    for line in md.splitlines():
        if (m := re.match(r"^#+\s*(.+?)\s*$", line)):
            cur = m.group(1).strip().lower(); sections[cur] = []
        elif cur is not None:
            sections[cur].append(line)
    sec = {k: "\n".join(v).strip() for k, v in sections.items()}

    def bullets(text):  return [re.sub(r"^[-*]\s*", "", l).strip() for l in text.splitlines() if re.match(r"^\s*[-*]\s+", l)]
    def first(text):    return next((l.strip() for l in text.splitlines() if l.strip()), "")
    def table(text):
        rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in text.splitlines() if l.strip().startswith("|")]
        rows = [r for r in rows if not all(re.fullmatch(r"-+", c) for c in r)]
        return {r[0]: r[1] for r in rows[1:] if len(r) >= 2} if rows else {}

    product_line = first(sec.get("product type", ""))
    cat_line = first(sec.get("vehicle category", ""))
    title = first(md).lstrip("# ").strip()
    return {
        "product": re.split(r"\s+[—–-]\s+", product_line)[0].strip() or re.sub(r"^spec:\s*", "", title, flags=re.I),
        "product_description": product_line,
        "vehicle_category": (re.match(r"\s*([A-Z]\d?)\b", cat_line).group(1) if re.match(r"\s*([A-Z]\d?)\b", cat_line) else None),
        "vehicle_category_note": cat_line,
        "functions": bullets(sec.get("function", "")),
        "parts": [{"part_id": f"P-{i+1:02d}", "name": p} for i, p in enumerate(bullets(sec.get("main parts", "")))],
        "materials": table(sec.get("materials", "")),
        "spec_sections": sorted(sec),
    }


# ------------------------------------------------------------------ 2. model + vision ------------------------------
PROFILE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["product", "vehicle_category", "mounting", "parts", "has_electronics", "electronic_subassemblies",
                 "light_sources", "regulated_functions", "search_terms", "drawing_observations"],
    "properties": {
        "product": {"type": "string"},
        "vehicle_category": {"type": "string", "description": "UNECE category, e.g. M1, N1, L3"},
        "mounting": {"type": "string", "description": "where on the vehicle, e.g. 'rear exterior', 'front exterior'"},
        "parts": {"type": "array", "items": {"type": "object", "additionalProperties": False,
            "required": ["part_id", "name", "function", "regulated_function", "material", "evidence_spec", "evidence_drawing"],
            "properties": {
                "part_id": {"type": "string"},
                "name": {"type": "string"},
                "function": {"type": "string", "description": "what the part does, in engineering terms"},
                "regulated_function": {"type": ["string", "null"], "description": "the exact term UN Regulations use for this part/function (e.g. 'rear position lamp', 'stop lamp', 'retro-reflector Class IA', 'electrical/electronic sub-assembly (ESA)', 'front fog lamp', 'external projection'); null if none"},
                "material": {"type": ["string", "null"]},
                "evidence_spec": {"type": ["string", "null"], "description": "verbatim quote from the spec that supports this part, or null"},
                "evidence_drawing": {"type": ["string", "null"], "description": "what in the drawing shows this part, or null"}}}},
        "has_electronics": {"type": "boolean"},
        "electronic_subassemblies": {"type": "array", "items": {"type": "string"}},
        "light_sources": {"type": ["string", "null"], "description": "e.g. 'non-replaceable LED modules', 'replaceable filament lamp category P21W', or null"},
        "regulated_functions": {"type": "array", "items": {"type": "string"}, "description": "deduplicated list of all regulated_function values"},
        "search_terms": {"type": "array", "items": {"type": "string"}, "description": "terms to search the regulations for: part names, functions, and their synonyms as the regulations phrase them"},
        "drawing_observations": {"type": "array", "items": {"type": "string"}, "description": "what the exploded view shows, as short factual notes"},
    },
}

SYSTEM = """You are a vehicle type-approval engineer. You receive a short technical specification of a vehicle component and,
usually, an exploded-view drawing of it. Produce a product profile that a regulatory engineer would use to decide which
UN Regulations (UNECE, 1958 Agreement) and which of their sections apply to the component.

Rules:
- Use the spec as the primary source and the drawing to confirm, refine and add parts the spec does not name.
- Name regulated functions with the exact vocabulary the UN Regulations use (e.g. "rear position lamp", "stop lamp",
  "reversing lamp", "retro-reflector", "front fog lamp", "daytime running lamp", "direction indicator", "electrical/electronic
  sub-assembly (ESA)", "external projection", "bumper", "protective device"). If a part has no regulated function, say null.
- Any electronics (LED driver, PCB, sensors, control gear) make the component an ESA for electromagnetic compatibility purposes.
- Light sources: say whether they are replaceable (then the light-source regulations apply) or non-replaceable modules.
- Quote evidence verbatim from the spec; describe drawing evidence factually. Do not invent parts that neither source supports.
- Keep the vehicle category exactly as stated in the spec."""


def _image_block(path: Path) -> dict:
    media = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return {"type": "image", "source": {"type": "base64", "media_type": media,
                                        "data": base64.standard_b64encode(path.read_bytes()).decode()}}


def enrich_with_model(spec_md: str, deterministic: dict, image_paths: list[Path], client=None) -> dict:
    import anthropic
    client = client or anthropic.Anthropic()
    content = [_image_block(p) for p in image_paths]
    content.append({"type": "text", "text":
        "## Specification (markdown)\n\n" + spec_md +
        "\n\n## Fields already parsed from the specification (confirm or correct)\n\n" + json.dumps(deterministic, indent=1) +
        ("\n\nThe image(s) above are the exploded view(s) of the component." if image_paths else "\n\nNo drawing was provided.")})
    with client.messages.stream(
        model=MODEL, max_tokens=16000,
        system=SYSTEM,
        output_config={"effort": "high", "format": {"type": "json_schema", "schema": PROFILE_SCHEMA}},
        messages=[{"role": "user", "content": content}],
    ) as stream:
        msg = stream.get_final_message()
    if msg.stop_reason == "refusal":
        raise RuntimeError(f"model refused: {msg.stop_details}")
    text = next(b.text for b in msg.content if b.type == "text")
    out = json.loads(text)
    out["_model"] = msg.model
    out["_usage"] = {"input": msg.usage.input_tokens, "output": msg.usage.output_tokens}
    return out


# ------------------------------------------------------------------ 3. entry point ---------------------------------
def build_profile(case_dir: Path, use_model: bool = True, client=None) -> dict:
    case_dir = Path(case_dir)
    spec_path = next(iter(sorted(case_dir.glob("*.md"))), None)
    images = sorted(p for p in case_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    spec_md = spec_path.read_text() if spec_path else ""
    det = parse_spec_md(spec_md) if spec_md else {"product": None, "parts": [], "functions": [], "materials": {}}
    profile = {"case_id": case_dir.name, "spec_file": spec_path.name if spec_path else None,
               "image_files": [p.name for p in images], "deterministic": det}
    if use_model:
        profile["model"] = enrich_with_model(spec_md, det, images, client)
    return profile
