"""Three-stage, product-specific regulatory requirements pipeline.

Dependency: pip install openai
Usage:
  export OPENAI_API_KEY=...
  python regulatory_pipeline_3stage.py --regulation sections.json --product product.json --out run.json

Inputs are JSON. Regulation may be a list of sections or {"sections": [...]};
product is a JSON object (e.g. merged spec, profile, mappings, open_options).
Adapt `call_json` and field names to the existing repository's LLM and chunk schema.
"""
import argparse
import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI

MODEL = os.getenv("REGULATORY_MODEL", "gpt-4.1-mini")

EXTRACT = """Extract ALL atomic normative obligations from this regulatory section, WITHOUT considering a product.
Include conditional obligations, tests, prohibitions, numeric limits, tables, and annexes present in the supplied text.
Preserve exact units, conditions, exceptions, device categories, and citations. Never invent source references.
Do not emit standalone obligations from headings or purely referential text; instead keep cross-references.
If a referenced paragraph/table is unavailable, flag it as unresolved (do not assume its contents).
Return JSON: {"obligations": [{"id": "...", "text": "...", "subject": "...", "conditions": [],
"source": {"regulation": "...", "version": "...", "paragraph": "...", "page": null},
"references": [], "unresolved_references": []}], "coverage_notes": []}.
IDs must be unique within this section. Return [] where appropriate."""

CLASSIFY = """For EACH supplied obligation, determine applicability to THIS product using the supplied profile.
Statuses: APPLICABLE (product evidence establishes scope), CONDITIONAL (an explicit open design option
triggers scope), NOT_APPLICABLE (product evidence establishes exclusion), UNKNOWN (insufficient data).
Include own product functions and relevant general installation obligations; exclude unrelated vehicle
functions/signals and other categories only with evidence. Absence of mention is NOT evidence of absence.
Never alter obligation text. Every input obligation must produce exactly one classification.
Return JSON: {"decisions": [{"obligation_id": "...", "status": "...", "reason": "...",
"product_evidence": [], "components": [], "conditions": [], "missing_information": []}]}.
"""

TRANSLATE = """For each APPLICABLE/CONDITIONAL obligation, produce a faithful atomic engineering 'shall'
requirement. Preserve numeric limits, conditions, exceptions, source references, and parent IDs.
Do not create thresholds, tests, designs, or facts not in the regulation. If verification method is not
prescribed, leave it null and record an open question. Never silently merge distinct obligations.
Return JSON: {"requirements": [{"id": "...", "obligation_id": "...", "text": "...",
"components": [], "conditions": [], "source": {}, "verification_method": null,
"open_questions": []}]}.
"""


def call_json(client: OpenAI, instruction: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Single adapter point: swap with the repository's existing model client."""
    response = client.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        temperature=0,
    )
    return json.loads(response.choices[0].message.content or "{}")


def run(sections: list[dict], product: dict, client: OpenAI) -> dict:
    records, decisions, requirements, issues = [], [], [], []
    for n, section in enumerate(sections):
        extracted = call_json(client, EXTRACT, {"section": section})
        obligations = extracted.get("obligations", [])
        # Prefix per-section IDs so merges cannot collide across regulation chunks.
        for i, obligation in enumerate(obligations):
            obligation["id"] = f"S{n:04d}-O{i:04d}"
        records.extend(obligations)
        issues.extend({"section": n, "note": x} for x in extracted.get("coverage_notes", []))
        if not obligations:
            continue

        classified = call_json(client, CLASSIFY, {"obligations": obligations, "product": product})
        batch = classified.get("decisions", [])
        by_id = {x["obligation_id"]: x for x in batch if "obligation_id" in x}
        for obligation in obligations:
            oid = obligation["id"]
            if oid not in by_id:
                # Never silently lose an obligation due to imperfect LLM JSON.
                by_id[oid] = {"obligation_id": oid, "status": "UNKNOWN",
                              "reason": "Classification missing", "missing_information": ["manual review"]}
            decision = by_id[oid]
            if decision.get("status") not in {"APPLICABLE", "CONDITIONAL", "NOT_APPLICABLE", "UNKNOWN"}:
                decision["status"] = "UNKNOWN"
                decision["reason"] = "Invalid classification status; review required"
            decisions.append(decision)

        selected = [o for o in obligations if by_id[o["id"]]["status"] in {"APPLICABLE", "CONDITIONAL"}]
        if selected:
            produced = call_json(client, TRANSLATE, {
                "obligations": selected,
                "decisions": [by_id[o["id"]] for o in selected],
                "product": product,
            })
            valid_ids = {o["id"] for o in selected}
            for item in produced.get("requirements", []):
                if item.get("obligation_id") not in valid_ids:
                    issues.append({"section": n, "note": "Requirement has unknown parent ID", "item": item})
                    continue
                item["id"] = f"ENG-{len(requirements)+1:05d}"
                requirements.append(item)
            covered = {r["obligation_id"] for r in requirements}
            for o in selected:
                if o["id"] not in covered:
                    issues.append({"section": n, "obligation_id": o["id"],
                                   "note": "Selected obligation has no generated requirement"})

    return {"obligations": records, "applicability": decisions,
            "requirements": requirements, "review_items": issues,
            "counts": {"obligations": len(records), "requirements": len(requirements),
                       "unknown": sum(d["status"] == "UNKNOWN" for d in decisions)}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regulation", required=True, type=Path, help="JSON section list or {sections: [...]} file")
    parser.add_argument("--product", required=True, type=Path, help="Product profile JSON")
    parser.add_argument("--out", default=Path("pipeline_result.json"), type=Path)
    args = parser.parse_args()
    regulation = json.loads(args.regulation.read_text(encoding="utf-8"))
    sections = regulation if isinstance(regulation, list) else regulation["sections"]
    product = json.loads(args.product.read_text(encoding="utf-8"))
    result = run(sections, product, OpenAI())
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result["counts"], indent=2))


if __name__ == "__main__":
    main()
