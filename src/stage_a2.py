"""Stage A2: structured re-check of stage A's candidate regulations (which PDFs get read at all).

    python src/stage_a2.py <case_dir> --selection runs/<run>/selection.json --profile runs/<run>/profile.json [--votes 3]

Why: the expert lists regulations where the product is what the regulation is about (the device's own approval, its
installation, electronics on board) and leaves out regulations where the product is only an incidental part of a
vehicle-level topic (rear lamp: R26 external projections was proposed by stage A and was pure noise; front bumper: 4
of 11 cited regulations were noise). Stage A scores a likelihood from short cards and cannot tell these apart.

Per candidate regulation, one call answers (all candidates in one call, VOTES calls with shuffled order):
  approval_subject   device_or_component | vehicle | both
  product_role       is_the_regulated_device | installation_of_its_functions | main_carrier_of_the_topic |
                     electronics_on_board | incidental_part | not_related
  part_links         spec part or function -> the regulation's defined term / regulated item
  same_topic_as      another candidate covering the same topic (a GTR and its UN Regulation, an old regulation
                     replaced by a consolidated one), else null
  probability        that an expert lists requirements of this regulation for this product
Rule (code): include when the majority role is one of the first four, at least one part link exists, it is not the
weaker of a same-topic pair (UN Regulation preferred over GTR, consolidated over replaced), and the mean probability is
>= MIN_PROB. No expected answers are used.
"""
from __future__ import annotations
import argparse, json, random, re, sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline as rp
import reqagent
from regcards import CARDS
from regparse import Corpus

ROLES = ["is_the_regulated_device", "installation_of_its_functions", "main_carrier_of_the_topic",
         "electronics_on_board", "incidental_part", "not_related"]
GOOD = set(ROLES[:4])
MIN_PROB = 0.5
CARD_FIELDS = ["title", "subject_kind", "scope_summary", "scope_quote", "vehicle_categories", "applies_to_components",
               "relation_to_other_regulations", "applicability_hints"]

SYSTEM = """You decide which UN Regulations an expert type-approval engineer would list requirements from for ONE
specific product (a vehicle part or module, described by its specification, exploded view and parsed profile). You get
candidate regulations with their scope and a summary card. Answer for every candidate:

approval_subject: what the regulation approves: device_or_component, vehicle, or both.
product_role: how the product relates to the regulation's subject:
  is_the_regulated_device        the product (or one of its parts) IS a device/component the regulation approves
  installation_of_its_functions  the regulation governs installing devices of the kind the product contains or
                                 provides on the vehicle (position, visibility, switching of those functions)
  main_carrier_of_the_topic      a vehicle-level regulation whose topic is mainly delivered by this product, so the
                                 product's design decides compliance (e.g. the part that IS the protective structure,
                                 the impact zone or the regulated surface the regulation is written about)
  electronics_on_board           the regulation applies because the product carries electrical/electronic
                                 sub-assemblies (e.g. electromagnetic compatibility)
  incidental_part                a vehicle-level regulation in which the product is only one of many parts that
                                 merely have to be considered (e.g. it forms part of the external surface, or houses a
                                 sensor whose system function is approved at vehicle level)
  not_related                    the regulation does not concern this product
part_links: concrete links from parts or functions named in the product inputs to the regulation's defined terms or
  regulated items (e.g. "outer lens -> lens of a lamp"). Only links supported by the inputs; empty if none.
same_topic_as: if another candidate covers the same topic for new approvals (a global technical regulation and the UN
  Regulation implementing it, or an older regulation and the consolidated one replacing it), name that candidate,
  else null.
probability: that the expert lists requirements of this regulation for this product.
Judge from the regulation's scope and the product inputs. A regulation the product merely touches is not listed."""

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["regulations"], "properties": {
    "regulations": {"type": "array", "items": {"type": "object", "additionalProperties": False,
        "required": ["regulation", "approval_subject", "product_role", "part_links", "same_topic_as", "probability", "reason"],
        "properties": {"regulation": {"type": "string"},
                       "approval_subject": {"type": "string", "enum": ["device_or_component", "vehicle", "both"]},
                       "product_role": {"type": "string", "enum": ROLES},
                       "part_links": {"type": "array", "items": {"type": "string"}},
                       "same_topic_as": {"type": ["string", "null"]},
                       "probability": {"type": "number"}, "reason": {"type": "string"}}}}}}


def canon(name: str) -> str:
    """'UN GTR No. 9' -> 'UN GTR 9', 'UN R94 (version 5.1.0)' -> 'UN R94'."""
    m = re.search(r"GTR\D{0,6}(\d+)", name)
    if m:
        return f"UN GTR {m.group(1)}"
    m = re.search(r"\bR\s*(?:No\.?\s*)?(\d+[A-Z]*)", name)
    return f"UN R{m.group(1)}" if m else name


def preferred(a: str, b: str) -> str:
    """Of two same-topic regulations: a UN Regulation over a GTR, a consolidated one over one it replaces."""
    if "GTR" in a and "GTR" not in b:
        return b
    if "GTR" in b and "GTR" not in a:
        return a
    for new, olds in rp.SUPERSEDED.items():
        if a == new and b in olds:
            return a
        if b == new and a in olds:
            return b
    return a


def run(case: Path, sel: dict, profile_base: dict, votes: int = 3, log=print) -> dict:
    from extend_profile import product_blocks
    import types
    H = types.SimpleNamespace(canon=canon)
    blocks = product_blocks(case, profile_base)
    C = Corpus(reqagent.REPO / "data/pdf")
    cards = {H.canon(c["regulation"]): c for c in (json.loads(p.read_text()) for p in sorted(CARDS.glob("*.json")))}
    regs = [r for r in sel["stageA"]["candidates"] if (r, "base") in C.by_reg]
    info = {r: {"regulation": r, **{k: cards.get(r, {}).get(k) for k in CARD_FIELDS},
                "scope_text": rp.scope_text(C, r, 900)} for r in regs}

    def one(seed):
        order = regs[:]
        random.Random(seed).shuffle(order)
        user = blocks + [{"type": "text", "text": f"## Candidate regulations ({len(order)})\n" +
                          json.dumps([info[r] for r in order], indent=1, ensure_ascii=False) +
                          "\n\nAnswer for every candidate, using the regulation names exactly as given."}]
        res = rp.call_json(SYSTEM, user, SCHEMA, max_tokens=32000, tag=f"A2_vote{seed}")
        return {canon(x["regulation"]): x for x in res["regulations"]}

    with ThreadPoolExecutor(votes) as pool:
        V = list(pool.map(one, range(votes)))
    out = {}
    for r in regs:
        ans = [v[r] for v in V if r in v]
        role = Counter(a["product_role"] for a in ans).most_common(1)[0][0] if ans else "not_related"
        prob = sum(a["probability"] for a in ans) / len(ans) if ans else 0.0
        links = max((a["part_links"] for a in ans), key=len, default=[])
        same = Counter(H.canon(a["same_topic_as"]) for a in ans if a["same_topic_as"]).most_common(1)
        out[r] = {"role": role, "roles": [a["product_role"] for a in ans], "subject": Counter(a["approval_subject"] for a in ans).most_common(1)[0][0] if ans else None,
                  "probability": round(prob, 3), "part_links": links, "same_topic_as": same[0][0] if same and same[0][1] * 2 > len(ans) else None,
                  "stage_a": sel["stageA"]["candidates"][r]["likelihood"], "reason": ans[0]["reason"] if ans else ""}
    for r, o in out.items():
        why = []
        if o["role"] not in GOOD:
            why.append(f"role {o['role']}")
        if not o["part_links"]:
            why.append("no part link")
        if o["same_topic_as"] in out and preferred(r, o["same_topic_as"]) != r:
            why.append(f"same topic as {o['same_topic_as']} (preferred)")
        if o["probability"] < MIN_PROB:
            why.append(f"probability {o['probability']:.2f} < {MIN_PROB}")
        o["include"], o["excluded_because"] = not why, why
    for r, o in sorted(out.items(), key=lambda kv: -kv[1]["probability"]):
        log(f"{'INCLUDE' if o['include'] else 'drop   '} {r:10} A {o['stage_a']:.2f} -> A2 {o['probability']:.2f}  {o['role']:<30} "
            f"{'; '.join(o['excluded_because'])}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("case_dir")
    ap.add_argument("--selection", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--votes", type=int, default=3)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    import bbpipe
    bbpipe.load_env()
    sel = json.loads(Path(a.selection).read_text())
    out = run(Path(a.case_dir).resolve(), sel, json.loads(Path(a.profile).read_text()), a.votes)
    dest = Path(a.out) if a.out else Path(a.selection).with_name("stage_a2.json")
    dest.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"-> {dest}")


if __name__ == "__main__":
    main()
