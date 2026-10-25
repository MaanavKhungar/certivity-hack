"""Stage D on full-PDF C1 output (c1full): which PARAGRAPHS an expert takes requirements from, for this product.

    python src/dpara.py ../industrial-ai-hackathon/cases/dev-rear-lamp runs/c1full-v3-R10 runs/c1full-v3-R148 ... \
        --selection runs/rearlamp-v7b/selection.json --profile data/product/dev-rear-lamp.profile.json --out runs/dpara-v1

Why paragraphs: on the rear lamp, where the expert takes a paragraph our C1 writes about as many requirements as the
expert (192 vs 190); all the noise came from paragraphs the expert takes nothing from. So D decides per paragraph and
keeps every candidate of an included paragraph.

G   (code) B's section gate: keep candidates whose source paragraph lies in a section stage B selected for the
    regulation, administrative sections excluded (as stage T does). Uses our own stage B, no labels.
P   (model, one call per regulation) per source paragraph: its text, the candidates written from it, and the product
    inputs -> include probability + category, under a generic selection policy (no paragraph numbers or case wording).
Eval (labels, evaluation only) the judge matches the UNFILTERED candidate list once; every gate / threshold is then
    scored offline from those labels, per regulation, and the chosen list is judged again for confirmation.
"""
from __future__ import annotations
import argparse, json, re, sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import judge
import pipeline as rp
import reqagent
from regparse import Corpus, paragraph_key

CATEGORIES = ["product_requirement", "test_severity_or_criterion", "marking", "installation_of_product_function",
              "general_for_device_kind", "other_device_or_function", "special_configuration_not_indicated",
              "measurement_or_test_procedure", "vehicle_level_for_component", "administrative_or_cop",
              "framework_or_summary", "alternative_or_optional_procedure"]

SYSTEM = """You select, paragraph by paragraph, which paragraphs of ONE UN Regulation an expert engineer takes
requirements from for a SPECIFIC product. The candidate requirements were extracted from the whole regulation without
looking at the product, so most paragraphs do not belong in the product's list. The expert's list is selective: in the
relevant sections they take roughly one paragraph in four or five that contain "shall".

Include a paragraph (high probability) when it states, for this product:
- design, construction or performance requirements (photometric, colorimetric, electrical, mechanical, durability) of
  a device type or function the product IS, in its standard configuration;
- the markings the product must bear;
- the stress or severity of a prescribed test the product must withstand, or its pass criterion;
- in a regulation about installation on vehicles: the number, position, visibility, orientation, switching and tell-tale
  provisions for the functions the product provides;
- general provisions that apply to every device of the product's kind, including options the regulation treats as part
  of every such device's general requirements.

Exclude a paragraph (low probability) when it is:
- about another device type, function, class or category than the product's, even inside a relevant section;
- about a special configuration the product inputs do not indicate: assume the standard configuration (steady output,
  a single self-contained unit, fixed mounting, no alternative or substitute sources, no additional signal uses);
  variable or adaptive output, interdependent or multi-part systems, units approved as pairs, movable or concealed
  mounting, sequential activation, additional emitters and other optional configurations are excluded unless the
  inputs point to them;
- how a measurement or test is carried out (method, set-up, equipment, detectors, distances, operating or loading
  conditions of the sample, test plans, sample handling), as opposed to the level, severity or criterion;
- a vehicle-level obligation when the product is a component and the regulation also has component-level paragraphs;
- conformity of production, approval administration, communication, arrangement or examples of approval marks,
  transitional provisions, obligations of the applicant, the authority or the technical service;
- a framework sentence that only says other paragraphs or annexes apply, or a summary of other paragraphs;
- an alternative procedure at the request of the manufacturer, or a procedure for a class or option the product does
  not have.

Give every paragraph id: include (probability 0-1 that the expert takes requirements from it), category, and a short
reason. Judge each paragraph by what it says, not by how many candidates it has."""

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["paragraphs"], "properties": {
    "paragraphs": {"type": "array", "items": {"type": "object", "additionalProperties": False,
        "required": ["id", "include", "category", "reason"],
        "properties": {"id": {"type": "string"}, "include": {"type": "number"},
                       "category": {"type": "string", "enum": CATEGORIES}, "reason": {"type": "string"}}}}}}


def para_key(r) -> str:
    return r["source_node_ids"][0]


def section_gate(C: Corpus, reg: str, b: dict) -> set:
    """Node labels ("5.2.1" / "Annex 5 1.2") inside the sections stage B selected, administrative sections excluded."""
    keep = set()
    for s in b.get("selected_sections", []):
        ns = rp.section_nodes(C, reg, s["key"])
        if ns and (ns[0].section_kind or "") == "administrative":
            continue
        keep |= {n.label if n.doc == "main" else f"{n.doc} {n.label}" for n in ns}
    return keep


def label_of_q(q: str) -> str:
    return q.split("|", 2)[2].split("#")[0]


def score_paragraphs(reg, reqs, C, A, b, profile) -> dict:
    nodes = {(n.label if n.doc == "main" else f"{n.doc} {n.label}"): n for n in C.by_reg[(reg, "base")]}
    groups = defaultdict(list)
    for r in reqs:
        groups[para_key(r)].append(r)
    ids = {q: f"P{i + 1:03d}" for i, q in enumerate(groups)}
    lines = []
    for q, rs in groups.items():
        n = nodes.get(label_of_q(q))
        head = f"\n### {ids[q]} §{label_of_q(q)}" + (f" ({n.heading})" if n and n.heading else "") + \
               (f" [section kind: {n.section_kind}]" if n and n.section_kind else "")
        text = (n.text[:700] + (" …" if len(n.text) > 700 else "")) if n else "(text not found)"
        lines.append(head + "\nParagraph text: " + text.replace("\n", " ") + "\nCandidates written from it:\n" +
                     "\n".join(f"- {r['text'][:300]}" for r in rs))
    user = ("## Product\n" + A["product_summary"] + "\n\n## Product profile\n" + json.dumps(profile, indent=1, ensure_ascii=False) +
            "\n\n## How this regulation applies (earlier analysis)\n" +
            json.dumps({k: b[k] for k in ("how_it_applies", "part_mapping", "open_options")}, indent=1, ensure_ascii=False) +
            f"\n\n## Paragraphs of {reg} ({len(groups)})\n" + "\n".join(lines) + "\n\nAnswer for every paragraph id.")
    res = rp.call_json(SYSTEM, [{"type": "text", "text": user}], SCHEMA, max_tokens=64000, tag=f"DPARA_{reg.replace(' ', '_')}")
    back = {v: k for k, v in ids.items()}
    out = {back[p["id"]]: p for p in res["paragraphs"] if p["id"] in back}
    for q in groups:
        out.setdefault(q, {"id": ids[q], "include": 0.5, "category": "product_requirement", "reason": "not scored"})
    return out


def submission(reqs, case_id, team="team-05") -> dict:
    return {"team_id": team, "case_id": case_id,
            "requirements": [{"id": f"R-{i + 1:04d}", "text": r["text"],
                              "source": {k: r["source"][k] for k in ("regulation", "version", "page", "paragraph")}}
                             for i, r in enumerate(reqs)]}


def metrics(kept_ids, good, n_exp):
    m = len(kept_ids & good); n = len(kept_ids)
    P, R = (m / n if n else 0), m / n_exp
    return n, m, P, R, (5 * P * R / (4 * P + R) if P + R else 0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("case_dir")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--selection", default=str(reqagent.REPO / "runs/rearlamp-v7b/selection.json"))
    ap.add_argument("--profile", default=str(reqagent.REPO / "data/product/dev-rear-lamp.profile.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.5, help="paragraph include probability for the confirmed list")
    a = ap.parse_args(argv)
    reqagent.load_env()
    import bbpipe
    bbpipe.load_env()
    case, out = Path(a.case_dir).resolve(), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sel = json.loads(Path(a.selection).read_text())
    A, B = sel["stageA"], sel["stageB"]
    base = json.loads(Path(a.profile).read_text())
    profile = base.get("model") or base.get("deterministic") or base
    C = Corpus(reqagent.REPO / "data/pdf")
    by_reg = {}
    for d in a.runs:
        m = json.loads((Path(d) / "extraction_manifest.json").read_text())
        by_reg[m["regulation"]] = json.loads((Path(d) / "requirements_with_context.json").read_text())["requirements"]
    allr = [r for rs in by_reg.values() for r in rs]
    exp = json.loads((case / "expected.json").read_text())
    n_exp = len(exp["requirements"])

    # labels: judge the unfiltered list once (evaluation only)
    full = submission(allr, case.name)
    (out / "judge_full").mkdir(exist_ok=True)
    Jf = judge.judge(full, exp, out / "judge_full")
    rid2c = {s["id"]: r["id"] for s, r in zip(full["requirements"], allr)}
    good = {rid2c[x["submitted_id"]] for v in Jf["per_regulation"].values() for x in v["matches"]}
    print(f"unfiltered: {len(allr)} candidates, {len(good)} matched of {n_exp}")

    gate = {reg: section_gate(C, reg, B[reg]) for reg in by_reg}
    in_gate = {r["id"] for r in allr if label_of_q(para_key(r)) in gate[r["regulation"]]}
    with ThreadPoolExecutor(5) as pool:
        parts = dict(zip(by_reg, pool.map(lambda reg: score_paragraphs(reg, by_reg[reg], C, A, B[reg], profile), by_reg)))
    pscore = {r["id"]: parts[r["regulation"]][para_key(r)] for r in allr}
    (out / "paragraph_scores.json").write_text(json.dumps({reg: p for reg, p in parts.items()}, indent=1, ensure_ascii=False))

    rows = []
    def row(name, ids):
        rows.append((name, *metrics(ids, good, n_exp)))
    allids = {r["id"] for r in allr}
    row("unfiltered", allids)
    row("G: B section gate", in_gate)
    for t in (0.3, 0.4, 0.5, 0.6, 0.7):
        row(f"P >= {t}", {i for i in allids if pscore[i]["include"] >= t})
        row(f"G + P >= {t}", {i for i in in_gate if pscore[i]["include"] >= t})
    lines = ["| list | submitted | matched | precision | recall | F2 |", "|---|---|---|---|---|---|"]
    lines += [f"| {n} | {s} | {m} | {p:.3f} | {r:.3f} | {f:.3f} |" for n, s, m, p, r, f in rows]
    # per regulation for the main variant
    chosen = {i for i in in_gate if pscore[i]["include"] >= a.threshold}
    lines += ["", f"Per regulation, G + P >= {a.threshold} (offline labels):", "| regulation | submitted | matched | expected | precision | recall |", "|---|---|---|---|---|---|"]
    for reg, rs in by_reg.items():
        ids = {r["id"] for r in rs} & chosen
        e = sum(x["source"]["regulation"] == reg for x in exp["requirements"])
        m = len(ids & good)
        lines.append(f"| {reg} | {len(ids)} | {m} | {e} | {m / max(len(ids), 1):.3f} | {m / max(e, 1):.3f} |")
    cat = Counter((pscore[i]["category"], i in good) for i in allids)
    lines += ["", "Categories (candidates matched / total):"] + \
             [f"- {c}: {cat[(c, True)]}/{cat[(c, True)] + cat[(c, False)]}" for c in CATEGORIES if cat[(c, True)] + cat[(c, False)]]
    # confirmation judge of the chosen list
    kept = [r for r in allr if r["id"] in chosen]
    sub = submission(kept, case.name)
    (out / "submission.json").write_text(json.dumps(sub, indent=1, ensure_ascii=False))
    (out / "judge_chosen").mkdir(exist_ok=True)
    Jc = judge.judge(sub, exp, out / "judge_chosen")["summary"]
    lines += ["", f"Confirmation judge, G + P >= {a.threshold}: submitted {Jc['submitted']}, matched {Jc['matched']}, "
                  f"P {Jc['precision']:.3f}, R {Jc['recall']:.3f}, F2 {Jc['f2']:.3f}"]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
