"""Requirement extraction agent: product profile + one parsed regulation -> submission-format requirements.

    python src/reqagent.py "UN R150" --product data/product/product.json [--run-name r150-v1]

The regulation is split into sections; each section goes to Claude with the product profile and the regulation's
table of contents. The model returns obligations tied to node ids; pages and paragraph labels are filled in from the
parsed nodes, so every citation resolves to a real paragraph of the base version.
"""
from __future__ import annotations
import argparse, json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PKG = REPO.parent / "industrial-ai-hackathon"
MODEL = "claude-opus-5-5"
CHUNK_CHARS = 40_000          # split a section into its sub-sections above this size

# --------------------------------------------------------------------------------------------- prompt ------------
# Rules must stay generic: the held-out product is different from the dev case, so no paragraph numbers or
# answer-specific wording go in here.
SYSTEM = """You are a vehicle type-approval engineer. You read one section of a UN Regulation (UNECE, 1958 Agreement)
and list the engineering requirements that a specific product must fulfil, so that an engineer can design and test
against them.

You receive the product profile, the regulation's table of contents, and the paragraphs of one section.
The product profile may include "regulation_mapping": an earlier analysis of how this regulation applies to the
product (parts mapped to the regulation's defined terms and classes, relevant and not relevant sections, assumptions).
Use it to decide which classes, variants and devices the product is, and which sections bind it. It is guidance: a
paragraph in a section marked not relevant still yields a requirement when it clearly binds the product.
Its "open_options" are product features the inputs do not settle (e.g. replaceable or non-replaceable light source):
write the requirements for every listed option, each stating its condition ("In case of ..., ..."). Each paragraph
is shown as [node_id] p.<pages> <label> followed by its text. Tables appear as markdown grids inside their paragraph.

What counts as a requirement:
- An obligation on the product itself: its design, construction, materials, dimensions, photometric/colorimetric or
  other performance, durability, markings, and the results it must achieve in the prescribed tests.
- For prescribed tests, keep what the product must withstand and achieve: the exposure or stress applied to the
  product (e.g. the test atmosphere or solution it is exposed to, with its temperature and duration), the conditions
  that define what is measured (which surface or area counts, the illuminant and angles for a measurement), and the
  pass criteria after the test. An exposure stays a requirement even when the same paragraph also describes the
  apparatus: "the samples shall be exposed according to standard X using the parameters of Table Y for N hours" is
  the stress the product must withstand; write it with its standard, parameters (from the table) and duration.
- Design options with a condition ("if the device has / is marked / is combined with X, it shall ...") are
  requirements: the engineer decides whether the condition applies. Keep them.
- Approval marking: the content, placement, size, legibility and format of the marks on the product are requirements.
- Permissions, exemptions and relaxations that apply to the product are requirements too: "X may be ...", "X is
  permitted / allowed provided that ...", "notwithstanding paragraph N, ...", "the requirement of N does not apply
  if ...", a permitted variation or a reduced field. Write them with their condition ("X may ... provided that ...").
- Provisions for optional devices of the product's kind ("where optional lamps are installed, ...") are requirements:
  each condition they set (position, height, distance, angles) is its own requirement.

What does not count:
- Obligations on the approval authority, the Contracting Parties or the technical service; application paperwork,
  communication forms, numbering of approvals, transitional provisions, definitions, and the scope statement.
- How the laboratory runs a test: apparatus and instruments, measuring geometry and distances, stray-light handling,
  how many samples go to which test and in what order, and the technician's handling steps (wiping, drying, rotating,
  cleaning the sample). These are test-method details, not obligations an engineer designs to.
- Do not turn the description of a test step into an obligation ("the device shall withstand being wiped / immersed
  / exposed ..."). A test paragraph yields a requirement only where it states a condition of the kind listed above or
  a pass criterion.
- Summaries of other paragraphs (e.g. a table listing the test sequence, or a paragraph saying samples go through
  tests X, Y and Z): skip them when the detailed paragraphs state the same obligations.
- Alternative test procedures that apply only "at the request of the manufacturer".
- Conformity-of-production statistics: sampling plans, sample sizes, deviation thresholds per sample group, and the
  consequences for the approval. Keep only a limit on how far a production part's performance may deviate.
- Figures in annexes that show example layouts of approval marks.
- Requirements that only apply to product types, classes, colours or variants the product is not (use the product
  profile to decide; if the profile does not settle it, keep requirements for the variant most likely to apply).
- Recommendations ("should", "it is recommended") and descriptions without an obligation.

How to write them:
- One obligation per requirement. Each "shall" (or "must") clause is its own requirement, also when two clauses
  share a sentence joined by "and" or a semicolon. Write several requirements with the same node_id.
- Granularity, SPLIT into separate requirements when the values or conditions differ: each frequency band, pulse,
  test level or severity with its own value; each supply voltage (12 V / 24 V) or device category with its own value;
  each direction (horizontal angle / vertical angle); each different property constrained for the same object
  (e.g. the kind of an element and the number of elements); each separate indicator, tell-tale or signal; each
  classification or status criterion stated for a different test. A limit that is constant over one range and
  varies over another is two requirements.
- Granularity, KEEP TOGETHER in one requirement: several attributes of one property in one clause ("clearly legible
  and indelible", "not rough and without protuberances"); the checks of one test at several times or stages
  ("after 1 min and after 10 min"); an obligation together with its own exception or proviso stated in the same
  paragraph ("shall meet Table N, except that ... may be ..."); one obligation that covers both sides of an "and/or".
- Cite the paragraph in which the obligation itself is stated (the deepest one), not its parent heading or the
  paragraph that only announces a test.
- Stay close to the regulation's wording; make each requirement understandable on its own (name the subject instead
  of "it", keep conditions such as "when tested according to ...").
- When a paragraph sets a limit by reference to a table: write the paragraph's own obligation as one requirement
  (table null), then the table's values that apply to the product (table set):
  - if the table gives ONE limit of one quantity over a measuring grid (angles, measuring points, zones of one
    function), write ONE requirement listing all applicable values;
  - if its rows are DIFFERENT tests or test conditions (pulse types, test methods, frequency bands or severities
    each with their own level, duration or pass criterion), write ONE requirement PER ROW, each stating the row's
    test, its values and its criterion.
  Write any further "shall" sentence stated around the table separately.
- If the requirement comes from a table, set "table" to the table name (e.g. "3" or "A6-1"), else null.
- Cite the node_id of the paragraph the obligation is stated in. Never invent node ids.
- If nothing in the section applies to the product, return an empty list."""

SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["requirements"],
    "properties": {"requirements": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["node_id", "table", "text", "applies_because"],
        "properties": {
            "node_id": {"type": "string"},
            "table": {"type": ["string", "null"]},
            "text": {"type": "string"},
            "applies_because": {"type": "string", "description": "one short sentence: why this binds this product"},
        }}}},
}


# --------------------------------------------------------------------------------------------- data --------------
def load_env():
    env = REPO / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))
    if not os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("ANTHROPIC_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = os.environ["ANTHROPIC_KEY"]


def load_nodes(reg: str, role: str = "base") -> list[dict]:
    nodes = json.loads((REPO / "data/pdf" / f"{reg.replace(' ', '_')}_{role}.json").read_text())
    for i, n in enumerate(nodes):
        n["id"] = f"n{i}"
    return nodes


def full_label(n: dict) -> str:
    return n["label"] if n["doc"] == "main" else f"{n['doc']} {n['label']}"


def render(n: dict) -> str:
    pages = f"{n['page_start']}" if n["page_start"] == n["page_end"] else f"{n['page_start']}-{n['page_end']}"
    return f"[{n['id']}] p.{pages} {full_label(n)}\n{n['text']}"


def toc(nodes: list[dict]) -> str:
    return "\n".join(f"p.{n['page_start']} {full_label(n)} {n['title'] or n['text'][:80]}"
                     for n in nodes if n["depth"] <= 2 and (n["title"] or n["depth"] == 1))


def chunks(nodes: list[dict]) -> list[tuple[str, list[dict]]]:
    """Main text: one chunk per top-level section; annexes: one chunk per annex / part. Big chunks are split by
    their depth-2 sub-sections."""
    groups: dict[tuple, list[dict]] = {}
    for n in nodes:
        groups.setdefault((n["doc"], n["label"].split(".")[0] if n["doc"] == "main" else ""), []).append(n)
    out = []
    for (doc, top), ns in groups.items():
        name = doc if doc != "main" else f"§{top}"
        if sum(len(n["text"]) for n in ns) <= CHUNK_CHARS:
            out.append((name, ns)); continue
        sub: dict[str, list[dict]] = {}
        for n in ns:
            sub.setdefault(".".join(n["label"].split(".")[:2]), []).append(n)
        cur, size = [], 0
        for part in sub.values():                       # pack sub-sections up to the size limit
            s = sum(len(n["text"]) for n in part)
            if cur and size + s > CHUNK_CHARS:
                out.append((f"{name} ({cur[0]['label']}–{cur[-1]['label']})", cur)); cur, size = [], 0
            cur += part; size += s
        if cur:
            out.append((f"{name} ({cur[0]['label']}–{cur[-1]['label']})", cur))
    return out


# --------------------------------------------------------------------------------------------- model -------------
def ask(client, product: dict, reg: str, version: str, contents: str, section: str, ns: list[dict]) -> list[dict]:
    user = (f"## Product profile\n{json.dumps(product, indent=1, ensure_ascii=False)}\n\n"
            f"## Regulation\n{reg}, version {version}\n\n## Table of contents\n{contents}\n\n"
            f"## Section {section}\n\n" + "\n\n".join(render(n) for n in ns))
    for attempt in range(3):
        try:
            with client.beta.messages.stream(
                model=MODEL, max_tokens=64000, system=SYSTEM,
                thinking={"type": "adaptive"},
                output_config={"effort": "high", "format": {"type": "json_schema", "schema": SCHEMA}},
                betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                messages=[{"role": "user", "content": user}],
            ) as stream:
                msg = stream.get_final_message()
            if msg.stop_reason == "refusal":
                raise RuntimeError(f"refused: {msg.stop_details}")
            text = next(b.text for b in msg.content if b.type == "text")
            return json.loads(text)["requirements"], msg.usage
        except Exception as e:                          # retry transient failures, then give up on this section
            if attempt == 2:
                raise
            print(f"  {section}: {e!r}, retrying", file=sys.stderr); time.sleep(5 * (attempt + 1))


def extract(reg: str, product: dict, workers: int = 8, log=print) -> tuple[list[dict], list[dict]]:
    import anthropic
    load_env()
    client = anthropic.Anthropic()
    nodes = load_nodes(reg)
    by_id = {n["id"]: n for n in nodes}
    version, contents = nodes[0]["version"], toc(nodes)
    parts = chunks(nodes)
    log(f"{reg} {version}: {len(nodes)} nodes in {len(parts)} sections")

    def one(part):
        section, ns = part
        t = time.time()
        reqs, usage = ask(client, product, reg, version, contents, section, ns)
        log(f"  {section:<30} {len(ns):>4} nodes -> {len(reqs):>3} reqs  ({time.time()-t:.0f}s, "
            f"in {usage.input_tokens} out {usage.output_tokens})")
        return section, reqs

    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(one, parts))

    out, raw = [], []
    for section, reqs in results:
        for r in reqs:
            raw.append({"section": section, **r})
            n = by_id.get(r["node_id"])
            if n is None:
                log(f"  dropped (unknown node {r['node_id']}): {r['text'][:80]}"); continue
            para = full_label(n) + (f". Table {r['table']}" if r.get("table") else "")
            out.append({"id": f"R-{len(out)+1:04d}", "text": r["text"][:2000],
                        "source": {"regulation": reg, "version": version, "page": n["page_start"], "paragraph": para},
                        "applies_because": r["applies_because"]})
    return out, raw


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("regulation", help='e.g. "UN R150"')
    ap.add_argument("--product", default=str(REPO / "data/product/product.json"))
    ap.add_argument("--case", default="dev-rear-lamp")
    ap.add_argument("--team", default="team-05")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args(argv)

    profile = json.loads(Path(args.product).read_text())
    product = profile.get("model") or profile.get("deterministic") or profile
    run = REPO / "runs" / (args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + args.regulation.replace(" ", "_"))
    run.mkdir(parents=True, exist_ok=True)
    reqs, raw = extract(args.regulation, product)
    sub = {"team_id": args.team, "case_id": args.case, "requirements": reqs}
    (run / "submission.json").write_text(json.dumps(sub, indent=1, ensure_ascii=False))
    (run / "raw.json").write_text(json.dumps(raw, indent=1, ensure_ascii=False))
    (run / "system_prompt.txt").write_text(SYSTEM)
    print(f"{len(reqs)} requirements -> {run / 'submission.json'}")
    return run


if __name__ == "__main__":
    main()
