# New C1 traverse integrate

Status: design proposal only. No implementation changes.

## Objective

Extract regulatory requirements in batches first, then attach their source context through traversal. Keep the number of LLM calls tied to the amount of source text, not the number of extracted requirements.

The output is a list of requirements stated in their own source sections, with referenced clauses, annexes, tables and other supporting text attached as context. Traversal does not rewrite requirements or require an LLM call.

This proposal supersedes the earlier traversal → C1 → applicability-blackbox integration direction for this workflow. It introduces no second applicability evaluator, condition-extraction call or requirement-completion call.

## Proposed flow

    Parsed regulation PDF: main text + annexes
                       ↓
    Build batches from document sections in source order
                       ↓
    C1: one extraction call per batch, many requirements per response
                       ↓
    Validate cited source nodes and preserve candidate records
                       ↓
    Code-only traversal rooted at each requirement's source node(s)
                       ↓
    Attach source text, parent scope and referenced material
                       ↓
    Save requirements + shared context nodes + traversal diagnostics

Illustration, not a performance estimate: five text batches can yield 200 requirements through five extraction calls. Traversal can then build 200 context associations without any additional LLM calls.

## Extraction scope

For complete-PDF extraction, enumerate every parsed source node in the chosen regulation version, including annexes and appendices. Do not use product-based A/B section selection or Stage T extract/context modes to exclude sections from C1 input.

Definitions, scope and explanatory material must be read where needed but need not become requirements. Requirements in an annex are extracted when that annex is processed as primary source text, independently of references from the main body.

This changes C1's extraction scope: retain its batching, structured-output call and source-bound writing approach, but remove instructions that suppress requirements based on the current product. Keep applicability conditions in the requirement text. Product matching is outside this design; these records must not be represented as an already product-filtered challenge submission.

“Every parsed node was processed” is a coverage claim, not proof that every requirement was correctly extracted. Missing PDF content, model omissions and incorrect grouping remain possible.

## What changes from the current code

| Component | Current behavior | Proposed behavior |
|---|---|---|
| C1 job selection in bbpipe.py | Reads traversal items marked extract | Reads all nodes from an explicitly chosen PDF/version |
| C1 job context | Adds referenced text before extraction | Supplies primary batch text and inherited local scope; external reference expansion follows extraction |
| C1 prompt | Product-specific extraction and applicability rationale | Source-local regulatory extraction, preserving conditional wording without product filtering |
| C1 response | Multiple requirements tied to node IDs | Same list-based pattern; each requirement identifies the source node(s) where it is stated |
| Traversal | Expands selected sections before C1 | Expands references from each extracted requirement's source node(s) afterward |
| Final processing | May filter, deduplicate, merge or truncate candidates | Preserve extracted records; attach context and diagnostic metadata |

Reuse the existing corpus, rendering, chunking and cached structured-call utilities where compatible. Adapt the traversal reference scanner and resolver rather than calling the current stage-B-oriented traverse entry point unchanged.

## 1. Batch the source text

Group nodes in document order by main-body sections and annex/part, using the existing approximately 40,000-character primary-text target as an initial configurable setting. This is a batching heuristic, not a guarantee of token size or a fixed number of calls.

Preserve clause boundaries, table headers and notes. The existing chunker can produce oversized chunks when a single subsection exceeds its target; explicitly detect those cases. Split at smaller source boundaries while repeating necessary headers/scope, or record a failed oversized batch for review. Never silently slice away source text.

Assign every node a primary extraction batch. Where ancestor text is repeated to preserve scope, mark it as context-only so it does not trigger repeated extraction. Maintain a manifest recording primary node IDs, supporting node IDs and extraction status for each batch.

Use qualified source IDs including regulation, version and node key. Keep primary extraction within one regulation/version per job.

Reading a whole PDF may cost more than reading a product-selected subset. Report batch count and input size before running; do not claim the same call count regardless of expanded coverage.

## 2. C1 extracts what is stated locally

Each call returns zero, one or many requirements from its primary source nodes. Retain source-grounded wording, numerical values, units, alternatives and conditions actually stated in the supplied source.

For an obligation stated by reference, preserve the reference rather than inventing or importing unavailable detail. For example:

- Main clause: “The device shall meet the requirements of Annex 6.” Extract that statement from the main clause.
- Annex clause, processed in its own batch: “After exposure, performance shall meet X.” Extract that statement from the annex.
- Attach Annex 6 to the main-clause requirement through traversal later. Do not automatically merge the two requirements.

The example is illustrative, not a quotation from a particular regulation.

Table requirements are extracted when their containing source is processed. Distinguish independent test rows from values belonging to a single measurement grid, preserving header relationships. Traversal does not repair malformed tables or verify generated numerical values.

Do not ask C1 to return one requirement per source node: a node may contain several requirements or none. Do not merge distinct requirements merely because they cite the same paragraph. Retain the existing useful granularity instructions, but remove product-specific exclusions and product-based applies_because generation. Test-case generation is not necessary for this extraction artifact and should be optional rather than an extra call.

## 3. Validate and preserve extraction results

Check that every requirement cites primary source node IDs supplied in the batch. If an output cites only supporting material or an unknown ID, retain it as a rejected/flagged candidate with a reason rather than fabricating provenance.

Derive regulation, version, page and paragraph metadata from the cited source nodes. Assign stable internal requirement IDs scoped to the extraction run and source version. Preserve the original response and the source text without truncation.

Keep batch failures and suspicious empty responses visible. An empty response is not proof that a section contains no requirements. Source coverage records should distinguish processed, failed and not processed nodes, and identify empty-output batches for inspection.

No automatic fuzzy deduplication or Stage D merge pass is included. Flag potential duplicates for analysis; shared source identity does not establish semantic equivalence. Scope text repeated across batches should not produce duplicate requirements because only primary nodes are extraction targets.

## 4. Traverse after extraction

For each requirement, use its cited source node(s) as roots. Include the source text itself, ancestor scope and targets reached through explicit references. Reuse deterministic reference parsing for clauses, annexes, appendices and tables.

Traverse the original source text, not only the LLM-written requirement: paraphrasing may omit a reference. Follow reference chains to the configured depth, resolve cycles and preserve source-to-target edges and reference text.

This is code-only work. If several requirements cite the same node or reference the same annex, resolve and cache those dependencies once and reuse them. Cache keys must include source version/corpus fingerprint and traversal settings.

A source paragraph may contain references relevant to several different requirements. Initially attach its full source-node reference context to each requirement from that paragraph. This is an intentional conservative approximation; do not introduce an LLM call merely to prune context. Retain edges so later consumers can see why text was attached.

Parent scope can be attached through document hierarchy. Term definitions without explicit references are not guaranteed to be discovered by a reference scanner. Include known scoped definition material where deterministically available and document the remaining coverage limitation.

### Cross-regulation references and limits

Context from another regulation is supporting material. It does not create additional extracted requirements from that regulation automatically. To extract those requirements, process that PDF separately through C1.

Resolve the explicitly referenced version where available. If the corpus only offers another version, record the mismatch; do not silently represent base as the requested amendment series.

Keep hop limits, unresolved references, missing target documents, ambiguous table targets, skipped nodes and truncated expansions in the output. Context completeness means “resolved within the declared traversal policy,” not proof that every legally relevant source was found.

If traversal reaches a budget, preserve the requirement and report incomplete context. Do not drop the requirement, guess the missing text or initiate automatic LLM completion. Handle a single oversized node without an empty-list indexing crash.

## 5. Output contract

Use two linked JSON collections so large annex text is not copied into every requirement record.

A requirement record contains:

- id and text;
- regulation/version and source_node_ids;
- source page/paragraph metadata;
- extraction batch ID;
- context_node_ids and traversal edges;
- context diagnostics and extraction validation status.

A shared context-node record contains:

- qualified id;
- original text, including parsed tables;
- regulation/version, node key, page and paragraph;
- source corpus fingerprint.

The reference edges identify the source node, target node, relation and original reference text. A requirement can share context nodes with many other requirements while retaining its own root and associations.

Suggested artifacts:

| Artifact | Contents |
|---|---|
| extraction_manifest.json | Input files/versions, fingerprints, model/prompt settings, batches and processing status |
| c1_candidates.json | Raw/enriched extracted requirements and flagged outputs |
| requirements_with_context.json | Requirement records and their context associations |
| context_nodes.json | Shared resolved source text |
| context_diagnostics.json | Unresolved references, limits, ambiguities and version mismatches |

The consumer can resolve context_node_ids into raw text when needed. There is no need for a new database or a separate service.

## LLM call budget

Normal path: one C1 call per text batch. No per-requirement applicability, condition extraction, context summarization or completion calls.

Retries for failed or malformed extraction responses are additional calls and should be bounded and reported. Large sections can require additional batches; small sections can be grouped where hierarchy and provenance remain intact.

After extraction, traversing 200 requirements entails local parsing/lookups, not 200 model requests. Context text can be large on disk without being sent back into an LLM in this workflow.

## Acceptance and evaluation

- Every parsed source node has an extraction-batch assignment or an explicit input failure.
- A single batch can produce multiple requirements, each with a valid source citation.
- Every resulting requirement has its own source text and associated traversal context or explicit diagnostics.
- Annex requirements are independently extracted, while annex text can also serve as context for main-body requirements.
- Calls scale with batches and retries; the traversal stage makes zero LLM calls.
- Shared references, cycles, cross-PDF references and differing versions preserve correct identities.
- Oversized batches and traversal targets do not silently truncate content or crash without a diagnostic.
- No model-generated requirement is silently removed or merged during context attachment.

Validate source fidelity on a manually reviewed sample: main-clause/annex pairs, measurement-grid tables, distinct test rows, exceptions and conditions inherited from headings. Compare omission and grouping errors with existing C1 runs. Full regulatory extraction includes requirements irrelevant to the rear-lamp product, so its raw output precision against product-specific expected.json is not an appropriate standalone quality measure. Keep that gold data evaluation-only.

## Implementation sequence

1. Add an opt-in entry point for full-PDF C1 batching; retain existing product-specific pipelines.
2. Adapt the C1 prompt/schema for source-local extraction while reusing the existing model-call infrastructure.
3. Preserve raw candidates and validated source links.
4. Adapt traversal to requirement-source roots and shared dependency caching.
5. Export linked requirement/context artifacts and processing diagnostics.
6. Compare source coverage, extraction fidelity and call count on representative PDFs before broader use.

## Deliberate boundaries

This design provides batched regulatory extraction plus attached context. It does not guarantee perfect extraction, fix upstream PDF parsing, determine product applicability, assess compliance, or format a final product-specific challenge submission. Those are separate consumers or experiments, not hidden extra passes in this workflow.
