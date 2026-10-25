# Design: traversal → C1 → applicability blackbox

Status: design only; no pipeline code changes. Based on main commit 4679d23 and the existing blackbox module.

## Goal

Use Stage T's reading list to feed the current C1 extraction, preserve the resulting candidates and their regulatory context, assemble product evidence and applicability conditions, and evaluate each candidate through blackbox.py. Keep extraction and applicability as separate, inspectable stages.

This is an integration design, not a replacement extractor. Initially retain the existing C1 extraction prompt/schema and explicitly measure its remaining product-filtering bias. C2 is outside the initial experiment; it can later process the same jobs as an independently measured recall pass.

## Current behavior versus proposed behavior

The current hybrid7 main function runs traverse.traverse(A['C'], B) and saves traversal.json, but then calls H.section_jobs(A, B, profile). C1 therefore still receives the old section selection and reference expansion, not Stage T's new reading list. C2 follows, then H.assemble performs deduplication and truncation, followed by Stage D review.

Proposed flow:

    Product documents → existing profile → A/B selection
                                             ↓
                              Stage T / traversal.json
                                             ↓
                             Resolve nodes from Corpus
                                             ↓
                            Traversal-backed C1 jobs
                                             ↓
                           Current C1 requirement extraction
                                             ↓
                       Preserve candidates and context links
                                  ↓                  ↓
                         Condition extraction   Context bundles
                                  ↓                  ↓
    Specification + profile → product evidence adapter
                                             ↓
                      blackbox.evaluate(candidate, context,
                                        product, conditions)
                                             ↓
                          Save all applicability decisions
                                             ↓
                   Applicable / not applicable / unresolved
                                             ↓
                         Explicit submission policy + format

For the first experiment, bypass C2 and Stage D and avoid H.assemble's destructive deduplication/truncation. Build submission objects with a separate formatter that preserves content and provenance. Keep the original pipeline runnable as the baseline.

## What produces each blackbox argument?

| Argument | Producer | Required mapping |
|---|---|---|
| requirement | Existing C1 output, linked by H.tie or an equivalent adapter | Add a stable internal candidate ID; preserve text, test_case, cited node and job ID. |
| context | traversal.json plus the parsed Corpus | Resolve node keys to actual text and metadata; select the candidate's source and dependency bundle. |
| product | Original spec.md, existing profile and Stage B mapping | Create passages and normalized facts with evidence references; distinguish explicit, inferred, unknown and conflicting information. |
| conditions | New candidate-level source-grounded extraction pass | Extract applicability predicates, scope and exceptions from the context, preserving links to their source. |

The existing blackbox does not construct these inputs, traverse references, read PNGs or generate requirements. It retrieves product passages, performs supported atomic checks, invokes a model and validates the returned decision.

## 1. Consume traversal.json as a reading manifest

Stage T emits items with id, regulation, key, nodes, origin, hop, cross_pdf, mode, via, pages and truncated, plus unresolved references, skipped items and regulation hints. It does not embed the raw node text or an explicit version per item.

Resolve each node using Corpus.index[(item.regulation, 'base', node_key)]. Persist the resolved version and corpus/file fingerprint in the run manifest so a saved traversal cannot silently bind to a different parse later. Missing node keys are input errors, not empty context.

Use mode='extract' items as C1 extraction targets. Attach relevant mode='context' items as supporting evidence. Do not turn every context item into a new extraction target automatically. Cross-PDF context can qualify a requirement without establishing that the entire referenced regulation applies independently.

Deduplicate node membership using regulation, version and node key. Keep every traversal path for provenance. Do not use traversal item IDs as candidate IDs: one item can contain multiple requirements.

### Limits that must remain visible

- Default traversal depth is three same-PDF hops; cross-PDF targets are not recursively scanned.
- By default, same-PDF references beyond hop one are context, not extraction targets.
- Item and reference expansion budgets can truncate or skip content. The current add routine also applies per-item limits to selected material.
- Definitions can be removed from selected subtrees, and inherited scope may not have an explicit reference.
- Cross-regulation targets currently resolve against base even when reference wording names a particular amendment series. A version mismatch must be flagged, not represented as verified context.
- A regulation hint is a lead, not resolved evidence.

Propagate these diagnostics into each affected bundle. Include parent scope and necessary definitions explicitly. If a potentially material dependency is missing, keep the candidate for review and require unresolved status until it is resolved or demonstrated immaterial. Do not globally invalidate unrelated candidates for another section's missing reference.

## 2. Build C1 jobs from resolved traversal items

Reuse C1's structured extraction call, existing requirement schema, text rendering and extraction instructions. Replace only the job producer initially. Each job has primary extract nodes, supporting context nodes, product/profile context as currently expected, and a provenance manifest.

Chunk primary content at clause boundaries. Keep a table with its headers, notes and invoking clause where possible. Attach necessary context per job; do not concatenate every regulation's entire traversal into every prompt. If the complete bundle exceeds a budget, split deliberately or flag incomplete context rather than cutting strings.

Use unique prompt node IDs across regulations and versions. Existing n0-style IDs are local to a regulation; cross-PDF material makes those unsafe without a qualified lookup. Reject an extraction citing an unknown node. Record whether it cites a primary or supporting node. Candidates extracted from supporting material should be retained and marked for review rather than silently lost or duplicated across jobs.

C1's current prompt already makes product relevance decisions. Retaining it is intentional for a controlled first integration, but blackbox cannot recover requirements C1 never produces. A later experiment may remove product-based suppression; this proposal must not claim to solve that omission in its initial form.

## 3. Preserve candidates before assembly

Save the raw C1 response and enriched candidate records before calling any deduplication or formatting logic. Include stable candidate ID, job ID, original node_id, resolved source identity, text, test_case and traversal item IDs. Stable IDs should be scoped to the run/corpus version and remain unchanged when final submission IDs are renumbered.

A candidate may cite one node but need several sources. Preserve its complete bundle separately. Identical-looking candidates can be flagged as possible duplicates; do not delete them before the applicability comparison. Final duplicate consolidation needs explicit lineage and semantic preservation checks.

## 4. Construct candidate-specific context

Start with the candidate's cited node and the C1 job's evidence. Include relevant parent scope, definitions, referenced clauses, annexes, table blocks and footnotes. Every context record must have a unique id and nonempty text, plus source metadata and its relationship to the candidate.

Use traversal via paths to identify connections, but do not assume via is a complete node-level dependency graph: items aggregate subtrees and deduplication records paths at item ownership level. When an edge cannot be resolved precisely, retain the larger associated job context and flag the uncertainty. A later improvement can persist explicit node-to-node edges.

Preserve raw source text. Do not replace it with a condition summary. Preserve source and target regulation versions for cross-PDF references. Context-only documents support interpretation and are not automatic evidence of product applicability.

Persist completeness metadata with each requirement payload so the evaluator can see it. Because blackbox currently does not enforce traversal completeness mechanically, the integration layer must override a decisive result to unresolved when material dependency issues remain, recording the original model decision and the reason for the override.

## 5. Adapt product evidence once per run

Build product.passages from original specification sections, preserving document identity and section/line or page locations. Create textual drawing-observation records from the existing vision profile, with image and region/callout references when available. Blackbox currently consumes text, not image pixels.

Build product.facts using a shared vocabulary agreed with the condition extractor, such as component-specific replaceability, lamp function or voltage. Retain value, unit, status and evidence_ids. Qualify fields by component where necessary: one replaceable bulb does not establish that every module is replaceable.

Verify profile evidence quotes against their source passages before marking facts explicit. Preserve model-derived class mappings as inferred unless the source explicitly states the class. Stage B's confirmed boolean alone is not enough. Keep open_options as uncertainty, not several simultaneously true facts. Preserve contradictory evidence instead of selecting a convenient value.

The existing rule engine only evaluates explicit facts and matching units. It performs no unit conversion. Normalize units with a tested conversion step or leave comparisons unresolved. A plain narrative condition can still be evaluated by the LLM; it need not be forced into an unreliable machine predicate.

## 6. Extract applicability conditions after C1

Introduce a source-grounded pass taking candidate text and its context bundle. Product evidence must not influence what conditions the regulation is said to impose.

Return condition records with id and text; optional field, op, value and unit enable current deterministic checks. Preserve regulatory evidence IDs, exceptions, AND/OR structure and interpretation issues in companion metadata attached to the requirement. Current blackbox delegates Boolean logic and exceptions to the LLM; a flat condition list must not be interpreted as an implicit conjunction.

Conditions answer 'when does this requirement apply?' They are not compliance criteria. For example, 'module is replaceable' may be an applicability condition; 'module carries a marking' is the requirement to satisfy. Failure to carry the marking must never make the requirement inapplicable.

An empty conditions array is allowed only with an explanation, such as no additional local conditions after inherited scope was captured. It does not prove universal applicability. Record ambiguous extraction as an interpretation issue, not a guessed condition.

## 7. Invoke the existing blackbox consistently

All variants receive the same persisted input bundle:

    engine.evaluate(requirement, context, product, conditions)

- direct_gpt receives all supplied evidence and uses the explicitly configured GPT model.
- retrieval_rules selects product passages, pins fact evidence, performs supported comparisons and invokes the configured LLM. Default LLM is the existing pipeline caller, not GPT.
- bert adds an explicitly configured trained CrossEncoder reranker. It is not a trained applicability classifier. Optional semantic scoring can augment retrieval in the latter two variants.

Use the same LLM adapter/model for controlled comparisons where possible; otherwise label results as both architecture and model changes. Record variant, provider/model, retrieval settings, reranker checkpoint, prompt/input hashes and source versions. Do not assume model decision quality from the fact that a response passed schema validation.

Save every decision and trace, including excluded and unresolved candidates. Treat API errors, invalid citations, malformed responses and exceeded input budgets as failed evaluations eligible for retry/review, never as not_applicable. Current blackbox may raise these errors; the orchestrator must catch and persist them per candidate without concealing a partial run.

## Illustrative mapping (not a quotation from the actual regulation)

Source context: 'For replaceable modules, an identification code is required.'

C1 candidate: 'Replaceable modules must carry an identification code.'

Condition pass: field='module.replaceable', op='eq', value=true, unit=null, linked to the source paragraph.

Product passage: 'This module is non-replaceable.' Product fact: same field, value=false, status='explicit', evidence_ids pointing to that passage.

Blackbox: not_applicable, with the source clause and product evidence cited. If replaceability is missing or conflicting, unresolved. If the module is replaceable but lacks the code, the requirement is applicable; compliance would be a separate assessment.

## 8. Decisions and submission policy

Keep separate applicable, not_applicable, unresolved and evaluation_failed collections. Generate the main review list from applicable candidates. Preserve conditional requirements and missing-information questions in the unresolved review artifact.

Choose the challenge submission policy explicitly: confirmed-only versus inclusion of clearly worded conditional requirements. Measure the precision/recall tradeoff and preserve the policy in the run manifest. Do not silently treat unresolved as irrelevant.

Do not run current Stage D after blackbox in the initial experiment: its independent keep/drop/merge policy could undo the applicability decisions. Retain candidate-to-final mappings through any later formatting or duplicate consolidation. Enforce output length limits through deliberate reformulation and completeness checks, never character slicing.

## Proposed run artifacts

| Artifact | Purpose |
|---|---|
| traversal.json | Original Stage T reading manifest and diagnostics |
| run_manifest.json | Corpus fingerprints, source versions, configuration, model choices and submission policy |
| c1_jobs.json | Actual primary/context source IDs sent to extraction |
| c1_candidates.json | Raw and enriched candidates before destructive processing |
| context_bundles.json | Per-candidate source text, provenance and completeness diagnostics |
| product_evidence.json | Shared passages, normalized facts, uncertainty and provenance |
| conditions.json | Per-candidate applicability predicates and source links |
| blackbox_inputs.jsonl | Exact evaluation inputs for reproducibility |
| applicability_<variant>.jsonl | Decisions, validation diagnostics, traces and failures |
| submission_mapping.json | Candidate IDs represented by each final requirement |
| unresolved.json | Missing facts, source ambiguities and incomplete context for review |
| submission.json | Challenge-compatible formatted output |

Keep expected.json and traversal trace.json strictly evaluation-only. trace.json contains gold-derived coverage information and must never feed extraction, context selection, condition generation or applicability.

## Validation and rollout

1. Offline adapter checks: every traversal node resolves against the recorded corpus; IDs are unique; every candidate maps to a source; all fact references resolve; no unknown context is silently dropped.
2. Traversal-only comparison: current C1 with old jobs versus current C1 with traversal-backed jobs. Hold prompt/model and product inputs fixed.
3. Freeze traversal-backed candidates and context. Manually inspect a sample of condition extraction and product fact mappings before evaluating blackbox.
4. Run all three variants against identical persisted inputs. Report model choices and retrieval settings, applicability errors, unresolved rate, evidence quality, cost and latency.
5. Compare final benchmark precision/recall separately from source coverage and semantic correctness. Missing C1 requirements cannot be credited to or blamed on blackbox.
6. Inspect R150 weathering coverage, R10 table interpretation, inherited conditions, cross-regulation version mismatches, optional configurations, contradictory product facts and duplicate candidates. Add unseen cases beyond those used to develop the adapters.
7. Only after the first comparison, evaluate C2 as an optional recall addition using the same traversal-backed jobs and adapters.

Acceptance: traversal is demonstrably the source of C1 jobs; each blackbox argument has an identifiable producer; all decisions have inspectable inputs; material missing context cannot silently yield a final confident decision; no candidate disappears without a disposition; and evaluation ground truth never enters generation.

## Boundaries

This integration does not repair damaged PDF tables, guarantee complete reference discovery, recover sections excluded by A/B, or recover requirements suppressed by the unchanged C1 prompt. It provides a reproducible path for evaluating applicability using the current extractor and new traversal. Those remaining sources of omission must be measured separately.
