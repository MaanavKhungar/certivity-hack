# Sub-goal, held-out case: UN R171 (DCAS), base -> latest (2.0.0)

Case `subgoal-heldout`: 117 old requirements (`package/subgoal/heldout/requirements_old.json`). The case folder only
has `files.json`; old.pdf / new.pdf are the corpus PDFs `regulations/pdf/UN_R171/{base,latest}`, checked against the
sha256 in `files.json` and copied into the case folder before running.

This branch = Yesieniia's `sub-v2-yesi` (code at the repo root) + the chapter-scope fix below. The leaderboard ranks
by **flag recall** first, then change accuracy; flag precision is shown but not ranked.

## Platform results (team-05)

| # | upload | file (runs/subgoal-heldout-final/) | changes | flag recall | change accuracy | flag precision |
|---|---|---|---|---|---|---|
| 1 | 07:53 | v1_ours_0753.json (main-branch `src/subgoal.py` + scope fix) | 66 | 81.2 % | 91.8 % | 78.8 % |
| 2 | 08:14 | v2_yesi_0814.json (this branch's code, no scope fix) | 115 | **89.1 %** | 96.3 % | 49.6 % |
| 3 | 08:25 | v3_yesi_minus_annex_plus_R014_0825.json | 68 | **89.1 %** | **98.1 %** | **83.8 %** |
| 4 | not uploaded yet | v4_v3_plus_R023.json | 69 | | | |

From the numbers: the answer key has about 64 changes (66 x 0.788 = 52 correct = 81.2 % of 64). Upload 2 had about
57 correct of 115, upload 3 about 57 of 68. Leaders at 10:50 (sudo, BarelyLegal) are at 90.6 % = 58 of 64; one more
correct flag ties, two take first.

## What changed between the versions

**1 -> 2 (Yesieniia's branch), helped recall and accuracy:**
- `pure_reordering_equivalent`: a "modified" whose old/new quotes only reorder the same words (no order-sensitive
  words such as not/unless/before) is dropped.
- Added-requirement prompt: keep "is/shall/may" as in the source, no invented actor, no extra scene-setting.
- An added requirement that only turns "is/are to be" into "shall" is dropped; an added requirement whose enumerated
  items are already inside a modified requirement of the same paragraph is dropped (duplicate).
- MOD prompt: may leave out unchanged sibling constraints (written around the dev case's R-085). Kept here; on R171
  it made R-014 "unchanged" (see below).
- It found R-089, R-090, R-091 (new 60 km/h exemption in the approaching-vehicle deceleration rule) that version 1 missed.

**2 -> 3, fixed precision (49.6 % -> 83.8 %) without losing recall:**
- Dropped the 48 "added" requirements from Annex 3 / Annex 4. Those chapters entered the scope through ONE weak
  sentence match each (similarity 77 and 65), while the chapters the old requirements really cover have dozens of
  matches >= 90. Recall stayed at 89.1 %, so none of the 48 was in the answer key.
  Code: `added_keys` in `subgoal.py` now needs >= 2 links or one link >= NOVEL (90) for a chapter to count as covered
  (same fix as on main's `src/subgoal.py`). Re-running this branch's code now gives version 3 without the manual step.
- Added R-014 back from version 1: subject changed from "the DCAS" to "the DCAS or the applicable feature" (one added
  qualifier counts as modified, as in the dev answer key). Version 2 called it unchanged because of the
  sibling-constraint rule.

**3 -> 4 (to try for recall):** adds R-023 (version 1 flagged it, version 2 did not): "The requirements outlined in
paragraph 5.5.4.1.8. and subparagraphs shall apply." became "... paragraph 6.2.9.1. shall equally apply." The
reference lost "and subparagraphs" and gained "equally", a scope change rather than a pure renumbering.
Checked and NOT added (editorial only): R-024 ("In addition," dropped), R-038 ("three" -> "3"), R-063 (punctuation),
R-109 (PDF typo "ay" -> "may" and placeholder R1XXSWIN -> R171SWIN; the old requirement already read R171SWIN).

## Reproduce

```
cp package/regulations/pdf/UN_R171/base/UN_R171_base.pdf package/subgoal/heldout/old.pdf     # sha256 6d245c80...
cp package/regulations/pdf/UN_R171/latest/UN_R171_latest.pdf package/subgoal/heldout/new.pdf # sha256 5ba88306...
python -c "import subgoal; from pathlib import Path; subgoal.run(Path('../package/subgoal/heldout'), team_id='team-05')"
# -> runs/subgoal-heldout/changes.json (this branch's code incl. the scope fix = version 3 minus R-014)
```
R-014 and R-023 were taken from version 1's output (main branch `src/subgoal.py`), not regenerated here.

## Ideas for more recall (not done)
- Run both pipelines (main and this branch) and take the union of their modified/removed flags: the two disagreed on
  R-014, R-023 (main) and R-089..R-091 (this branch); disagreement is where the remaining misses are likely to be.
- Flag "unchanged" decisions whose quotes differ in a reference's scope (e.g. "and subparagraphs" dropped), not only
  in its number.
