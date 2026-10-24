# Sub-goal dev case: the one remaining miss (R-085)

Case: `package/subgoal/dev/`, UN R175 (accelerator control misapplication), old version 0.1.0 -> new version 1.0.0.
Our score locally: flag recall 24/24, change accuracy 23/24. R-085 is the only item judged wrong.

## 1. Old PDF (`old.pdf`, page 9, paragraph 6.5)
> The test procedure shall be performed with and without an expected ACPE intervention both in a forward and rearward
> direction and at least with the vehicle situated 1.0 m (+0.1 m tolerance) and 1.5 m (-0.1 m tolerance) from the
> target position at the start of the test (as detailed in Table 1).

## 2. New PDF (`new.pdf`, page 10, paragraph 6.6 "Test procedures")
> The test procedures shall be performed both in a forward and rearward direction and at least with the ACPE
> interventions triggered at 1.0 m (+0.1 m tolerance) and 1.5 m (-0.1 m tolerance) distance from the target position
> (as detailed in Table 1).

## 3. What changed in the regulation (old -> new)
| Old | New |
|---|---|
| test procedure | test procedure**s** |
| with and without an expected ACPE intervention | *(removed)* |
| both in a forward and rearward direction | both in a forward and rearward direction (**unchanged**) |
| with the vehicle situated 1.0 m ... and 1.5 m ... | with the ACPE interventions triggered at 1.0 m ... and 1.5 m ... |
| from the target position | **distance** from the target position |
| at the start of the test | *(removed)* |
| (as detailed in Table 1) | (as detailed in Table 1) (unchanged) |

## 4. Old requirement (`requirements_old.json`, R-085, cited page 8, paragraph 6.5)
> The accelerator control misapplication test procedure shall be performed with and without an expected ACPE
> intervention both in a forward and rearward direction and at least with the vehicle situated 1.0 m (+0.1 m
> tolerance) and 1.5 m (-0.1 m tolerance) from the target position at the start of the test (as detailed in Table 1).

## 5. Expected (`expected.json`)
- change: `modified`, source: page 10, paragraph 6.6
- new_text:
  > The ACPE test procedures shall be performed at least with the ACPE interventions triggered at 1.0 m (+0.1 m
  > tolerance) and 1.5 m (-0.1 m tolerance) distance from the target position (as detailed in Table 1).
- reason: Paragraph 6.6 (formerly 6.5) of UN R175 was amended in version 1.0.0: 'vehicle situated' became 'ACPE
  interventions triggered at'; 'with and without an expected ACPE intervention' was removed.

## 6. Ours (`upload/subgoal-dev_changes.json`)
- change: `modified`, source: page 10, paragraph 6.6 (same as expected)
- new_text:
  > The accelerator control misapplication test procedure shall be performed both in a forward and rearward direction
  > and at least with the ACPE interventions triggered at 1.0 m (+0.1 m tolerance) and 1.5 m (-0.1 m tolerance)
  > distance from the target position (as detailed in Table 1).
- reason: Paragraph 6.6 (formerly 6.5) of UN R175 was amended in version 1.0.0: 'procedure' became 'procedures';
  'with and without an expected ACPE intervention' was removed; 'vehicle situated' became 'ACPE interventions
  triggered at'; 'distance' was added; 'at the start of the test' was removed.

## 7. The difference
| | Ours | Expected | New PDF |
|---|---|---|---|
| change type | modified | modified | |
| page / paragraph | 10 / 6.6 | 10 / 6.6 | 10 / 6.6 |
| "both in a forward and rearward direction" | **kept** | **missing** | **present** |
| subject | "The accelerator control misapplication test procedure" | "The ACPE test procedures" | "The test procedures" |

The only meaningful difference is "both in a forward and rearward direction". It is in the old and the new
regulation (and Table 1 still lists Forward and Rearward rows), but the expected new_text leaves it out, and the
expected reason does not mention removing it. The judge compares against the expected text, so ours is marked
"adds a constraint". Ours follows the regulation; matching the expected text would mean dropping a real requirement,
so we keep it.
