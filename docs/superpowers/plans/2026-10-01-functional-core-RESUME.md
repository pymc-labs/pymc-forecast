# Resume: functional core (issue 57)

Paused 2026-10-01 so the laptop can shut down. The approval gate was already
accepted. Do not stop at the old "waiting for approval" line.

## Checkout

- Repo: `/Users/juanitorduz/Documents/pymc-forecast`
- Branch: `feat/functional-core` (this checkout, no worktree)
- Base: `1ab725e` (`main`, "Add observation-driven recursions and forecasting examples (#56)")
- Task 1: `bf073fc` Take covariates then data in model bodies
- HEAD: `26322c9` Replace time_series with innovations
- No pull request. Do not open one until Task 8. Version stays `0.2.0`.

## How to resume

Use `superpowers:subagent-driven-development` against
`docs/superpowers/plans/2026-10-01-functional-core.md`.

The working ledger is gitignored:

`.superpowers/sdd/2026-10-01-functional-core/progress.md`

If that directory is still on disk, append to it. If it is gone, this file
is the source of truth; recreate the ledger and continue.

## Done

Task 1 is complete. Review was spec compliant with no findings. Reviewer
output schema dropped the template headings; the transcript was a full diff
read. Ruling: that counts as spec pass and task quality Approved.

Landed:

- Model bodies are `(covariates, data=None)`.
- `Horizon.from_data` exists. `from_arrays` is gone. No alias.
- `build_model` still owns `pm.Model` and coords, then calls
  `model_fn(cov_da, data_da)`. No `isinstance` branch.

Task 2 is committed and not reviewed. `26322c9` replaced `time_series` with
`innovations`, added private `expand_dist`, and extended `predict`.
`markov_series` was not added. `markov_time_series` still exists.

Implementer concerns, noted and not resolved before review:

- Three existing 4-argument factories in `tests/test_alignment.py` and
  `tests/test_forecaster.py` named the observed parameter `obs`. Renamed to
  `observed` so the spec's factory dispatch still calls them. They still
  receive `(name, latent, dims, observed)`.
- Installed PyMC `StudentT.dist` takes `sigma` keyword-only.

A reviewer (`Task2Review`) was dispatched on `bf073fc..26322c9` and the wait
was aborted. No verdict came back. Do not mark Task 2 complete. Re-review
from scratch.

Regenerate the package if the local diff file is missing:

```bash
bash ~/.claude/plugins/cache/claude-plugins-official/superpowers/6.3.0/skills/subagent-driven-development/scripts/review-package \
  docs/superpowers/plans/2026-10-01-functional-core.md \
  bf073fc242736d2eaf00102d31588fbd2436310d \
  26322c9cddad927f535d6df87c6cb47ae98a0f8f
```

Brief: `.superpowers/sdd/2026-10-01-functional-core/task-2-brief.md`
(regenerate from the plan's Task 2 section if missing).
Dispatch table the implementer had to follow: plan sections `innovations`
and `predict`, copied at review time to `task-2-api.md`.
Report, if still on disk: `task-2-report.md`.

## Next action

1. Re-dispatch the Task 2 reviewer. Read-only. Do not re-run the suite.
2. Clean review: mark Task 2 complete and dispatch Task 3.
3. Fixes required: fix loop, then one scoped re-review.

## Remaining

| Task | State |
|---|---|
| 3 markov_series rename | not started |
| 4 ssoe argument order | not started |
| 5 functional fitters | not started |
| 6 migrate the nine current notebooks | not started |
| 7 docs and changelog | not started |
| 8 full verification, then one PR | not started |

Task 3 ruling, already made: add `ForecastingModel.markov_series` as a thin
bound helper and edit `model.py` for that only. No `advance`. Delete
`markov_time_series`. No alias. Files: `markov.py`, `__init__.py`,
`tests/test_markov.py`, any other importer, plus that one method in
`model.py`.

## Rulings already made

- Implement on `feat/functional-core` in this checkout. No worktree.
- Task 2 edits `__init__.py` even though its file list omits it. Done.
- Task 2 replaces bound `ForecastingModel.time_series` with `innovations`. Done.
- Class bodies that never read the horizon do not assign `h = self.horizon`.
- Do not invent a `drift_raw` test. That rewrite is the notebook migration.
- Do not add an `isinstance` branch in `build_model`.
- Task 2 retargets the one `:func:`~pymc_forecast.model.time_series`` role in
  `markov.py` to `innovations`. Done. No other markov edit in Task 2.
- Do not change `StatespaceModel`. Do not patch it for Python 3.14.
- No VAR, no `time_reparam`, no Haar/DCT, no new notebooks, no version bump.

## Baseline

After `uv sync --all-extras`, on Python 3.14 / pymc 6.3.2 / pymc-extras 0.15.1:

- At `1ab725e`: 13 failed, 315 passed, 5 errors, all in `tests/test_statespace.py`.
- After Task 2: 13 failed, 326 passed, 5 errors, same file. No new failure.

CI on `1ab725e` is green for Python 3.11 and 3.13. A task is green when
`uv run pytest -q` adds no failure outside that set. Task 8 will not be fully
green on this machine. PR CI on 3.11/3.13 is the release gate.

Notebooks stay red until Task 6. Pytest does not execute them.
