# SHADER INSTRUCTIONS

## Overview

This tree contains the installable Iris shader pack: GLSL entry points, reusable rendering modules, shared bindings and
utilities, and runtime assets.

## Where to Look

Use `docs/en/modules/pipeline.md` for live pass order. Module contracts and validation targets are under
`docs/en/modules/` for geometry/materials, shadows, GI, atmosphere/clouds, water/translucency, and post-processing.

## Placement and Naming

- `pass/` contains only entrypoint shaders and is the standard location for new entries.
- Existing Bloom, RTWSM, and atmospheric compute entries under `techniques/` stay in place, but are not a template for
  new placement.
- Put module-local implementation in `techniques/<module>/` and only cross-module helpers in `util/`. Debug code belongs
  under `techniques/debug/`.
- Use `.glsl` for shared code without `main`. Entry shaders use `comp.glsl`, `frag.glsl`, `geom.glsl`, or `vert.glsl`.
  Root `.csh`, `.fsh`, `.gsh`, and `.vsh` files exist for Iris compatibility or generated program wrappers.
- `gbuffers_*`, `shadow*`, `dh_*`, and `final.*` include implementations under `pass/`. Voxy entry points implement
  `voxy_emitFragment` directly and do not include the general geometry pass.

## Estimator and GPU Rules

- Keep estimator, PDF, weight, MIS, and bias logic explicit. Do not hide estimator changes behind clamps or fallbacks.
- Prefer physically meaningful names and justified algorithms over heuristic patches.
- Prioritize mathematical correctness and GPU performance; deliberate redundancy is acceptable when it is faster.
- Never index `const` arrays or local variable arrays with runtime indices. Use shared memory, if-chains, or direct
  computation.
- Iris `workGroups` and `workGroupRender` values must be literals or macros expanding to literals. Select alternate
  values with preprocessor branches, not expressions such as `VOXEL_POOL_SIZE * 16` in the directive.

## Data Contracts

- Change shared declarations, formats and lifetimes, packing and coordinates, clears, producers, and every consumer as
  one contract.
- Keep temporal and feedback state aligned across current/previous transforms, jitter and motion, history resolution and
  encoding, resets, and filtering/confidence logic.
- Register new `SETTING_*` values in `scripts/options.main.kts`, including screen/profile placement and English and
  Chinese language entries, before using them in GLSL or `cond(...)`.

## Validation

Exercise relevant setting/profile branches, motion and disocclusion, history reset, edge conditions, and the
module-specific cases in `docs/en/modules/`.
