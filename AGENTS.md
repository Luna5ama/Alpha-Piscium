# Coding Instructions

Write clean, direct, maintainer-style code.

Prefer:
- minimal diffs
- simple control flow
- deleting obsolete code
- replacing old paths cleanly
- explicit logic over generic abstractions
- targeted fixes over broad frameworks

Avoid:
- overengineering
- speculative abstractions
- defensive checks for impossible states
- fallback paths that hide bugs
- preserving legacy behavior unless explicitly required
- compatibility wrappers during refactors
- feature flags/options with only one real use
- verbose comments explaining obvious code

## Refactoring

When a new implementation replaces an old one, remove the old code.

Do not keep both old and new paths unless compatibility is explicitly required.

Update call sites directly instead of adding adapters or aliases.

Delete dead code, stale helpers, obsolete comments, and unused settings in the same change.

## Defensive Code

Only validate real trust boundaries: user input, files, external APIs, GPU/driver output, or serialized data.

Do not add null checks, bounds checks, fallbacks, or catch-all handling for states that should be impossible by construction.

Prefer failing visibly over silently masking invalid state.

## Abstractions

Do not add a new abstraction unless it has at least two real uses now.

Prefer plain functions and direct data flow over managers, factories, registries, services, or generic pipelines.

## Graphics / Shader Code

For rendering, GI, ReSTIR, denoising, atmosphere, and shader code:

- keep estimator, PDF, weight, and bias logic explicit
- do not hide estimator changes behind clamps or fallbacks
- avoid heuristic fixes unless clearly justified
- prefer physically meaningful names
- prioritize mathematical correctness over defensive engineering

## Final Response

Summarize:
- what changed
- what old code was removed
- what tests/checks were run, or why not