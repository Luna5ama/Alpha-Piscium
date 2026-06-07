# Coding Instructions

Write clean, direct, maintainer-style code.

Prefer:
- minimal diffs
- simple control flow
- deleting obsolete code
- replacing old paths cleanly
- explicit logic over generic abstractions
- targeted fixes over broad frameworks
- compact formatting without alignment padding

Avoid:
- overengineering
- speculative abstractions
- defensive checks for impossible states
- fallback paths that hide bugs
- preserving legacy behavior unless explicitly required
- compatibility wrappers during refactors
- feature flags/options with only one real use
- verbose comments explaining obvious code

Do not add extra spaces only to align variable names, type declarations, or operators across multiple lines.

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
- prioritize performance and optimization, even when deliberate redundancy is faster
- never index `const` arrays or local variable arrays with runtime indices in GLSL; use shared memory, if-chains, or direct computation instead
- keep Iris `workGroups` and `workGroupRender` values as constant literals or macros that expand to constant literals
- use preprocessor conditionals for alternate Iris work group values; do not use math expressions such as `VOXEL_POOL_SIZE * 16` in the directive

## Generated Shader Files

Only run Shadesmith when `shaders/shadesmith.json` changes. Changes to `.glsl` files do not require a Shadesmith rebuild because `.csh` files include them at runtime.

Run Shadesmith from the repository root with:

```powershell
.\scripts\shadesmith.ps1
```

When adding a transient, persistent, or history texture tile, update `shaders/shadesmith.json` and run `.\scripts\shadesmith.ps1` to update the Textile macro.

Register every new `#ifdef SETTING_*` in `scripts/options.main.kts` before using it in GLSL, then regenerate options:

```sh
cd scripts
kotlin options.main.kts
```

Run the programs DSL after changing pass order, includes, compile-time defines, or pass enable conditions in `scripts/programs.main.kts`:

```sh
cd scripts
kotlin programs.main.kts
```

`programs.main.kts` generates pass wrapper `.csh` files and `scripts/programs.shaders.properties`. Use `pass("/path/to/shader")` for single passes, `pass("...", "...")` for grouped suffixed outputs, and block form for `define`, `constDefine`, and `cond`.

## Validation

Prefer IDE diagnostics for shader and code validation. Use external validators such as `glslangValidator` only when explicitly requested or when IDE validation is unavailable.

## Reference Sources

Check repository-local references before external sources:

- CG and rendering references: `agent_inputs/cg-resources/`
- Iris shader documentation: `agent_inputs/iris-docs/`

## Final Response

Summarize:
- what changed
- what old code was removed
- what tests/checks were run, or why not
