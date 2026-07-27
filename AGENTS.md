# PROJECT KNOWLEDGE BASE

**Generated:** 2026-07-19
**Commit:** 2fa383c3
**Branch:** dev

## Overview

Alpha Piscium is an Iris-targeted Minecraft shader pack implemented in GLSL, with Kotlin and PowerShell tooling for
program registration, options, generated bindings, packaging, and offline assets.

## Structure

```text
shaders/
├─ Base.glsl       common include root
├─ base/           shared interfaces and generated bindings
├─ pass/           standard home for shader entry points
├─ techniques/     reusable rendering modules and existing direct compute entries
├─ util/           cross-module GLSL helpers
└─ textures/       runtime LUT, noise, sky, cloud, and sampling assets
scripts/           generators, packaging, release, and offline asset tools
docs/              bilingual maintainer workflows and rendering-module maps
```

## Where to Look

| Task | Location | Notes |
|------|----------|-------|
| Compute pass order and enable conditions | `scripts/programs.main.kts` | Source of truth for numbered compute wrappers |
| Settings, profiles, screens, localization | `scripts/options.main.kts` | Uses the DSL in `scripts/options.lib.kts` |
| Iris properties and custom resources | `scripts/shaders.properties` | Hand-maintained source, unlike the generated shader-root file |
| Textile tile layout | `shaders/shadesmith.json` | Owns generated `shaders/base/Textile.glsl` bindings |
| Shader entry points | `shaders/pass/` | Grouped by setup, begin, geometry, shadowcomp, composite, and general |
| Shared rendering implementations | `shaders/techniques/` | GI, atmospherics, RTWSM, display transform, FFX, and other modules |
| Cross-module GLSL helpers | `shaders/util/` | Math, coordinates, sampling, material, packing, and color helpers |
| Pipeline and module contracts | `docs/en/modules/` | Pass flows, resources, settings, limitations, and validation targets |

## Code Map

| Symbol / entry | Location | Reach | Role |
|----------------|----------|-------|------|
| `Base.glsl` | `shaders/Base.glsl` | 24 direct include sites | Common compatibility, binding, and option interface root |
| `Rand.glsl` / `Coords.glsl` / `Math.glsl` | `shaders/util/` | 35 / 36 / 41 include sites | Highest-fan-in shared helpers |
| `ProgramScope.pass` | `scripts/programs.main.kts` | All generated compute wrappers | Records order, grouping, defines, conditions, and indirect dispatch |
| `OptionBuilder` / `ScreenBuilder` | `scripts/options.lib.kts` | Settings, profiles, screens, languages | Builds option GLSL, localization, and final properties |
| `main` entry families | `shaders/pass/` and `shaders/techniques/` | 53 standard / 23 existing direct entries | Runtime pass entry points |

## Project Policy

- Write clean, direct maintainer code with minimal diffs, simple control flow, explicit logic, and compact formatting.
  Never add spacing only to align declarations or operators.
- When replacing a path, update callers directly and delete the old path, dead helpers, stale comments, and unused
  settings. Do not preserve legacy behavior, parallel implementations, adapters, aliases, or compatibility wrappers
  unless explicitly required.
- Do not add an abstraction without two real uses. Avoid speculative managers, factories, registries, services, generic
  frameworks, and feature flags or options with one use.
- Validate only trust boundaries: user input, files, external APIs, GPU/driver output, and serialized data. Do not add
  null/bounds checks, catch-all handling, or fallbacks for impossible states; fail visibly.
- Prefer targeted fixes over broad frameworks. Do not add verbose comments that explain obvious code.

## Scoped Instructions

- Shader implementation and placement rules: `shaders/AGENTS.md`.
- Generator, packaging, and release rules: `scripts/AGENTS.md`.
- Documentation and language-parity rules: `docs/AGENTS.md`.

## Generated Files and Commands

Run generators from `scripts/`. Ordinary `.glsl` edits require no generator because wrappers include source at runtime.

| Maintained source or task | Command from `scripts/` | Result |
|---------------------------|--------------------------|--------|
| `programs.main.kts` | `kotlin options.main.kts` | Reruns programs, then aggregates wrappers and final properties |
| `options.main.kts`, `options.lib.kts`, `shaders.properties`, `sponsors.txt` | `kotlin options.main.kts` | Rebuilds options, languages, wrappers, and final properties |
| `../shaders/shadesmith.json` | `.\shadesmith.ps1` then `kotlin options.main.kts` | Rebuilds Textile/image bindings and final properties |
| Test package | `kotlin make-zip.main.kts [version] [--no-commit-hash]` | Writes an ignored package under `../builds/` |

On a fresh checkout, run Shadesmith once before the first options generation so the ignored image/atlas fragment exists.
Except for that bootstrap and ZIP packaging, run Shadesmith only when `shaders/shadesmith.json` changes.
Use `programs.main.kts` alone only for a quick wrapper-numbering preview; `options.main.kts` is the shortest complete
generation because it invokes the program generator.

Do not edit tracked generated wrappers, `shaders/base/{Options,TextOptions,Textile}.glsl`, `shaders/lang/*.lang`, or
`shaders/shaders.properties` directly. Do not commit ignored `scripts/*.shaders.properties` fragments, `shadesmitth/`,
or AOT caches.

## Validation

- Keep routine QA brief; the user performs manual acceptance. Do deeper review or validation only when explicitly requested in the current turn.
- Vibris is profiling-only and is not a screenshot, debugging, or correctness-validation surface.
- Run `git diff --check` and `git diff --cached --check` before committing.
- Keep tracked generated outputs in the same change as their maintained source.
- Prefer IDE diagnostics. Use external validators such as `glslangValidator` only when explicitly requested or when IDE validation is unavailable.
- No automated command replaces target-Iris compilation and visual validation. Check affected settings and profile
  branches, history resets, and representative scenes.

## Reference Sources

Check the maintained project documentation before external sources:

- Architecture and workflow: `docs/en/development/`
- Pipeline and rendering-module contracts: `docs/en/modules/`
- Generic Iris behavior: use the upstream documentation linked from `docs/en/README.md` instead of restating it here.

When present in the working environment, check repository-local references before external sources:

- CG and rendering references: `agent_inputs/cg-resources/`
- Iris shader documentation: `.agents/iris-docs/`

## Final Response

Summarize:
- what changed
- what old code was removed
- what tests/checks were run, or why not
