# GENERATOR AND TOOLING INSTRUCTIONS

## Overview

This directory owns program registration, option generation, property aggregation, Shadesmith integration, packaging,
release automation, and offline asset generation.

## Where to Look

| Source | Ownership |
|--------|-----------|
| `programs.main.kts` | Compute order, grouped wrappers, conditions, defines, and indirect dispatch |
| `options.main.kts` | Settings, profiles, screens, translations, and version selection |
| `options.lib.kts` | Options DSL implementation and final property aggregation |
| `shaders.properties` | Hand-maintained textures, images, SSBOs, blending, and uniforms |
| `shadesmith.ps1` / `shadesmith.jar` | Textile preprocessing and image/atlas property generation |
| `make-zip.main.kts` | Test/release ZIP assembly |
| `release.main.kts` | Branch, tag, push, upload, and announcement workflow |

## Generator Rules

- Kotlin entrypoints use relative paths and must be launched from this directory. `shadesmith.ps1` self-locates through
  `$PSScriptRoot` but should use the same documented working directory.
- `pass(pathA, pathB, ...)` emits same-number suffix wrappers. With concurrent compute enabled, grouped entries must be
  independent; never use suffix letters as an execution order.
- Keep `define`, `constDefine`, `cond`, and `indirect` decisions explicit in the program declaration that owns the pass.
- When removing or replacing a pass, remove the obsolete entry source and unused settings; let the generator replace
  wrappers and property branches.

## Local and Ignored State

- Do not edit `programs.shaders.properties` or `shadesmith.shaders.properties`; both are ignored intermediates.
- Local `config.properties` supports `JAVA_PATH` and `SHADESMITH_OUTPUT`. Keep it machine-local.
- `tokens.properties` contains release credentials. Never commit it.

## High-Side-Effect Tools

- `release.main.kts` switches branches, creates and pushes tags, and calls publishing APIs. Run it only when the user
  explicitly requests a release and the checkout is the exact content to publish.
- Offline LUT/texture generators are not part of the normal GLSL loop. When changing their outputs, verify binary
  dimensions/layout and synchronize `customTexture.*` declarations, sampler types, channel assumptions, and shader
  constants.

## Validation

Review generated diffs for only the expected wrappers, option bindings, languages, properties, or Textile changes.
