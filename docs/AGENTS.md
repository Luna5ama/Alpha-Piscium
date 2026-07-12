# Documentation Instructions

Language: English | [简体中文](sc/AGENTS.md)

These instructions apply to documentation under [`docs/`](./).

## Scope

- Document Alpha Piscium-specific behavior only. Link to the Iris documentation instead of repeating generic Iris
  shader-pack syntax or behavior.
- Keep the documentation focused on maintainer workflows, project structure, generated configuration, and major
  rendering modules.
- Cover the commonly used build/generation tools in the workflow guide: Shadesmith, the program list generator, the
  options generator, and ZIP packaging. Keep the remaining utility scripts in the separate scripts document.
- Each major rendering module should briefly identify its code locations, pipeline integration, internal pass flow,
  resources, settings, and relevant limitations.

## Languages

- Maintain English documentation under [`docs/en/`](en/) and Simplified Chinese documentation under [`docs/sc/`](sc/).
- Keep paired documents structurally and semantically equivalent. A content or formatting change in one language
  normally requires the same change in the other.
- Put the language selector directly below every document title. List the current language first as plain text, then
  list every other available language as links; for example, `Language: English | [简体中文](...)` and
  `语言：简体中文 | [English](...)`.
- Keep the root English overview in [`README.md`](../README.md) and the Simplified Chinese overview in [
  `README.sc.md`](../README.sc.md).

## Links and References

- Link every concrete repository path, file, script, and shader pass so maintainers can jump directly to it.
- Check every inline-code span. If it names a real file or pass, make the code span the Markdown link text; leave
  settings, macros, resource names, offsets, and literal values as plain inline code.
- Use repository-relative Markdown links and verify that every local target exists.
- Refer to passes by their complete logical names and link them to their implementation files. Omit shader filename
  extensions from the visible link text for higher information density; do not use generated wrapper indices or
  truncated pass names.

## Workflow Formatting

- Present multi-stage pass flows as Markdown tables with order, pass/stage, and purpose columns.
- Use a vertical arrow flow only when it communicates the full high-level pipeline more clearly than a table.
- Put multi-command shell examples in fenced code blocks and specify the language for syntax highlighting.
- Run Kotlin generators from [`scripts/`](../scripts/):

  ```sh
  cd scripts
  kotlin programs.main.kts
  kotlin options.main.kts
  ```

- Run Shadesmith from [`scripts/`](../scripts/) as well:

  ```powershell
  cd scripts
  .\shadesmith.ps1
  ```

## Content Requirements

- The README files should contain a concise project-structure outline and links to the development quick start and
  detailed documentation.
- The quick-start documentation should explain the shortest normal development loop and which generator to run for each
  type of change.
- When similarly named configuration or generated files have different ownership, link each source and clearly
  distinguish its role.
- Present resources with different lifetimes, sizing rules, or persistence semantics as separate bullets or table rows.
- Put implementation details in the flow row for the pass or stage that performs them instead of separating them into
  detached prose.

## Validation and Commits

- Before committing, check paired-language parity, unresolved wrapper-number references, unlinked resolvable inline-code
  references, broken local Markdown links, and `git diff --check`.
- Commit each completed documentation round so review can proceed incrementally.
