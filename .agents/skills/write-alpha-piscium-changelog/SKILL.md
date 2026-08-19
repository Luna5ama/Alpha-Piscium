---
name: write-alpha-piscium-changelog
description: Draft, revise, translate, and finalize Alpha Piscium stable, beta, alpha, and hotfix changelogs from Git history. Use when asked to prepare release notes, merge earlier prerelease changelogs, write an English review draft, apply editorial feedback, or generate the matching Simplified Chinese `.sc.md` file.
---

# Write Alpha Piscium Changelog

Read `agent_inputs/prompts/prompt_changelog.txt` before starting. Follow this skill for release boundaries, output paths, and the review workflow. Let explicit user instructions override defaults.

## Resolve the release

Require the target version. Accept optional explicit target and `since` revisions; explicit revisions always win.

Classify the version case-insensitively:

- Treat a version containing `Alpha` as alpha.
- Treat a version containing `Beta` as beta.
- Treat every other version as stable, including `Hotfix`.

Name alpha versions canonically as `X.Y.Z-Alpha-<sha8>`:

- Require the user to supply the `X.Y.Z` release stem; accept either `X.Y.Z-Alpha` or the full canonical version as input.
- Resolve the target commit first, then set `<sha8>` to the first eight lowercase hexadecimal characters of the full target commit ID.
- The suffix always identifies the target commit. Never use the base or `since` commit in the alpha version name.
- Normalize the output label to `Alpha`. When the input ends at `X.Y.Z-Alpha`, append the computed target suffix automatically.
- When an input suffix does not match the resolved target, stop before writing files, report both commits, and ask to use the canonical name. Do not silently preserve or rewrite a mismatched full version.

For example, if the range is `16fb7f14..HEAD` and `HEAD` resolves to `bc8f4db9...`, use `1.10.0-Alpha-bc8f4db9`, not `1.10.0-Alpha-16fb7f14`.

Resolve the target in this order:

1. Use an explicit target revision.
2. Use an existing `v<version>` or `<version>` tag.
3. For alpha, use the trailing canonical eight-character hexadecimal commit suffix.
4. For an untagged stable or beta draft, use `HEAD`.

If alpha has no explicit target, resolvable tag, or commit suffix, stop and ask for the target revision.

Resolve the base as follows:

- Use an explicit `since` revision when supplied.
- For stable, select the nearest reachable tag whose name contains neither `Alpha` nor `Beta`. This includes hotfix tags.
- For beta, select the nearest reachable stable or beta tag and exclude alpha tags.
- For alpha, require an explicit `since` version or revision. Do not infer it from alpha files or tag dates.

For automatic selection, consider only tags whose peeled commits are ancestors of the target, exclude a tag resolving to the target itself, and choose the candidate with the smallest positive commit count in `candidate..target`. Do not choose by version-string order, tag date, or commit timestamp.

Resolve version-like revisions by trying the value as written, its optional `v`-prefixed form, and any trailing commit suffix. Verify both revisions with `git rev-parse`, then require `git merge-base --is-ancestor <base> <target>` to succeed. Use the exact range `<base>..<target>`: exclude the base, include the target, and exclude uncommitted worktree changes.

Use `git --no-pager` for commands that may page.

## Select output files

Use these paths:

- Stable, beta, and hotfix English: `changelogs/<version>.md`
- Alpha English: `changelogs/alpha/<version>.md`
- Simplified Chinese: insert `.sc` before `.md` beside the English file, for example `changelogs/1.0.0.sc.md`

Create `changelogs/alpha/` when needed. Preserve the existing rule that ignores this directory; do not edit `changelogs/.gitignore`.

Use this title in both languages:

```markdown
# Alpha Piscium v<version>
```

## Gather evidence

Inspect all of the following before drafting:

- The reverse chronological release range with commit subjects and bodies
- The net diff stat and changed paths for `<base>..<target>`
- Patches for ambiguous or user-visible commits
- Existing English prerelease changelogs whose resolvable release commits fall inside the range
- Changelog text supplied by the user

Treat earlier changelogs as editorial input, not ground truth. Verify every retained claim against the final target. Do not use Chinese files as source material for the English draft.

Do not let an unrelated dirty worktree block history-based drafting. Preserve all unrelated changes and edit only the requested changelog files.

## Edit the release notes

Use this section order:

1. `Highlight`
2. `New`
3. `Improvement`
4. `Fix`
5. `Misc`

Omit empty sections instead of adding placeholders.

Write concise, user-readable final outcomes. Preserve necessary product names, setting names, and code identifiers. Avoid raw commit-by-commit narration.

When the range contains intermediate alpha or beta releases:

- Merge duplicate bullets.
- Fold a fix for a feature introduced in the same range into that feature's final description.
- Omit intermediate fixes that no longer describe a separate user-visible outcome.
- Omit features removed before the target.
- Describe the target's final behavior, not the sequence of failed or superseded implementations.

## Run the review workflow

### First round

Write only the English file. Do not create or update the Simplified Chinese file.

After writing:

1. Run `git diff --check -- <english-file>`.
2. Read the file back to verify its content.
3. Report the resolved base, target, and English path.
4. Stop and let the user review it.

If the English file already exists and the user did not clearly request regeneration, revision, translation, or finalization, read it and ask which review phase it is in rather than overwriting possible manual edits.

### Editorial iterations

When the user explicitly requests additions, deletions, rewrites, or continued iteration, update only the English file, validate it, and stop for another review. Do not create or refresh the Chinese file during an editorial iteration unless the user explicitly overrides the review gate.

### Final round

When the user approves, asks to finalize or translate, or says they edited the English file without requesting another editorial iteration:

1. Read the current English file from disk and treat it as the only source of truth.
2. Do not regenerate or silently rewrite its content from Git history.
3. Write or update the matching `.sc.md` file.
4. Preserve title, section order, bullet order, bullet count, Markdown, product names, setting names, and code identifiers.
5. Translate headings as `重点`, `新增`, `改进`, `修复`, and `杂项`.
6. Run `git diff --check -- <english-file> <chinese-file>` and read both files back.
7. Verify that corresponding sections and bullet counts match.

## Stop on invalid state

Stop with one precise question when:

- A required version, alpha base, or alpha target is missing.
- A revision cannot be resolved.
- The base is not an ancestor of the target.
- An alpha version suffix does not match the resolved target commit.
- Automatic stable or beta base selection has no eligible tag.
- An existing English file has an ambiguous review state.

Do not run Shadesmith, options/program generators, builds, or shader validation for changelog-only work.

## Report completion

For an English draft, report the range and English path, then state that review is required. For a final round, report both file paths and the validation performed. Mention that no code or old implementation was changed.
