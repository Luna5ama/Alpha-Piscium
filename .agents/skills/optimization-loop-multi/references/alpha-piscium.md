# Alpha-Piscium Constraints

- Follow the active checkout's root and scoped `AGENTS.md` files.
- Edit maintained sources. When Shadesmith or another generator owns the result,
  regenerate and validate its tracked shader/binary outputs and contracts before
  measuring or committing.
- Measure only passes that execute the changed path, plus a workload sentinel;
  do not substitute whole-pipeline time unless requested.
- Keep scene, shader config, warmup, sample count, statistic, and source ordering
  comparable. Use adjacent/reverse pairs or repeated rounds when noise matters.
- Verify every runtime load and reject measurements from the wrong pack, failed
  compilation, missing pass, or changed scene.
- Write commit subjects as concise English imperative sentences that name the
  optimization directly, matching the wording and capitalization of recent
  first-parent commits on the receiving branch; for example,
  `Optimize initial AABB X slab rejection`.
- Preserve unrelated worktrees and local artifacts such as `.vibris/`, captures,
  screenshots, replay caches, and debug files.
