# Working on Alpha Piscium

Start with the [project knowledge index](docs/README.md) for the question you are answering. Check which branch implements the behavior and where the decision was tested. Follow the [shader guide](shaders/AGENTS.md) for rendering and the [scripts guide](scripts/AGENTS.md) for registration, generation and packaging.

## Change the owner

Trace the active path from its registration through the producer and consumers. Update the maintained source and its callers together. Extract a shared abstraction when two real uses need it. Remove replaced paths and unused options. Keep a compatibility path only for a required caller. Put derivable constraints in the existing code or generator and validate external inputs where they enter. Use the [topic record](docs/repository-structure.md) for the reason behind a choice, counterexamples and the conditions under which it holds.

Use the project’s [work publication index](VELA.md) for authorized work and its frozen specification. The [Vela configuration](.vela/project.toml) maps topic records to `docs/`.

## Verify the claim

- For a rendering change, verify the intended pack revision, Iris version, scene, preset and affected options loaded. Exercise the changed path on target Iris. For compatibility claims, check the target version's actual allocation or output with the feature enabled and disabled. Keep routine QA brief and leave final visual acceptance to the user.
- For open performance work, use measured cost to choose the algorithm, sampling strategy or pass split. Compare image quality and GPU time on the same scene and settings. Reuse comparable measurements. Prefer 720p for initial comparisons when it exercises the workload. Measure the affected passes and any stage that could inherit their cost. Use a stable sentinel and matched, reverse-order pairs when timing drifts. Respect reservations and finish a measurement group before handing over its instance. Use Nsight for a deep GPU hotspot that Vibris timings cannot explain.
- Check the configured game with `vibris_get_status` and reuse it when online. If Core is offline, `vibris_ensure_runtime` checks the target before launching it. See [Vibris startup](docs/development/vibris-startup.md). Restarting a running game or launcher requires an explicit user request.

## Deliver

Keep work in progress and other checkouts intact. Before committing, run `git diff --check` and `git diff --cached --check`. Store raw captures and full logs in an ignored run directory or the project artifact store. In the existing Git topic, record the conclusion, conditions, source revision and a retrievable evidence location. Use the [version publication rules](docs/repository-structure.md#版本与发布分支) to select the target. For authorized publication, merge the approved changes locally into the named target, run the affected checks, and push that branch directly. Read back the remote commit and entry points. If the remote rejects the push, report the exact rule to the maintainer. Open a pull request only on explicit user request. Report what changed, what was removed, what was verified and where the result is available.
