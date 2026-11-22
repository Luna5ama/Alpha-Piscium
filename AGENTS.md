# Working on Alpha Piscium

Start with the [project knowledge index](docs/README.md) for the question you are answering. Check which branch implements the behavior and where the decision was tested. Follow the [shader guide](shaders/AGENTS.md) for rendering and the [scripts guide](scripts/AGENTS.md) for registration, generation and packaging.

## Change the owner

Trace the active path from its registration through the producer and consumers. Change the maintained source and update its callers together. Remove replaced paths and unused options. Keep a compatibility path only for a required caller. Put derivable constraints in the existing code or generator and validate external inputs where they enter. Use the [topic record](docs/repository-structure.md) for the reason behind a choice, counterexamples and the conditions under which it holds.

Use the project’s [work publication index](VELA.md) for authorized work and its frozen specification. The [Vela configuration](.vela/project.toml) maps topic records to `docs/`.

## Verify the claim

- For a rendering change, verify the intended pack revision, Iris version, scene, preset and affected options loaded. Exercise the changed path on target Iris. For compatibility claims, check the target version's actual allocation or output with the feature enabled and disabled. Record what still needs visual or user acceptance.
- For GPU work, reuse comparable measurements. Prefer 720p for initial comparisons when it exercises the workload. Measure the affected passes and any stage that could inherit their cost. Use a stable sentinel and matched, reverse-order pairs when timing drifts. Respect reservations and finish a measurement group before handing over its instance.
- Check the configured game with `vibris_get_status` and reuse it when online. If Core is offline, `vibris_ensure_runtime` checks the target before launching it. See [Vibris startup](docs/development/vibris-startup.md). Restarting a running game or launcher requires an explicit user request.

## Deliver

Keep work in progress and other checkouts intact. Before committing, run `git diff --check` and `git diff --cached --check`. Record reusable results in the existing topic during authorized work. Keep large captures and logs in the run record and link the revision and evidence needed to reproduce the result. For authorized publication, use the target branch's PR workflow and read back the remote commit and entry points. Report what changed, what was removed, what was verified and where the result is available.
