---
name: optimization-loop-multi
description: Run measured multi-agent performance optimization rounds for Alpha-Piscium GLSL, generated shaders, and related GPU hot paths across two or more user-specified Git worktrees. Use when asked to optimize, continue optimizing, benchmark candidate implementations, or work for a specified duration or number of rounds with parallel subagents. Keep only verified wins and automatically commit every successful optimization round. Stop immediately unless at least two worktree paths are supplied.
---

# Optimization Loop Multi

Read [references/alpha-piscium.md](references/alpha-piscium.md) for the small set
of repository-specific constraints.

## Scope and tracking

- Require at least two distinct worktree paths in the user's request. If fewer
  are supplied, stop immediately without inspecting code, creating or choosing
  worktrees, starting a goal, or spawning subagents.
- Verify the supplied paths are safe worktrees for the intended repository. Do
  not substitute unspecified worktrees; stop if fewer than two remain usable.
- Spawn exactly one persistent subagent for every usable supplied worktree and
  give each a distinct worker branch; three worktrees mean three subagents. The
  main agent only schedules and does not consume or reserve a supplied
  worktree.
- Initialize each subagent with the absolute path to
  [worker-prompt.md](references/worker-prompt.md) plus its worktree, branch,
  accepted HEAD, target, measurement contract, and coordinator handle. Do not
  rewrite the full worker workflow in each spawn message.
- Keep workers persistent for this optimization run: from the initiating user
  instruction through its requested boundary and final audit. Reuse them with
  follow-up tasks across rounds, then release them. A later instruction after
  completion starts a new run.
- If the user specifies a duration, call `create_goal` once for the entire run,
  track it with `get_goal`, and complete it only after the duration and final
  audit are satisfied. Set a token budget only when explicitly requested.
- If the user specifies a round count, run exactly that many rounds.
- Otherwise run one round.
- A round is a coordinated search on one accepted HEAD. It ends with one
  verified win, no defensible win, or a blocker and may include multiple
  concurrent or rejected candidates.
- Keep the scheduler active in one long-running turn while the requested run is
  open; never use worker or round completion as a turn boundary.

## Optimization rule

- Keep the search open-ended; previous sessions are evidence about process, not
  limits on optimization ideas.
- The main agent coordinates the accepted HEAD, candidates, assignments,
  measurement contract, and likely conflicts. It analyzes reports but does not
  implement or test candidates.
- Each subagent owns exploration, implementation, correctness testing,
  performance measurement, conflict resolution, and cleanup in its assigned
  worktree. Reuse the same subagent across rounds.
- Start each round from the latest accepted HEAD. Have workers independently
  report one or more candidates with expected value, scope, conflict footprint,
  and validation plan before implementation.
- Choose the most valuable defensible candidate at each scheduling decision.
  Keep non-overlapping work in flight and serialize likely conflicts; idle
  workers may continue exploring.
- Serialize access to a shared live measurement runtime unless independently
  isolated runtimes are confirmed.
- Change one attributable candidate per worker and compare it with the latest
  accepted baseline under identical conditions. Require semantic correctness
  and repeatable improvement in the affected GPU pass, relevant sibling passes,
  and a stable workload sentinel.
- Remeasure small or noisy deltas. Reject and fully restore regressions, mixed
  results without a clear target-specific split, compile failures, missing
  metrics, or invalid comparisons.
- Use each accepted result as the next baseline, notify other workers that the
  HEAD changed, and remember rejected ideas so a long loop does not repeat them.

## Commit rule

Completing a successful round authorizes its commit. After the correctness and
performance gates pass, the responsible subagent must place the optimization on
its worker branch at the latest accepted HEAD, resolve any Git conflicts, rerun
the gates, and commit it immediately; do not wait for a separate user request.
Accept concurrent completions one at a time and advance the accepted HEAD after
each commit. A failed round produces no commit. When generator and generated
outputs live in different repositories, create one atomic commit in each.

## Runtime boundary

Use Vibris for normal live measurement. Verify the active runtime/configuration
instead of relying on stale tool assumptions. Never restart Minecraft or its
launcher unless explicitly requested. If safe non-restart recovery cannot make
Vibris usable, stop and report the blocker; do not silently switch to replay or
another runtime unless the user authorized that fallback.

## Finish

Run a final comparable correctness/performance audit, collect or clean up
in-flight work, confirm unrelated worktree state is untouched, and report
accepted commits, rejected candidates, measurement conditions, absolute and
relative timing changes, conflicts resolved, and any blocker. Do not close a
timed goal before its requested duration has elapsed.
