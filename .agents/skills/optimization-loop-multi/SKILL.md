---
name: optimization-loop-multi
description: Run measured multi-agent performance optimization rounds for Alpha-Piscium GLSL, generated shaders, and related GPU hot paths across two or more user-specified Git worktrees, committing every verified win to one user-specified target branch. Use when asked to optimize, continue optimizing, benchmark candidate implementations, or work for a specified duration or number of rounds with parallel subagents. Stop immediately unless at least two worktree paths and one unambiguous target branch are supplied.
---

# Optimization Loop Multi

Read [references/alpha-piscium.md](references/alpha-piscium.md) for the small set
of repository-specific constraints.

## Scope and tracking

- Require at least two distinct worktree paths and one unambiguous target branch
  in the user's request. If either is missing, stop immediately without
  inspecting code, creating or choosing worktrees or branches, starting a goal,
  or spawning subagents.
- Verify the supplied paths are safe worktrees for the intended repository and
  the target branch exists. Do not substitute unspecified worktrees or choose a
  different branch; stop if fewer than two worktrees remain usable.
- Keep the target branch free from unrelated attached worktrees during the run.
  Supplied workers detach before working; stop rather than detach or move an
  unrelated checkout.
- Spawn exactly one persistent subagent for every usable supplied worktree and
  keep every worker detached; n worktrees mean n subagents. Do not
  create worker branches. The main agent does not consume or reserve a supplied
  worktree.
- Initialize each subagent with the absolute path to
  [worker-prompt.md](references/worker-prompt.md) plus its worktree, target
  branch, assigned optimization boundary, measurement contract, and coordinator
  handle. Do not rewrite the full worker workflow in each spawn message.
- Keep workers persistent for this optimization run: from the initiating user
  instruction through its requested boundary and final audit. Reuse them with
  follow-up tasks across rounds, then release them. A later instruction after
  completion starts a new run.
- If the user specifies a duration, call `create_goal` once for the entire run,
  track it with `get_goal`, and complete it only after the duration and final
  audit are satisfied. Set a token budget only when explicitly requested.
- If the user specifies a round count, run exactly that many rounds.
- Otherwise run one round.
- At the start of every round, require every worker to clean up its own prior
  candidate state and check out the current target-branch HEAD detached in its
  assigned worktree. Do this again even if the worker is already detached.
- A round is one synchronized search from the target branch. It ends after the
  scheduled workers report accepted, rejected, or blocked results and may
  produce multiple verified wins. Commit each win separately to the target
  branch.
- Keep the scheduler active in one long-running turn while the requested run is
  open; never use worker or round completion as a turn boundary.

## Optimization rule

- Keep the search open-ended; previous sessions are evidence about process, not
  limits on optimization ideas.
- The main agent only divides the target into clear enough worker boundaries,
  tracks the requested run, and collects reports. It does not choose candidates,
  maintain a separate accepted HEAD, implement or test changes, integrate
  commits, or resolve Git conflicts.
- Each subagent owns candidate selection, implementation, correctness testing,
  performance measurement, integration, conflict resolution, commit, and
  cleanup inside its assigned boundary and worktree. Reuse the same subagent
  across rounds.
- Keep worker boundaries disjoint where practical. Workers may continue
  independently when boundaries overlap unexpectedly; the publishing worker
  owns any resulting replay and conflict resolution.
- Serialize access to a shared live measurement runtime unless independently
  isolated runtimes are confirmed.
- Change one attributable candidate per worker and compare it with the latest
  accepted baseline under identical conditions. Require semantic correctness
  and repeatable improvement in the affected GPU pass, relevant sibling passes,
  and a stable workload sentinel.
- Remeasure small or noisy deltas. Reject and fully restore regressions, mixed
  results without a clear target-specific split, compile failures, missing
  metrics, or invalid comparisons.
- Treat the target branch as the only accepted baseline. Each published result
  advances it; workers reread it while publishing and check it out detached
  again at the start of the next round. Remember rejected ideas so a long loop
  does not repeat them.

## Commit rule

Completing a successful optimization authorizes its commit. The responsible
subagent must replay the verified change onto the current target-branch HEAD in
detached state, resolve conflicts, rerun the correctness and performance gates,
and create one commit containing only that optimization. It must then advance
the specified target branch with a compare-and-swap fast-forward from the exact
HEAD it integrated against. If another worker advanced the branch first, replay
onto the new HEAD, resolve conflicts, rerun the gates, and retry. Never force or
overwrite the target branch, and do not wait for a separate user request.
Rejected or blocked work produces no commit. When generator and generated
outputs live in different repositories, create one atomic commit in each on the
user-specified target branch.

## Runtime boundary

Use Vibris for normal live measurement. Verify the active runtime/configuration
instead of relying on stale tool assumptions. Never restart Minecraft or its
launcher unless explicitly requested. If safe non-restart recovery cannot make
Vibris usable, stop and report the blocker; do not silently switch to replay or
another runtime unless the user authorized that fallback.

## Finish

Run a final comparable correctness/performance audit from the target branch,
collect or clean up in-flight work, confirm every accepted commit is reachable
from that branch, and leave worker worktrees detached with unrelated state
untouched. Report the final branch tip, accepted commits, rejected candidates,
measurement conditions, absolute and relative timing changes, conflicts
resolved, and any blocker. Do not close a timed goal before its requested
duration has elapsed.
