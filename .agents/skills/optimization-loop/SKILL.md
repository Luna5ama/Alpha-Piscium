---
name: optimization-loop
description: Run measured performance optimization rounds for Alpha-Piscium GLSL, generated shaders, and related GPU hot paths. Use when asked to optimize, continue optimizing, benchmark candidate implementations, or work for a specified duration or number of rounds. Keep only verified wins and automatically commit every successful optimization round.
---

# Optimization Loop

Read [references/alpha-piscium.md](references/alpha-piscium.md) for the small set
of repository-specific constraints.

## Scope and tracking

- If the user specifies a duration, call `create_goal` with that duration and
  target, track it with `get_goal`, and complete it only after the duration and
  final audit are satisfied. Set a token budget only when explicitly requested.
- If the user specifies a round count, run exactly that many rounds.
- Otherwise run one round.
- A round is a bounded search on one target. It ends with one verified win, no
  defensible win, or a blocker. It may include multiple rejected candidates.

## Optimization rule

- Keep the search open-ended; previous sessions are evidence about process, not
  limits on optimization ideas.
- Change one attributable candidate at a time and compare it with the latest
  accepted baseline under identical conditions.
- Require both semantic correctness and repeatable improvement in the affected
  GPU pass. Watch relevant sibling passes and a stable workload sentinel.
- Remeasure small or noisy deltas. Reject and fully restore regressions, mixed
  results without a clear target-specific split, compile failures, missing
  metrics, or invalid comparisons.
- Use each accepted result as the next baseline and remember rejected ideas so
  a long loop does not repeat them.

## Commit rule

Completing a successful round authorizes its commit. After the correctness and
performance gates pass, stage only that optimization and commit it immediately;
do not wait for a separate user request. A failed round produces no commit.
When generator and generated outputs live in different repositories, create one
atomic commit in each.

## Runtime boundary

Use Vibris for normal live measurement. Verify the active runtime/configuration
instead of relying on stale tool assumptions. Never restart Minecraft or its
launcher unless explicitly requested. If safe non-restart recovery cannot make
Vibris usable, stop and report the blocker; do not silently switch to replay or
another runtime unless the user authorized that fallback.

## Finish

Run a final comparable correctness/performance audit, confirm unrelated work is
untouched, and report accepted commits, rejected candidates, measurement
conditions, absolute and relative timing changes, and any blocker. Do not close
a timed goal before its requested duration has elapsed.
