# Four-hour GI tracing measurements, November 27–28, 2026

The baseline is `8ce1cf34c`. The measured release shader revision is `fd0ff43c6`. The continuous window ran from 20:31:31 to 00:31:31 Pacific with both gpu:0 and game:vibris. Dedicated Vibris used PID 52676, Iris 1.10.6-vibris.3+mc1.21.11, an RTX 3080 Ti and NVIDIA 617.42.

## GPU time

Each row combines an ABBA group and an independently reloaded BAAB group at the same Ultra preset. Each arm warmed for 144 frames and measured six eight-frame bursts separated by 64 normal frames. The table averages the four arm medians for each package. Each median comes from the sum of native timers on the same frame. A positive difference is a saving.

| Scene at 720p | Baseline ms | Candidate ms | Saved ms | Four adjacent pair savings in ms |
|---|---:|---:|---:|---|
| basilica-1-720p | 7.7490 | 7.5754 | 0.1736 | 0.1792 / 0.1710 / 0.1843 / 0.1597 |
| non-cube-1-720p | 9.4762 | 9.1726 | 0.3036 | 0.3133 / 0.3062 / 0.2985 / 0.2964 |
| night-gi-1-720p | 9.3092 | 9.0964 | 0.2127 | 0.1884 / 0.2115 / 0.2212 / 0.2299 |
| mirror-room-1-720p | 4.3693 | 4.2721 | 0.0972 | 0.0927 / 0.0927 / 0.0947 / 0.1085 |

| Scene | Initial trace C21 | Cache C17 | Spatial visibility C30 | Terrain shadow | Legacy 29 | With shared consumers |
|---|---:|---:|---:|---:|---:|---:|
| basilica-1-720p | 0.0709 | 0.0138 | 0.0054 | 0.0872 | 0.0858 | 0.1720 |
| non-cube-1-720p | 0.0612 | 0.0116 | 0.0061 | 0.2239 | 0.0772 | 0.3035 |
| night-gi-1-720p | 0.0335 | 0.0095 | 0.0081 | 0.1687 | 0.0471 | 0.2124 |
| mirror-room-1-720p | 0.0837 | 0.0024 | -0.0008 | 0.0096 | 0.0858 | 0.0966 |

| Scene | Face/voxel clear | Global update | Tree prepare2 | Probe stages | Shared Hi-Z | Shadow solid | Shadow translucent |
|---|---:|---:|---:|---:|---:|---:|---:|
| basilica-1-720p | -0.0010 | 0.0000 | 0.0013 | 0.0006 | 0.0003 | 0.0860 | 0.0001 |
| non-cube-1-720p | -0.0013 | 0.0003 | -0.0012 | 0.0003 | 0.0006 | 0.2239 | 0.0004 |
| night-gi-1-720p | -0.0015 | 0.0000 | 0.0020 | 0.0000 | -0.0006 | 0.1676 | 0.0009 |
| mirror-room-1-720p | -0.0010 | 0.0000 | 0.0008 | 0.0006 | 0.0004 | 0.0097 | 0.0001 |

| Scene | Enclosing composite A / B | Terrain sentinel min–max across eight arms |
|---|---:|---:|
| basilica-1-720p | 48.5245 / 48.1494 | 0.3492 / 0.3523 |
| non-cube-1-720p | 49.0625 / 49.0689 | 0.3451 / 0.3482 |
| night-gi-1-720p | 49.9283 / 48.8704 | 0.3302 / 0.3328 |
| mirror-room-1-720p | 52.3476 / 49.8030 | 0.1546 / 0.1556 |

The primary total includes the established 29 GI stages, six environment-probe stages, shared Hi-Z, begin1 and both terrain-shadow draws. The collector also includes begin12 and the registered C16/C17/C19/C20/C21 helper dispatches when they emit a timer. The shared-consumer total also includes C48, C51, C71_a and C75. The enclosing composite timer overlaps GI and other effects. The independent terrain sentinel and the 64-frame normal wall intervals help identify drift. End-to-end GPU frame time was not captured.

| Scene | Positive paired bursts / total | Largest external 3D engine per group | Normal-frame median paired wall difference (ms) |
|---|---:|---|---:|
| basilica-1-720p | 24 / 24 | OxygenNotIncluded 15.8% / OxygenNotIncluded 15.4% | 0.2497 / 0.1049 |
| non-cube-1-720p | 24 / 24 | OxygenNotIncluded 16.6% / OxygenNotIncluded 15.3% | 0.0345 / 0.1832 |
| night-gi-1-720p | 24 / 24 | OxygenNotIncluded 14.9% / OxygenNotIncluded 15.9% | 0.2398 / 0.3648 |
| mirror-room-1-720p | 24 / 24 | OxygenNotIncluded 16.7% / OxygenNotIncluded 17.9% | 0.6152 / 0.1359 |

Core recorded MANAGED with Realtime 5 on entry and Idle 0 after each default comparison. The raw receipts retain every native UUID, each arm’s source, scene and effective options, external-engine samples, and paired burst distributions.

## Additional inputs

- final-time-spatial-off ran in BAAB order at 8.4808 / 8.1772 ms. Adjacent savings were 0.3062 and 0.3011 ms.
- final-all-programs ran in ABBA order at 9.2672 / 9.0537 ms. Adjacent savings were 0.2263 and 0.2007 ms.
- The unfiltered night inventory covered 89–89 named compute programs per arm. All-compute paired savings were 0.0855 / 0.0635 ms.
- With spatial reuse off, the unfiltered non-cube capture covered 87–87 named compute programs. Summing each program once per frame saved 0.0625 / 0.0763 ms in the two adjacent comparisons.

## Occlusion, radiance and motion

The same-input comparison reran the baseline packaged C21 on live candidate inputs. Across 13,824,000 pixels, 0 differed in geometry or state and 0 differed in stored RGB. The largest per-capture relative RGB P99 was 0.000%, maximum 0.000% and pooled fraction over 1% 0.000%. The FP16 linear RGB denominator was max(reference RGB length, 1e−6).

At C17, 173,601 active stored cache reservoirs were compared. Geometry or metadata, weight or M, and radiance or estimate mismatch counts were 0, 0, 0.

The packaged baseline trace ran at 256 steps for each of 7,110,380 rays. The comparison covered material, distance, normal, hit position and continuation state. 0 rays changed, with an 8192-step reference available for every mismatch. The baseline itself exhausted on 1,503 rays.

The camera path translated 1.5 blocks, turned 14 degrees and returned across nine frames. Settled images followed 48 further frames. The table compares normal display RGB with the independently reloaded baseline. Each cell lists A/B followed by the A/A repeat. The display denominator was max(reference RGB length, 0.01). The motion rows pool all captured motion pixels.

| Input / interval | P99 | Maximum | Fraction over 1% | Mean absolute RGB |
|---|---:|---:|---:|---:|
| Non-cube, spatial reuse off / still | 35.613% / 34.221% | 356.886% / 318.699% | 92.966% / 92.799% | 0.0111 / 0.0110 |
| Non-cube, spatial reuse off / settled | 42.569% / 47.272% | 465.076% / 731.863% | 94.681% / 94.764% | 0.0142 / 0.0145 |
| Non-cube, spatial reuse off / motion | 520.746% / 384.545% | 10110.653% / 10145.122% | 73.465% / 72.066% | 0.0277 / 0.0244 |
| Basilica / still | 29.801% / 32.203% | 326.738% / 345.193% | 93.775% / 93.242% | 0.0103 / 0.0102 |
| Basilica / settled | 43.512% / 49.162% | 246.118% / 403.598% | 95.279% / 95.438% | 0.0143 / 0.0147 |
| Basilica / motion | 425.720% / 422.366% | 15868.842% / 15222.255% | 72.456% / 71.529% | 0.0380 / 0.0362 |
| Non-cube / still | 45.070% / 44.171% | 516.748% / 457.342% | 92.623% / 93.261% | 0.0112 / 0.0111 |
| Non-cube / settled | 41.073% / 57.994% | 315.897% / 508.685% | 94.022% / 95.408% | 0.0126 / 0.0147 |
| Non-cube / motion | 523.203% / 203.771% | 12156.168% / 13715.012% | 74.272% / 71.141% | 0.0305 / 0.0275 |
| Night GI / still | 82.699% / 30.679% | 366.622% / 269.013% | 96.639% / 86.959% | 0.0117 / 0.0069 |
| Mirror room / still | 18.325% / 17.413% | 157.398% / 118.393% | 82.600% / 83.599% | 0.0087 / 0.0088 |

## Changes and attribution

The initial and spatial screen traces can reject a segment using its smallest enclosing Hi-Z cell. Both projected endpoints must be inside the viewport margin, the cell must exist, and its depth interval must be disjoint after rounding margins. Ambiguous segments follow the existing walk. At a voxel hit, a valid radiance-cache answer avoids the screen reprojection and history lookup. Direct lighting skips a light whose surface cosine or caster visibility gives zero contribution. The reference mode and the 256-step voxel budget are preserved.

The voxel face producer now keeps sprite keys and alpha thresholds in separate contiguous columns. The per-material rotating clear owns both words. After that clear, begin4_c takes a snapshot of each face threshold. Values only increase through shadow drawing. When the proposed unsigned threshold bits exceed the snapshot the original atomic maximum still runs. This preserves arbitrary input thresholds, including a cleared face with contributions from several draws.

At setup, each of the eight octants receives its two packed bounds and directional quad header in a single four-word record. Missing headers are explicitly zeroed. Initial, visibility and cache traces read the same header for the original sign mask. SSBO 3 grows by 1 MiB at each grid size. The fourth word of the first six records per material stores the full pre-shadow threshold. This removes the extra image binding from the byte-snapshot trial.

A separate packed test of the Hi-Z and shading changes saved 0.0804 and 0.0876 ms in the two adjacent basilica comparisons, and 0.0788 and 0.0942 ms in the mirror room. The sprite-key/alpha column layout alone added 0.0036 and 0.0128 ms of savings on the non-cube input over the untransposed byte snapshot. These increments depend on the preceding changes. Use the end-to-end paired totals above for the combined effect.

## Producer checks

A shadow diagnostic applied values from 0 through 0.95 to the same vertices in a conventional reference table. It rotated resets through 16 frequently drawn materials. All four byte-snapshot captures matched cumulative alpha maxima for all 98,304 faces. Rounding an 8-bit lower bound upwards failed 724,405 out of 5,767,168 offline inputs. The original and modified shaders both have order-dependent sprite-key/UV selection across independent scene reloads. The same-input comparison reconstructs the baseline physical layout from live candidate values before running its own packaged trace. The directional record check also compares the expanded header with the independent material LUT when that layout is used.

The unchanged C21 oracle produced zero differences. Deliberately changing the result triggered 2,241,563 geometry/state flags and 1,680,245 radiance flags. The cache and spatial controls also included independent unchanged and deliberately altered paths.

Iris can supply −1 for an unregistered block. Shadow vertices previously indexed the face table on any nonzero unsigned ID. The final shadow producer reuses its validated LUT ID before indexing the fixed face table. All 12740 registered IDs keep their original indices. The six expanded shadow vertex wrappers differ from the pilot only in this guard. All final timing pairs and the normal image captures use the guarded package.

The full-threshold pilot accumulated 0 alpha differences over 4 accelerated-reset captures.

The same-input C21 comparison also ran through the moving non-cube scene. It exposed its live state and mismatch flags through the final pass on all nine camera frames and the settled frame. Over 9,216,000 pixels, 0 geometry/state and 0 stored RGB flags differed.

The packed C30 input compared 1,264,756 spatial visibility rays against the baseline screen and voxel tests. 0 decisions differed across the four captures.

Additional live C21 comparisons exercised grid16. The raw-pixel totals above include those captures.

The Kotlin options generator reproduced the checked-in shaders. The updated voxel contract passed 12,740 material mappings and 59,804 quads, including the eight direction masks and the three buffer allocations. The baseline contract still referred to removed call paths. Its unchanged failure was reproduced before updating the source assertions.

| Normal input | Alpha differences A/B and A/A | Directional box differences A/B and A/A | Sprite key differences A/B and A/A | UV differences A/B and A/A |
|---|---:|---:|---:|---:|
| Non-cube, spatial reuse off | 0 / 0 | 0 / 0 | 130 / 141 | 396 / 403 |
| Basilica | 0 / 0 | 0 / 0 | 3 / 5 | 4 / 5 |
| Non-cube | 0 / 0 | 0 / 0 | 130 / 129 | 376 / 377 |
| Night GI | 0 / 0 | 0 / 0 | 13 / 15 | 52 / 51 |
| Mirror room | 0 / 0 | 0 / 0 | 0 / 0 | 2 / 4 |

## Night reload repeatability

The first A/B night screenshot had a larger relative error than its A/A repeat. A fresh B/A/B sequence at the same preset gave three independently loaded captures per revision. The table reports the median and full range across image pairs. Every mixed pair uses the baseline image in the denominator.

| Reload pairs | Count | Relative RGB P99, median [range] | Mean absolute RGB, median [range] |
|---|---:|---:|---:|
| Baseline / baseline | 3 | 30.68% [29.46–32.05%] | 0.0068 [0.0066–0.0069] |
| Candidate / candidate | 3 | 50.51% [30.03–51.75%] | 0.0129 [0.0068–0.0130] |
| Baseline / candidate | 9 | 32.18% [28.68–90.44%] | 0.0070 [0.0064–0.0129] |

One candidate capture accounts for the three largest mixed-pair errors. Against each of the other two fresh candidate loads, all three baselines had P99 between 28.7% and 34.1%. The first outlier is retained. Its cause was not isolated.

## Frame-to-frame changes during motion

For each adjacent pair in a nine-frame turn, the native display RGB change was calculated at every pixel. The table reports mean absolute change over pixels, channels and frame pairs. The last column compares the signed temporal changes between runs. Units are display code values normalized to 0–1.

| Motion input | Baseline / candidate / baseline repeat | Change residual, A/B and A/A |
|---|---:|---:|
| Non-cube, spatial reuse off | 0.1065 / 0.1036 / 0.1055 | 0.0425 / 0.0387 |
| Basilica | 0.1172 / 0.1161 / 0.1187 | 0.0615 / 0.0581 |
| Non-cube | 0.1052 / 0.1036 / 0.1075 | 0.0463 / 0.0434 |

## Alternatives and remaining cost

At the first non-cube workload capture, 870,192 rays made 1.31 million model visits. The median was 15 traversal steps, the 90th percentile 45, and zero rays exhausted the budget. A separate model capture counted 8,563,196 quad tests and 750,809 intersections. At the first basilica capture, 907,079 rays had a median of 29 steps, a 90th percentile of 69, and 388 exhausted. Those loads motivated separate traversal, model and downstream experiments.

- Resuming long rays in a compacted batch after 24 steps preserved the baseline ray result in the basilica diagnostic. Initial trace time increased from 3.485 to 7.634 ms on non-cube and from 2.490 to 5.293 ms on basilica.
- Charging coarser empty-space skips against the original budget preserved the tested ray states but made the basilica walk slower. Without charging, 187 changed continuations in four basilica captures failed the requirement for a correct new hit against an 8192-step reference. The release retains the baseline accounting.
- Skipping slab setup for rays originating inside the voxel grid passed the offline ownership check on 1,179,648 rays. Native basilica initial trace time increased from 2.423 ms to 2.499–2.503 ms.
- Model intersection and traversal remain major costs. The measured directional-model workload offered little opportunity for reordering or eliminating coplanar quads. The saved milliseconds in this record apply to the measured Iris version, RTX 3080 Ti and input presets.

The source-screen inventory records 92 distinct timed variants, including ablations and diagnostic controls. The compacted tail and the coarse-skip counterexamples used separate ray-state checks.

Prepared variants still awaiting a native comparison include clipping screen traversal to a nearby voxel winner and per-quad half-block masks. They have no measured saving in this record.

After the eight default timing groups and the four static C21 checks, the game runtime temporarily stopped answering at idle. Six later requests failed before native admission. A read-only probe completed on the original PID, after which the affected camera and cache inputs were resubmitted. The original preflight errors, thread snapshot and recovery receipts are retained in the archive.

The archive includes a Git bundle of 337 experimental refs. The prerequisite commit is 8ce1cf34c. The bundle and its individual refs are identified in experimental-refs.json.

## Package identity and evidence

Both ZIPs were produced by the normal Kotlin packaging entry point. All expanded shader paths and individual hashes were compared against the Git snapshots. Vibris matched the canonical source digest on every timed load. Source screening used private patch revisions. The directional-record pilot was `6835f5778`. A final mapped-material guard changed six shadow vertex wrappers. The packaged revision in these paired tables includes that guard.

| | ZIP SHA-256 | Loaded source SHA-256 | Pack snapshot |
|---|---|---|---|
| Baseline | `fa4f5a146799f7bec03435a201bdcf6953c754801ef06b9288868acc01de3a49` | `f515a7d40c6a5b49a3ac7c6eadaf065d65ea6e4544cad0345ac621168a128578` | `0e68c610d` |
| Candidate | `5afbd4a5df9e0db33f555079c6eb60dc02196ee98753b397aa33a3f58ac6aa4b` | `ce1384574d25e86f757f66efb83858890d0ecf01cbb0ef900851994d733ce627` | `9483eecf1` |

The primary checkout retains the packages, normal images, face-table dumps, GPU receipts, ordered diagnostics, scripts and patch variants at `builds/gi-tracing-4h-20261127/evidence.zip`. The sibling `evidence.sha256` identifies the archive.

<!-- vela:evidence {"run":"19fec089-3d33-4cac-965e-88290e49fdef","path":"evidence.zip"} -->
