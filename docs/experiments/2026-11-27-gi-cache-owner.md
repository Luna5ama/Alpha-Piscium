# GI cache-owner lookup on 2.0/dev, 2026-11-27

Published to 2.0/dev at 92599a42d8498533696ae9019707143cf88a8a67.

Two release ZIP files were compared on the dedicated Minecraft 1.21.11 instance, using Iris 1.10.6-vibris.3+mc1.21.11, the Ultra defaults, an RTX 3080 Ti and NVIDIA 617.42. The source baseline was 2cd3467c1 and the isolated cache candidate was 1428b8466. The candidate moves the traced voxel material guard to the cache reader that already fetches the same owner. A rejected cache tag or reservoir now avoids the voxel material fetch. The nearest-face lookup still rejects the wrong material before contributing. Generic and smoothed cache lookups have no traced material constraint. Both packages use the existing settings and storage layout.

| Scene | Baseline ms | Candidate ms | Forward saving ms | Reverse saving ms | C21 saving ms (two pairs) | Terrain sentinel ms (all arms) |
|---|---:|---:|---:|---:|---:|---:|
| Basilica 720p | 7.6997 | 7.6836 | +0.0138 | +0.0184 | +0.0113, +0.0138 | 0.3492–0.3492 |
| Forest horizon 1080p | 11.6470 | 11.6196 | +0.0328 | +0.0220 | +0.0358, +0.0287 | 0.4762–0.4787 |
| Mirror room 720p | 4.3213 | 4.2959 | +0.0236 | +0.0271 | +0.0154, +0.0174 | 0.1536–0.1536 |
| Non-cube 720p | 9.4221 | 9.4024 | +0.0205 | +0.0189 | +0.0195, +0.0108 | 0.3389–0.3415 |
| Night GI 720p | 9.2577 | 9.2396 | +0.0266 | +0.0097 | +0.0118, +0.0108 | 0.3256–0.3267 |

Each arm uses the median of 48 native GPU samples. Six separated eight-frame captures follow 64 normally rendered frames each after the initial 144-frame warmup. Four arms per scene provide both adjacent orders. Positive savings are baseline minus candidate. The first two columns average the two arm medians for each version. The sentinel measures terrain geometry outside the changed shader. The raw receipts contain every sample, source hash, scene, effective setting and group ID. Per-arm external WDDM engine samples are in the comparison JSONs and the original observation logs.

| Scene | Legacy 29 | Six probe passes | Shared Hi-Z | Terrain shadow | Including four shared consumers | Outer composite |
|---|---:|---:|---:|---:|---:|---:|
| Basilica 720p | 5.9267 / 5.9103 | 0.0973 / 0.0973 | 0.0806 / 0.0778 | 1.5959 / 1.5959 | 7.7873 / 7.7719 | 44.0461 / 44.3727 |
| Forest horizon 1080p | 7.9404 / 7.9145 | 0.0922 / 0.0937 | 0.1390 / 0.1398 | 3.4780 / 3.4729 | 11.8300 / 11.8031 | 44.7132 / 44.9856 |
| Mirror room 720p | 3.6091 / 3.5886 | 0.0993 / 0.0978 | 0.0804 / 0.0799 | 0.5299 / 0.5297 | 4.4104 / 4.3825 | 44.2076 / 44.9390 |
| Non-cube 720p | 6.7505 / 6.7341 | 0.1034 / 0.1014 | 0.0799 / 0.0794 | 2.4873 / 2.4868 | 9.5191 / 9.4968 | 52.1029 / 46.4479 |
| Night GI 720p | 6.6867 / 6.6680 | 0.0988 / 0.0996 | 0.0804 / 0.0799 | 2.3864 / 2.3875 | 9.3501 / 9.3330 | 45.9220 / 46.1693 |

All values are baseline / candidate GPU milliseconds. Totals are computed from each frame before taking the median. The comparison includes the 29 previously timed GI stages, the six active environment-probe stages (begin4_a, composite2, 3, 12, 13 and 16), shared Hi-Z composite2_a once and both terrain-shadow draws. The shared-consumer column also includes C48, C51, C71_a and C75, whose generated wrappers include the changed cache helper. Outer composite is a separate enclosing timer that overlaps GI and other effects. The native inventory covers individual stages and their enclosing timers. End-to-end GPU frame time remains unmeasured. Wall time includes profiling overhead. Optional programs not active under Ultra are listed in the unfiltered inventory and paired receipts.

## Screening and drift

In one 1080p source screen, the two untouched baseline arms gave 11.5364 and 12.6060 ms. A probe center-valid shortcut gave 12.3228 ms. Its two dilation passes cost only 0.0307 ms compared with 0.0348–0.0389 ms. An inside-grid exit-plane shortcut gave 12.0310 ms, within the baseline drift. The cache-owner candidate gave 11.4908 ms and eliminated a duplicate read at C21, the largest GI compute stage in this screen. Only that candidate moved to release-ZIP confirmation. The probe oracle matched both FP16 stages on five occupancy densities. A deliberately omitted hole fill changed thousands of texels. The grid interval oracle matched 100,000 rays including 46,125 starts inside. The cache guard agreed in all 64 boolean states, and deleting the guard changed two. The earlier sample-address and half-rate counterexamples motivated testing accuracy alongside timing.

Two earlier 128-frame release-ZIP groups reversed direction under substantial intra-arm drift. The horizon differences were −1.8493 and +2.5661 ms. The basilica differences were −0.2944 and +1.2503 ms in acquisition order. The basilica terrain timer started near 0.345 ms and reached about 0.71 ms during sampling. Post-hoc matching on the terrain timer suggested a small C21 change and motivated the independently run short groups above. Those earlier groups are retained as a counterexample to reading an unpaired median or a favorable ordering as a saving.

## Quality and provenance

The basilica motion test reloaded A/B/A, captured a still, translated 1.5 blocks while turning 14 degrees over nine frames, then captured the settled result after 48 frames. The paired screenshot data include sampling and temporal initialization variance. The separate same-input test ran both versions of the cache read within the packaged C21 on the same live inputs. It compared the fallback/accepted decision and all three float bit patterns for accepted RGB, while keeping the candidate shader output for rendering. Three snapshots and the RC-off control loads are retained in the audit receipts.

Across 2,619,258 evaluated rays, 2,427,774 selected the cache. The fallbacks and accepted RGB float bit patterns agreed on every ray. Omitting the guard would have changed 854 decisions on these same inputs. The P99 and maximum relative RGB error for the accepted cache results were zero, and 0% exceeded 1%.

For independent basilica static reloads, relative display RGB error is the norm of the difference divided by max(baseline RGB norm, 0.01), using normalized display channels. A/B P99, maximum and fraction above 1% were 24.28%, 423.45%, 88.57%. A/A values were 24.49%, 352.71%, 88.90%. Per-motion-frame errors and the second baseline comparison are in visual.json. Geometry, disocclusion and settled lighting were visually compared. At maximum camera displacement the A/B display MAE was 0.0562 versus 0.0682 for A/A. After settling the values were 0.0161 and 0.0155.

Package identity was validated against all 247 shader paths and the frozen source digest at each runtime load. Baseline ZIP SHA-256 291c5ae2290168db89ca9a48f6390e3e830bbeb2b9ab9019b69b6c396b516f2b. Candidate ZIP SHA-256 6ccb61993dadd54f74ad2dcab4ac9eaf03f29a9edaedfcb0a41ce11cb3d23373. Native snapshots d38b8c47a and 012afb603. Eight flattened shader files changed. The diagnostic texture and duplicated comparison function exist only in b5dc93bdb and were excluded from both timing packages.

The first cross-world timing job could not restore the scene and provided no usable measurements. A separate native recovery reproduced the entry scene, source and settings. Subsequent cross-world groups validated the entry and restored all three identities within the lease. Their camera, source and settings hashes are saved per group. All groups reused the same game process.

The result covers the tested default workload, GPU and Iris version. Other drivers, Voxy and nondefault GI reference modes have no new timing evidence. The old long-window measurements do not quantify normal-frame throughput.

## Enclosing timer and normal-render intervals

The enclosing composite timer includes unrelated effects and the profiling interval. In the basilica the candidate enclosing medians were 0.177–0.477 ms higher, while the separately timed GI and shared consumers were 0.013–0.018 ms faster. The mirror enclosing medians were 0.300–1.163 ms higher. The forest changed sign across orders. The 64 normal frames between captures also provide elapsed wall time, including CPU work and presentation. Their arm medians follow in execution order.

Basilica 720p — base 14.259, owner 14.222, owner 14.082, base 14.214 ms/frame.

Forest horizon 1080p — owner 19.541, base 19.329, base 19.439, owner 19.461 ms/frame.

Mirror room 720p — base 10.487, owner 10.406, owner 10.393, base 10.432 ms/frame.

Non-cube 720p — owner 20.974, base 21.088, base 21.252, owner 21.242 ms/frame.

Night GI 720p — base 18.627, owner 18.368, owner 18.834, base 18.962 ms/frame.

The unfiltered night capture covered 89 named compute programs. Summing one instance of each named program per frame, including programs outside GI, gave baseline minus candidate of +0.0276 and +0.0128 ms in the forward and reverse pairs. Per-program changes and composite compute subtotals are in night-all-programs.json.

The shared GI topic records this result at `b43b687b1`. The primary checkout retains the raw receipts and tested packages in `builds/gi-whole-20261127/evidence.zip` (SHA-256 `1a1032ba9c818b6f9c0ffe4528849850da4cde3f3dc451f1cd655c574069d835`).
