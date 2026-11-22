# Cosine STBN direction clustering, 2026-10-20

Retain the original cosine-direction texture. Both clustered layouts reduced the measured
time of the HiZ producer plus voxel fallback. Each pixel sampled nearly the same
direction on consecutive frames. Reordering directions independently within
each slice preserved the sample set and degraded coverage across the 64-frame cycle.

The experiment ran on `32b38c264cb719ac1736969db85e1edbc1b1d4a2` with Minecraft
1.21.11, Iris 1.10.6-vibris.2 and an RTX 3080 Ti. At 1280 × 720 with path guiding
and the radiance cache enabled, each run had 256 warmup and 512 measured frames.
The order was original, K128, K256, K256, K128, original. Night used three rounds
and room used two. Timings are medians of per-run means.

| Measurement | Original | K128 | K256 |
| --- | ---: | ---: | ---: |
| Night, HiZ producer + voxel fallback (ms) | 0.973157 | 0.899318 | 0.899763 |
| Room, HiZ producer + voxel fallback (ms) | 0.506318 | 0.480073 | 0.479245 |
| Consecutive-frame direction dot product, mean | 0.250012 | 0.953375 | 0.969622 |
| RMS error in the 64-frame direction mean against (0, 0, 2/3) | 0.033317 | 0.714649 | 0.725177 |

The direction probe covered all 16,384 destinations across 64 slices using the
runtime RGB8 decode. Screenshots by themselves did not distinguish the candidates
from baseline variability. The probe was the reason to reject runtime integration.
Future candidates should test per-pixel temporal coverage alongside spatial coherence.

The [offline clustering script](../scripts/stbn-clustering.main.kts) retains the
export and verification paths. From `scripts/`, run its small deterministic checks
with `kotlin stbn-clustering.main.kts self-test --suite final-export` and
`kotlin stbn-clustering.main.kts self-test --suite angular-tail`.

| Artifact | SHA-256 |
| --- | --- |
| Original cosine direction texture | `8B4715FB4DA4CD83D5AB47E4DEAF36F7D71E7032F9C06498C4FAD83FED8A0536` |
| K128 permutation from the experiment | `E8463A09B9E59D3270A36B78A9C29922E6F810230C7680584CD08B04417B101F` |
| K256 permutation from the experiment | `7EBAE7A92160330CB1FA458F24EAEB495396CC36D0790AE6C0101DD0ECD7199B` |
