# 2.0 GI performance measurements before the continuous window

Board input 110 reserves a fresh 14,400 seconds after both release checks return the GPU and dedicated game. Start the clock when both leases are held and usable. The earlier work began at 09:08:37 UTC on November 22 and the final baseline restore finished at 10:16:25 UTC. This work consumes zero seconds of the new window. The shader baseline is `5ba8718d4245f3d2cf4db627bfbe3258e18aaaea`. All shader changes below are experimental patches. The hybrid GI checkout belongs to its release owner.

## Timing

RTX 3080 Ti, 720p, default pack options. For each frame, sum the 29 active typed GPU program samples covering voxel maintenance, radiance cache, ReSTIR and GI denoising, then take the median across frames. The frozen program membership, actual loaded scene, settings, shader snapshot and samples are in the receipts. Program times below are medians. Ambient GPU work and whole-frame time varied.

Under the old priority policy, `ff8836ed-3f76-4adb-8871-568d5bad117b` completed eight adjacent ABBA cases with 192 measured frames each. The candidate used occupancy masks from the previous frame to clear written material quads and excluded pending cache entries without carried or feedback faces.

| Scene | GI chain A/B/B/A (ms) | Adjacent savings forward / reverse (ms) | Material clear A → B (ms) | Cache update A → B (ms) |
| --- | --- | --- | --- | --- |
| Non-cube | 5.514 / 5.460 / 5.522 / 5.638 | 0.054 / 0.116 | 0.118–0.119 → 0.054 | 0.475–0.478 → 0.482–0.483 |
| Basilica | 5.617 / 5.491 / 5.491 / 5.656 | 0.126 / 0.165 | 0.103–0.104 → 0.046 | 0.816–0.818 → 0.829–0.830 |

The cache change also cut allocation time. Its reservoir update increased in both scenes. The full-frame wall averages ranged from 26.20 to 29.21 ms on Non-cube and 18.03 to 20.15 ms on Basilica. Shadow terrain ranged from 4.131 to 4.246 ms and 1.917 to 1.918 ms respectively. Isolate the material clear in fresh ABBA arms before attributing a chain saving to it. Cache overflow, allocation order, visual convergence and motion remain open for the combined candidate.

After deployment, `7e8efb9f-8260-4752-bf29-84036f823aca` completed five Non-cube screening arms, 80 warmup and 128 measured frames each. PID 54872 had MANAGED class 5 at each Core arm start and end. Shadow terrain ranged from 3.823 to 3.848 ms. WDDM samples in the four candidate arms include System copy and Chrome 3D traffic. The first baseline arm lacks contemporaneous per-process samples. These are screens awaiting independent confirmation.

| Change | Baseline → candidate (ms) | Observation |
| --- | --- | --- |
| Stagger material generation tags by pool slot | Clear 0.119 → 0.005, tree 0.186 → 0.191, spatial visibility 0.328 → 0.419, chain 5.291 → 5.240 | Most clear savings moved into a later consumer. CPU wrap oracle passed. Same-frame GPU material oracle prepared. |
| Compact continuing rays after 16 visits | Initial trace 2.004 → 4.893 | Discard this scheduling scheme. |
| March consecutive empty children within a mask word | Initial trace 2.004 → 2.115 | Reduced traversal overhead did not pay for the extra child processing. |
| Bound cache validation by the expected surface distance | Cache update 0.465 → 0.470 | No useful saving observed. |

Earlier screens also rejected the bounded half-node replay, directional ray bins, separate trace and shading, and a single-subgroup spatial material reuse. On Non-cube the exact empty-octant candidate reduced tree loads by about 14.2% while the initial trace rose from 2.011 to 2.438 ms. Two same-frame probes found zero changed outcomes across 1,741,430 rays. None exhausted the original budget. This result covers the tested rays only.

## Correctness and resumption

Six exhaustive GPU scans in `49434b72-e8b0-4d5b-928b-7a65fc267389` found zero nonzero material words after the sparse clear. Each of the two grid-64 scans covered 67,108,864 words with 5,588 allocated bricks. Grid 32 and grid 16 covered 4,194,304 and 1,048,576 words each, with all 1,024 and 256 slots allocated. The two grid-16 scans ran with the radiance cache off. Verify motion across brick boundaries, disocclusion, reset, normal output and independent scene timings during the continuous window. The material-tag CPU oracle compared 100,667,904 cell states over 131,078 frames, including 27 physical slot wraps and 5,314,465 same-cell writes competing for the maximum.

The sealed Vela preparation bundle contains 10 workflow receipts including the invalid cold scene load and the corrected compile failure, candidate and counterexample patches, CPU experiments, eight SHA-verified GPU buffers, program samples, resource deployment boundaries and background readings. Its SHA-256 is `fd776694576e997093d356d6e99c606654222d1fb2491ee3f5cfea4555bb6b28`. The last restore, `156eb7a1-1a45-4ed6-92aa-f72ef2b6437f`, verified baseline source `d1ddb43bb109` and default settings `4c082fd98204`. The game queue was empty when both leases were released. Thirty consumed frozen shader copies (1,485,834,505 bytes) were removed after terminal-state and provenance checks. The next window holds the evidence as a registered consumer.

<!-- vela:evidence {"run":"3b463183-271f-4926-a480-ea84174e4656","path":"preparation.zip"} -->
<!-- vela:evidence {"run":"3b463183-271f-4926-a480-ea84174e4656","path":"gpu-policy-groups.jsonl"} -->
<!-- vela:evidence {"run":"3b463183-271f-4926-a480-ea84174e4656","path":"offline-results.json"} -->
