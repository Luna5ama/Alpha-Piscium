# Opaque hybrid GI runtime acceptance

Language: English | [简体中文](../../sc/development/hybrid-gi-acceptance.md)

The frozen specification is `48c9e1f0:docs/rendering/hybrid-gi.md`. Implementation `b59d5ef7` starts from `5ba8718d4`. The 1280 × 720 captures ran on Minecraft 1.21.11, Iris 1.10.6-vibris.3 and an RTX 3080 Ti. The receipts identify the loaded source, settings, scene, passes and compilation outcome. The user deferred real Voxy distance storage and endpoint reconstruction at long range.

| Ordinary scene | Observation |
| --- | --- |
| Iron golem, slabs and intervening wall | Both intersections were positive at 88,040 samples. None selected the farther surface outside the 0.015-block ownership tolerance. Voxels won 4,749 strict comparisons and screen geometry won 7,332. The rest fell within tolerance. With the additional wall, a ray from (49.0625, 66.0625, 49) had screen, voxel and selected distances of 3.125, 2.084 and 2.084 blocks. Jobs `d26213fd` and `3b70deaa` retain the raw pixels and images. |
| Grid boundary | A 16-brick grid yielded 1,537 pre-entry hits, 4,238 post-exit hits and 1,758 screen-only hits at the far-wall viewpoint. 1,316 finite connections with outside segments passed and 77 were rejected during voxel traversal. 256,034 samples used the environment after a real exit. With an instrumented one-step traversal budget, all 563,053 exhausted samples kept zero contribution. |
| Secondary lighting | Looking away from the lamp yielded 5,507 offscreen emissive voxel hits and 82,210 offscreen RC selections. All 94,742 observed selections with both reliable history and RC took RC. Disabling RC left zero RC selections and 5,147 offscreen emissive hits. The synthetic check covers a valid black RC estimate with nonzero history. |
| Motion, disocclusion and reset | Three frames of a moving golem had 1,298, 1,609 and 1,381 selected screen-only endpoints. All three had zero nonempty next-frame primary or temporal reservoirs for these endpoints. Captures include the moving entity, camera turn and the second frame after reload. Final images include the temporal response of TAA and denoising. |
| Filter input and spatial reuse | `49afc6e1` captured all eight prefilter FP16 components and the final image. Diffuse and specular RGB were finite throughout. The glossy room ran with spatial reuse on and off using the compatible binding layout. The mean absolute 8-bit channel difference was 2.74 and P95 was 9, including sampling and filtering differences. |

## Paired 720p timings

Job `09e53519` measured 96 frames after 48 warmup frames per arm in the same room and configuration. Values are compute P50 in milliseconds.

| Pass | Baseline A | Hybrid B | Hybrid B | Baseline A |
| --- | ---: | ---: | ---: | ---: |
| Initial directions composite20 | 0.099 | 0.098 | 0.098 | 0.099 |
| Voxel and screen tail composite21 | 0.400 | 2.039 | 2.038 | 0.401 |
| Finite visibility composite30 | 0.048 | 0.335 | 0.335 | 0.048 |
| Direct-light control composite13_a | 0.104 | 0.104 | 0.104 | 0.104 |

The implementation retains the four-block near interval and 128 screen steps. Additional performance work belongs to 2.0.x. Earlier offline checks covered eight entries under three configurations and 601 oblique analytic-plane rays. Rejected receipts cover terrain stability timeouts at `gi-shadow-1-720p`, the first diagnostic exceeding eight image uniforms, and an export quota failure.

Vela run `e04d872b-8670-4d47-8fa4-04c669e1804f` retains `gi-iris.zip` with readbacks, screenshots, numerical comparisons, diagnostic patches, scene construction and settings receipts. SHA-256 is `dbc5bc41b77ba7360640f7f2bf9e4432e762d2a7b64cb1c0c6e8f10d91bc31b5`.
