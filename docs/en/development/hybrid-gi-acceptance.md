# Opaque hybrid GI runtime acceptance

Language: English | [简体中文](../../sc/development/hybrid-gi-acceptance.md)

The frozen specification is `48c9e1f0:docs/rendering/hybrid-gi.md`. Implementation `b59d5ef7` starts from `5ba8718d4`. The 1280 × 720 captures ran on Minecraft 1.21.11, Iris 1.10.6-vibris.3 and an RTX 3080 Ti. The receipts identify the loaded source, settings, scene, passes and compilation outcome. The separate native Voxy acceptance below covers long-range distance storage and endpoint reconstruction.

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

## Native Voxy distance and visibility

The isolated Minecraft 1.21.11 instance ran `f10a01db5` on official Iris 1.10.7, Voxy 0.2.13-alpha and Sodium 0.8.7. All 496 shader files in the uninstrumented pack matched the frozen Git tree. The `aerial-perspective-4-720p` camera in Dump Jungle used three vanilla render chunks, a 16-brick grid and the Ultra profile with spatial reuse. Two native frames contained 64,292 and 64,291 opaque Voxy pixels whose view depth matched the merged depth bit for bit. The first frame reached 2,168.99 blocks in view depth. The Vibris.3 Iris constructor was incompatible with Voxy 0.2.16's injection signature. A local Fabric probe read Iris and GPU resources directly in the isolated instance. The ordinary-scene Vibris timings above remain the performance evidence. [Voxy release](https://modrinth.com/mod/voxy/version/k5PZxsfL) · [Iris release](https://modrinth.com/mod/iris/version/fDpuVzVr)

Sixteen diagnostic frames sampled one receiver per 4×4 pixels. The FP64 reference intersected the original ray with the plane recovered independently from the winning native Voxy depth and unpacked face axis. All 9,992 candidates ending on Voxy preserved the distance bits across production, tile storage and temporal consumption. The longest ray measured 713.417 blocks. The greatest endpoint view depth was 2,001.822 blocks.

| Absolute error (blocks) | P50 | P95 | P99 | Maximum |
| --- | ---: | ---: | ---: | ---: |
| Read distance vs. reference distance | 0.00000137 | 0.00002818 | 0.00008603 | 0.00269605 |
| GPU endpoint vs. reference intersection | 0.00000387 | 0.00003390 | 0.00009295 | 0.00270821 |

The pre-entry, near overlap, post-exit and grid-bypassing segments hit Voxy 714, 6,889, 99 and 2,290 times, respectively. Another 79 rays from Voxy primaries entered the grid at a positive distance and hit voxel geometry. One FP64 reference landed 0.000016 pixels across an x boundary, with a 0.00000322-block endpoint error.

Frame 1535 queued a 223.126-block spatial connection at receiver (432, 112). Its endpoint's ray distance differed from the projected Voxy plane by 0.00836 blocks. The test replayed this connection through the production visibility function with the origin, direction and endpoint fixed. It left the grid at 136.824 blocks. A 1×3×3 wall in the writable world copy intersected the ray from 15.543 to 16.806 blocks. All screen segments were clear in the selected open, walled and reopened frames. The voxel segment passed, rejected and passed. Diffuse RGB was preserved at (8.9141, 8.6406, 8.0469), rejected from (10.7812, 10.3906, 9.4062) to zero, then preserved at (4.1289, 5.0000, 8.8438).

Run `e0db906f-3f03-4496-b81f-25ded7d5390f` holds `voxy-gi.zip`, including native screenshots, 16 raw diagnostic frames, wall controls, full pre-filter diffuse/specular FP16 data and the final image from the same frame. SHA-256 is `5ebf9bcad6d3adfc12ec0361d0691c667ec1e93d4a1fc934f7aa554d28031668`. Both GPU and isolated-instance leases were released. Production shaders and the dedicated Voxy targets `colortex16/17/18` were preserved.

<!-- vela:evidence {"run":"e0db906f-3f03-4496-b81f-25ded7d5390f","path":"voxy-gi.zip"} -->
<!-- vela:evidence {"run":"e0db906f-3f03-4496-b81f-25ded7d5390f","path":"prefilter-final.png"} -->
