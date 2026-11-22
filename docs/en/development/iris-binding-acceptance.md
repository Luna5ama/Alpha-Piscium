# Iris 1.10.4 binding acceptance

Language: English | [简体中文](../../sc/development/iris-binding-acceptance.md)

Implementation `3e7f6be0` starts from `5ba8718d4` for the 2.0 release scope in `a6867d91:docs/roadmap.md`. Binding 3 holds face UVs, sprite keys and directional models followed by brick metadata. Binding 5 holds ping-ponged RC indirection, counters and active indices followed by optional guide statistics. Binding 6 holds the reservoirs. Screen-step diagnostics use an optional tail in binding 2. The declarations, readers, writers and generated properties no longer use 9, 10 or 11. Voxy retains `colortex16/17/18`.

An isolated game directory and copy of the mirror room ran Minecraft 1.21.11, official Fabric Iris 1.10.4, Sodium 0.8.2 and Fabric Loader 0.19.3. The Iris SHA-512 matched Modrinth version `Q4YiZeCX`. Fabric rejected the first launch because Sodium 0.8.4 declares a conflict with this Iris version. The compatible combination loaded `AP-compat`. The probe read storage directly from the live Iris pipeline and captured a 1280 × 720 final image for each configuration.

| Effective settings | Allocated bricks | RC active counts | Binding 5 bytes |
| --- | ---: | --- | ---: |
| RC, guide and normal mapping enabled | 137 | 18,847 / 18,935 | 69,206,032 |
| RC and normal mapping enabled, guide disabled | 136 | 17,169 / 17,103 | 52,428,816 |
| RC and normal mapping disabled | 136 | RC magic uninitialized | 52,428,816 |

With guiding active, 4,013,064 of 4,194,304 guide words were nonzero and the RC active header had the expected magic. Captures include the face/model region, brick allocation, reservoir data and final image. Buffer fill values in the disabled case have no cache-validity meaning.

Iris 1.10.6-vibris.3 compiled, loaded and rendered all three configurations in job `07caffb3`. Applying the shader changes to hybrid GI `b59d5ef7` also rendered with spatial reuse enabled and disabled in job `bbb18f15`. The combined test used patch SHA-256 `6730cefe8bb2022b57b6928ffc38fbb4c0d13580bfb858d685de8dd87d413b2b`.

Shadesmith and Kotlin regenerated the properties. The ReSTIR-CV layout check passed. The full script reported four temporal and spatial estimator assertions on both the baseline and candidate.

Vela run `32aae940-42b6-4f33-9686-9bdf0d696a3d` retains `binding-iris.zip` with the old client logs, readback probe, three sets of GPU readings and images, and the current client receipts and images. SHA-256 is `74d9bf75db6201007dc30ccd7bbae4f309de7874e7793403b9276dbff53fafbc`.
