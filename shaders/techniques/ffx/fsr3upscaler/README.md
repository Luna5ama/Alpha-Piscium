# FSR3 Upscaler GLSL port

This directory contains the GPU algorithm from FidelityFX SDK 2.3.0,
FSR3 Upscaler 3.1.5, adapted for the shaderpack's GLSL include system.
The source revision is:

`60f4ea81909200d8542eca14dccb2628b763a9a3`

The files retain AMD's MIT notice. The project-wide copy of the license is
`/licenses/MIT.txt`.

## GLSL adaptations

HLSL-only pragmas, Xbox paired kernels, and runtime-indexed local arrays were
removed. Array-backed sample sets are represented by explicit fields and
if-chains, preserving the SDK's sample order and weights.

Nearest and farthest depth extents are updated independently. In the 3.1.5 SDK
source, the farthest update is nested inside the nearest-depth branch and can
never move away from the nucleus depth. This port performs the intended min/max
reduction over the valid 3x3 neighborhood.

## Active integration

The port contains the complete required upscaler estimator. Sharpening uses the
shaderpack's shared FSR1 RCAS pass. It intentionally omits FSR2, frame generation,
optical flow, backend/provider code, shader blobs, debug rendering, auto-reactive generation,
Xbox-only paths, and backend resource aliasing code.

`/pass/composite/FSR3MotionVectors.comp.glsl` generates camera motion and packs the
reactive and transparency/composition masks in its Z/W channels. The seven
FSR3 entrypoints bind the callbacks from
`Integration.glsl`. Surfaces without object transforms are marked reactive.

## Pass graph

The required dispatch order is:

1. `prepare_inputs` (8x8 at render size)
2. `luma_pyramid` (256x1 SPD)
3. `shading_change_pyramid` (256x1 SPD)
4. `shading_change` (8x8 at half render size)
5. `prepare_reactivity` (8x8 at render size)
6. `luma_instability` (8x8 at render size)
7. `accumulate` (16x8 at output size)

The shared `/pass/composite/RCAS.comp.glsl` pass then sharpens the accumulated
output at presentation resolution and writes it to `main`.

Each pass must provide the SDK callback functions for its bound resources and
constants before including its algorithm file. The accumulate include order is
`common`, `sample`, `upsample`, `reproject`, then `accumulate`. The two pyramid
files include the existing `../spd/ffx_spd.glsl`.

## Integration contract

The input color is unexposed linear HDR from `TAAPrepare`, and FSR3 keeps its
accumulated output linear. The luma pyramid computes a frame-local reconstruction
exposure before every stage that consumes it. At SPD level 5, the integration
rebuilds log luminance from the unaffected 64x64 linear-luminance reduction,
working around the upstream dark-luminance clamp without changing AMD source.
The frame-info update skips temporal exposure smoothing, and all FSR3 stages read
the resulting current-frame exposure from `global_fsr3FrameInfo.x`.

Current color and scene-linear history are multiplied by the same FSR3 exposure.
Accumulation reverses the internal HDR transform and divides by that exposure
before storing scene-linear history and output. `DeltaPreExposure()` remains one
because the shaderpack does not pre-expose either input or history. The shared
RCAS pass separately applies the shaderpack display exposure exactly once, then
uses the AgX transform before sharpening, restores linear output, and writes
full-resolution `main` for Bloom and the remaining post-processing passes.

`RenderSize()` comes from the Iris CPU-side render-image size, while
`UpscaleSize()` comes from the view/output size. G-buffer texture gradients use
`0.5 * uval_mainImageScale`; relative to low-resolution raster derivatives this
implements the recommended `log2(render/output) - 1` mip bias. Accumulation,
shared RCAS, Bloom, and the remaining post-processing passes use output size.

The other inputs are device depth, motion vectors, reactive mask, and
transparency/composition mask. The integration uses the SDK's motion-vector sign
and UV units, passes current and previous jitter in render-pixel units, resets
history on temporal discontinuities or an incomplete previous frame, and keeps
pre-exposure consistent across frames.

Persistent ping-pong resources are accumulation (`R8_UNORM`), current luma
(`R16_FLOAT`), internal upscaled color (`RGBA16_FLOAT`), and luma history
(`RGBA16_FLOAT`). Transient resources include farthest depth/luma instability
(`R16_FLOAT`), half-resolution shading change (`R8_UNORM`), twelve packed SPD
levels (`RG16_FLOAT`), half-resolution farthest depth (`R16_FLOAT`), and dilated
reactive masks (`RGBA8_UNORM`). New locks use the output alpha channel, the SPD
levels share a full-render-size atlas, and frame info, the completed-frame marker,
plus the SPD counter live in the global SSBO. Dilated depth is `R32_FLOAT`,
dilated motion is `RG16_FLOAT`,
and reconstructed previous depth is `R32_UINT` because the prepare pass updates
its float bits atomically.

The upscaled-color atlas contains two full-output-size color-history regions, a
third full-output-size new-lock region, and four render-size luma/history tiles
below them. After accumulation and shared RCAS finish using the new-lock data,
Bloom reuses the third region's RGB channels. Packed Bloom filtering clamps every
read to the source tile's texel-center bounds so bilinear samples cannot mix
neighboring tiles or stale RGB from the previous frame.

Both FSR3 SPD passes use the dedicated `global_atomicCounters[14]` through
`SPD_IncreaseAtomicCounter` and `SPD_ResetAtomicCounter`; the counter is shared
across workgroups and reset between the sequential pyramid passes. Counter 15
remains dedicated to SST step debugging.

FP32 is the baseline path. Enabling `FFX_HALF` requires 16-bit arithmetic type
support in the entrypoint. Define `FFX_SPD_NO_WAVE_OPERATIONS` when subgroup
quad operations are unavailable.

The active shaderpack does not retain previous object transforms. Entities,
block entities, particles, hands, overlays, and translucent surfaces therefore
use camera motion with reactive masking instead of incorrect object motion.
