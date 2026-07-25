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

## Scope

The port contains the complete required upscaler estimator and optional RCAS
sharpening code. It intentionally omits FSR2, frame generation, optical flow,
backend/provider code, shader blobs, debug rendering, auto-reactive generation,
Xbox-only paths, and resource aliasing code.

There are no entrypoints, resource bindings, pass registrations, settings, or
allocated textures in this change. These files do not alter the active render
pipeline until pass-specific callbacks and entrypoints are added.

## Pass graph

The required dispatch order is:

1. `prepare_inputs` (8x8 at render size)
2. `luma_pyramid` (256x1 SPD)
3. `shading_change_pyramid` (256x1 SPD)
4. `shading_change` (8x8 at half render size)
5. `prepare_reactivity` (8x8 at render size)
6. `luma_instability` (8x8 at render size)
7. `accumulate` (8x8 at output size)
8. `rcas` (optional, 64x1 covering 16x16 output pixels)

Each pass must provide the SDK callback functions for its bound resources and
constants before including its algorithm file. The accumulate include order is
`common`, `sample`, `upsample`, `reproject`, then `accumulate`. The two pyramid
files include the existing `../spd/ffx_spd.glsl`; RCAS includes the existing
FFX core and FSR1 kernel.

## Integration contract

The inputs are jittered linear HDR color, device depth, motion vectors,
exposure, reactive mask, and transparency/composition mask. The integration
must use the SDK's motion-vector sign and UV units, pass the current and previous
jitter in render-pixel units, reset all histories on camera cuts or resize, and
keep pre-exposure consistent across frames.

Persistent ping-pong resources are accumulation (`R8_UNORM`), current luma
(`R16_FLOAT`), internal upscaled color (`RGBA16_FLOAT`), and luma history
(`RGBA16_FLOAT`). Transient resources include farthest depth/luma instability
(`R16_FLOAT`), half-resolution shading change (`R8_UNORM`), output-size new
locks (`R8_UNORM`), six half-resolution SPD mips (`RG16_FLOAT`),
half-resolution farthest depth (`R16_FLOAT`), dilated reactive masks
(`RGBA8_UNORM`), frame info (`RGBA32_FLOAT`, 1x1), and the SPD counter
(`R32_UINT`, 1x1). Dilated depth is `R32_FLOAT`, dilated motion is
`RG16_FLOAT`, and reconstructed previous depth is `R32_UINT` because the
prepare pass updates its float bits atomically.

Both SPD passes require the binding layer to define
`void SPD_IncreaseAtomicCounter(inout FfxUInt32 spdCounter)` and
`void SPD_ResetAtomicCounter()`. These callbacks must use the shared 1x1
`R32_UINT` resource; a workgroup-shared counter cannot synchronize separate
workgroups.

FP32 is the baseline path. Enabling `FFX_HALF` requires 16-bit arithmetic type
support in the entrypoint. Define `FFX_SPD_NO_WAVE_OPERATIONS` when subgroup
quad operations are unavailable.

The active shaderpack does not currently provide object motion vectors or a
dedicated reactive mask. Those inputs must be implemented before this code can
replace the existing TAA chain without temporal artifacts.
