# ReSTIR GI Specular Ratio-Estimator Overflow

**Touches:** `shaders/pass/composite/GIReSTIRTemporalReuse.comp.glsl`,
`shaders/pass/composite/GIReSTIRPairedSpatialShade.comp.glsl`

## Symptom

Black, grainy speckle blobs appeared on alpha-tested foliage (leaf blocks)
placed directly in front of bright, close-range light fixtures. The same
blobs were visible in SSR reflections of the same geometry, confirming the
corruption was in the G-buffer/shading pass, not the reflection system.

## Root cause

The ReSTIR GI specular path uses a **ratio estimator**: traced specular
radiance is divided by the surface's BRDF specular albedo
(`splitSumSpecularLUT`) before storage so the denoiser operates on pure
radiance, with the real albedo re-multiplied at composite time.

```glsl
vec3 specAlbedo = resampleMaterial_specularAlbedo(material, NDotV);
ssgiSpecOut.rgb *= safeRcp(specAlbedo);   // ← denominator can be near-zero
```

For rough, low-F₀ dielectrics such as leaves, the split-sum LUT returns
near-zero specular albedo at many view angles.  With a bright, close-range
light in the frame, the numerator was substantial (direct spec highlight
picked up via the traced sample), so dividing by a near-zero denominator
spiked the stored value toward `FP16_MAX` (~65504) before the downstream
`clamp(..., 0.0, FP16_MAX)` caught it.  That extreme value propagated through
the bilateral spatial/temporal denoiser, destabilising luminance-weighted
neighbour contributions and producing the characteristic black-speckle pattern.

`safeRcp` returns 0 when its argument is exactly zero, so pure-zero albedo
was safe.  The problem was the tiny-but-positive values that the LUT
quantises to near high roughness / low NDotV — large enough to pass the
`> 0` guard, small enough to produce a 10⁴–10⁶× amplification.

## Fix

Floor the demodulation denominator before taking its reciprocal in both
the temporal-reuse and spatial-reuse shading passes:

```glsl
ssgiSpecOut.rgb *= safeRcp(max(specAlbedo, vec3(0.01)));
```

`0.01` caps the ratio estimator's amplification to ≤ 100×, which is far
below the threshold that destabilises the denoiser.  For materials with
normal specular response the floor is never reached, so there is no
perceptible change to specular indirect lighting on metal, glass, or any
other high-albedo surface.
