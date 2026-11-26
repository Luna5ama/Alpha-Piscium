# Post-processing and Display Transform

Post-processing begins after GI, air/water volumes, and translucency are composed. The main chain is optional DOF,
temporal resolve, spatial antialiasing/sharpening, bloom, post composition, exposure, overlay, and final display
transform.

## Code map

| Path | Responsibility |
| --- | --- |
| [`shaders/techniques/DOF.glsl`](../../shaders/techniques/DOF.glsl) | Shared DOF sampling and circle-of-confusion logic |
| [`DOFFocus.comp.glsl`](../../shaders/pass/composite/DOFFocus.comp.glsl), [`DOFPrepare.comp.glsl`](../../shaders/pass/composite/DOFPrepare.comp.glsl) | Automatic focus and DOF input preparation |
| [`TAAPrepare.comp.glsl`](../../shaders/pass/composite/TAAPrepare.comp.glsl), [`TAAResolve.comp.glsl`](../../shaders/pass/composite/TAAResolve.comp.glsl) | Temporal-AA preparation and resolve |
| [`FXAA.comp.glsl`](../../shaders/pass/composite/FXAA.comp.glsl) | Spatial antialiasing |
| [`RCAS.comp.glsl`](../../shaders/pass/composite/RCAS.comp.glsl), [`techniques/ffx/fsr1/`](../../shaders/techniques/ffx/fsr1/) | RCAS sharpening |
| [`techniques/Bloom.comp.glsl`](../../shaders/techniques/Bloom.comp.glsl) | Bloom downsample/upsample pyramid |
| [`ExposureMip.comp.glsl`](../../shaders/pass/composite/ExposureMip.comp.glsl), [`ExposureGather.comp.glsl`](../../shaders/pass/composite/ExposureGather.comp.glsl) | Auto-exposure weights, mip, and statistics |
| [`PostComposite.comp.glsl`](../../shaders/pass/composite/PostComposite.comp.glsl), [`OverlayComposite.comp.glsl`](../../shaders/pass/composite/OverlayComposite.comp.glsl) | Main post composition and overlay |
| [`techniques/displaytransform/`](../../shaders/techniques/displaytransform/) | Exposure, DRT, and display transform |
| [`Final`](../../shaders/pass/composite/Final.frag.glsl) | Final screen output |

## DOF

[`DOFFocus`](../../shaders/pass/composite/DOFFocus.comp.glsl) runs only when DOF is enabled without manual focus.
After volumes/translucency, [`DOFPrepare`](../../shaders/pass/composite/DOFPrepare.comp.glsl) writes the current main
color to `transient_dofInput` for later post composition. Manual focus skips only the focus pass; preparation still
runs.

Settings cover focal length, f-stop, aperture shape, quality, maximum sample radius, masking heuristic, three-part
manual-focus distance, focus time, and focus-plane debug.

## Temporal and spatial AA

| Order | Pass | Purpose |
| --- | --- | --- |
| 1 | [`TAAPrepare`](../../shaders/pass/composite/TAAPrepare.comp.glsl) | Produces temporal input, luma difference, and supporting data |
| 2 | [`TAAResolve`](../../shaders/pass/composite/TAAResolve.comp.glsl) | Uses `history_taa`, motion/reprojection data, and current-frame input to write `transient_taaOutput` |
| 3 | [`FXAA`](../../shaders/pass/composite/FXAA.comp.glsl) | Writes `transient_fxaaOutput` |
| 4 | [`RCAS`](../../shaders/pass/composite/RCAS.comp.glsl) | Performs final spatial sharpening |

The program list enables FXAA and RCAS only when TAA is enabled. The TAA prepare and resolve entries remain
registered in either case, and Resolve handles the TAA-off color decode.

TAA settings cover enable, jitter, current/history filters, and CAS sharpness. Current/previous-jitter custom uniforms
are generated from the R2 frame sequence in [`scripts/shaders.properties`](../../scripts/shaders.properties);
changing the sampling sequence requires updating reprojection conventions as well.

### 2.1 原生 TAA 与超分实施

实施目标是 `2.1/dev` 的统一时域重建器。原生和降分辨率共用当前帧空间重建、真实抖动相位校正与历史更新。按已确认的规格接入覆盖率与位置矩、双色边缘的显示混色，以及使用重建信息的输出锐化。由重建写入未锐化的历史，撤下新管线末端的 FXAA 兜底。具体空间核、相位缓存和 RCAS 改法由同输入对照选择。[已确认的 2.1 设计](https://github.com/Luna5ama/Alpha-Piscium/blob/22226662bb94683df4ef5fb864280b966905b413/docs/rendering/temporal-reconstruction.md)

首个检查点复用已有捕获，对齐同一段静止、缓慢横移、停稳的轨迹、曝光和抖动。以现行 AP 原生 TAA 与 `2.0/fsr3@b72a8b5fd` 为基线，逐帧定位 GI 降噪前后、重建和锐化后的残留亮点，同时跟踪真实小高光。核对普通静止表面的颜色 AABB，以及粗糙玻璃在没有透明层专用遮罩时的最终历史。用这些画面接续原生与超分实现，两档沿用同一条验收轨迹。已有捕获与离线实现可先行，实机沿项目现有顺序预约。[基线与实景验收](https://github.com/Luna5ama/Alpha-Piscium/blob/22226662bb94683df4ef5fb864280b966905b413/docs/rendering/temporal-reconstruction.md#参考实现与实景验收) · [FSR 原型](https://github.com/Luna5ama/Alpha-Piscium/tree/b72a8b5fddeb2e52ea20a2c36eb16e0c30b0661b) · [透明遮罩修订](https://github.com/Luna5ama/Alpha-Piscium/commit/649eaa67287b9631d4411ec4cf4a1d8cd25b142c)

交付候选修订、匹配的原生 1080p 与 720p → 1080p 运动对照，以及整条重建和受影响后处理的 GPU 耗时。对照高采样参考检查细杆宽度与相对亮度、遮挡或熄灯后的拖尾、墙纹恢复和植被疏密。在 Iris 1.10.4 上分别开关法线贴图，核对实际加载、物理 custom image 数与显存。若实测表明 GI 残噪和真实高光难以兼顾，再按原设计转交 GI／降噪器。协调者负责验收后的 `2.1/dev` 整合与推送，最终画面交给用户查看。[尺寸与兼容范围](https://github.com/Luna5ama/Alpha-Piscium/blob/22226662bb94683df4ef5fb864280b966905b413/docs/roadmap.md#21-原生-taa-与超分辨率)

## Bloom

`SETTING_BLOOM_SHAPE` offers Standard (0, default), Diamond Four (1), Horizontal Streak (2), Cross Four (3),
Camera Six (4), Camera Ten (5), Cinema Star (6), Prismatic Streak (7), 67 (8), Trollface (9), and Custom (10).
Standard combines eight scales with equal weights to form a concentrated core and a long tail. The other presets
fit the complete RGB response, including the intended halo and shape. Each input channel has its own weights for
color dispersion. See [Bloom kernel fitting](../development/offline-tools.md#bloom-kernel-fitting) for the editor and export.

| Order | Pass/stage | Purpose |
| --- | --- | --- |
| 1 | [`Bloom`](../../shaders/techniques/Bloom.comp.glsl), downsample levels 1–8 | Five bilinear reads per output texel. Level 1 reads the decoded linear main image and applies Bloom highlight compression per read. |
| 2 | [`Bloom`](../../shaders/techniques/Bloom.comp.glsl), reconstruct from level 8 to level 1 | Upsample the coarser result, add the selected same-level taps from the raw pyramid, and write the combined result. |
| 3 | [`PostComposite`](../../shaders/pass/composite/PostComposite.comp.glsl) | Upsample level 1 to the viewport, then add Bloom using the intensity and underwater settings before the display transform. |

[`programs.main.kts`](../../scripts/programs.main.kts) owns the order and enable conditions.
[`Bloom.glsl`](../../shaders/techniques/Bloom.glsl) confines reads to the input mip tile. Downsampling and raw
injection zero-extend the bilinear footprint at the viewport boundary, and reconstruction clamps to the valid mip.

- `transient_bloomRaw` holds the frame-local RGBA16F downsample pyramid with ceil-sized, padded mip tiles.
- `transient_bloomReconstruction` holds the frame-local RGBA16F results from the coarse-to-fine passes.
  The allocation in [`shadesmith.json`](../../shaders/shadesmith.json) reuses atlas space after FXAA and RCAS.

Highlight compression runs in the first Bloom downsample, after AA decoding. RGB and luma modes use the same
curves. The decoded writes in [`RCAS`](../../shaders/pass/composite/RCAS.comp.glsl) and the TAA-off path of
[`TAAResolve`](../../shaders/pass/composite/TAAResolve.comp.glsl) clamp to the RGBA16F limit of 65504.

### Bloom validation and limits

The CPU fitter measures the complete impulse response across held-out decimation phases. Dyadic sampling creates
phase-dependent variation. Larger emitters broaden the directional patterns and can fill fine gaps. The CPU
estimate counts weighted bilinear reads across the pyramid. GPU cost also depends on the selected tap layout and
the scene. Compare the affected reconstruction passes, PostComposite, and a stable neighboring pass on the same
Iris setup.

The Bloom validation recorded in commit `211d772e5` used an RTX 3080 Ti and the 720p night scene. Iris ran
all nine shaped presets, Bloom Off, and the exported Custom kernel. The Off capture confirmed that [`PostComposite`](../../shaders/pass/composite/PostComposite.comp.glsl)
still ran. In the same-coefficient Standard/Custom A/A, pre-Bloom HDR means agreed within 0.2%. In the ABBA
comparison with the exported Custom kernel, p50 for per-frame Bloom and PostComposite compute time totaled
0.145 ms for Standard and 0.623 ms for Custom. The two recipes target different kernels, so these times describe
the cost of each recipe. The old overbright capture was invalid because its snapshot omitted the PostComposite wrapper.
Offline checks covered four held-out phases. The edited Camera Six export had up to 4.0% error at the origin.
The editor's HTTP fit and export were exercised. Browser rendering remains unverified.

## Post composition and exposure

| Order | Pass | Purpose |
| --- | --- | --- |
| 1 | [`PostComposite`](../../shaders/pass/composite/PostComposite.comp.glsl) | Combines the main color, DOF, bloom, and other post inputs; the RTWSM companion pass owns independent resources |
| 2 | [`ExposureMip`](../../shaders/pass/composite/ExposureMip.comp.glsl) | Builds luminance/weight levels |
| 3 | [`ExposureGather`](../../shaders/pass/composite/ExposureGather.comp.glsl) | Reduces histogram statistics when `SETTING_DEBUG_AE` is enabled |
| 4 | [`OverlayComposite`](../../shaders/pass/composite/OverlayComposite.comp.glsl) | Composes the overlay |

The exposure screen controls manual EV, minimum/maximum EV, emissive/distance/center weighting, average-luminance time
and target range, highlight/shadow percentiles, and adaptation parameters. `transient_exposureWeights` carries per-pixel
weights.

## Final display transform

Root [`final.fsh`](../../shaders/final.fsh) includes [
`Final.frag.glsl`](../../shaders/pass/composite/Final.frag.glsl). That entry point uses [
`Exposure.glsl`](../../shaders/techniques/displaytransform/Exposure.glsl), [
`DRT.glsl`](../../shaders/techniques/displaytransform/DRT.glsl), and [
`DisplayTransform.glsl`](../../shaders/techniques/displaytransform/DisplayTransform.glsl) to convert internal working
color space to DRT working space, then applies tone mapping and the output color space/transfer function.

After tone mapping, the result is converted to the configured color-grading color space. [`PrimaryColorCalibration`](../../shaders/techniques/displaytransform/PrimaryColorCalibration.glsl) adjusts its linear RGB primaries, then the configured grading transfer function is applied and [`HSLColorMixer`](../../shaders/techniques/displaytransform/HSLColorMixer.glsl) adjusts eight hue bands. The result is decoded and converted to the monitor output space. Both stages have independent toggles.

Related settings cover material transfer/color space, internal working space, DRT working space, color-grading color space/transfer function, tone-map look/dynamic range/offset/slope/power/saturation, and monitor color space/transfer function.

## Debug and special modes

Debug output in [`shaders/techniques/debug/`](../../shaders/techniques/debug/) can inspect TAA, PostFX, tone-mapping,
or final boundaries; [`SSTStepDebug`](../../shaders/pass/composite/SSTStepDebug.comp.glsl) is a conditional program.
Screenshot mode adjusts animation/temporal clamps, and video-render mode adjusts long-term temporal effects. They do not
define parallel production post-processing branches.

## Maintenance invariants

- Keep TAA history color space, pre-exposure, jitter, and motion/reprojection aligned.
- Keep the eight Bloom levels, layer weights, generated taps and atlas addressing consistent. Refit directional
  shapes if the raw level-1 carrier changes.
- Keep exposure production, adaptation and final display consumers synchronized.
- Validate display-transform changes numerically with color charts/gradients and across SDR, wide-gamut, and HDR
  configurations for clipping, NaN, and negative values.
- Test static detail, moving edges, disocclusion, bright emissives, dark adaptation, translucent edges, and UI/overlays.
