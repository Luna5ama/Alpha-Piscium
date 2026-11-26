# Post-processing and Display Transform

Post-processing begins after GI, air/water volumes, and translucency are composed. The main chain is optional DOF,
metering, temporal resolve, spatial antialiasing/sharpening, camera exposure, Bloom, the display transform, overlay and final output.

## Code map

| Path | Responsibility |
| --- | --- |
| [`shaders/techniques/DOF.glsl`](../../shaders/techniques/DOF.glsl) | Shared DOF sampling and circle-of-confusion logic |
| [`DOFFocus.comp.glsl`](../../shaders/pass/composite/DOFFocus.comp.glsl), [`DOFPrepare.comp.glsl`](../../shaders/pass/composite/DOFPrepare.comp.glsl) | Automatic focus and DOF input preparation |
| [`TAAPrepare.comp.glsl`](../../shaders/pass/composite/TAAPrepare.comp.glsl), [`TAAResolve.comp.glsl`](../../shaders/pass/composite/TAAResolve.comp.glsl) | Temporal-AA preparation and resolve |
| [`FXAA.comp.glsl`](../../shaders/pass/composite/FXAA.comp.glsl) | Spatial antialiasing |
| [`RCAS.comp.glsl`](../../shaders/pass/composite/RCAS.comp.glsl), [`techniques/ffx/fsr1/`](../../shaders/techniques/ffx/fsr1/) | RCAS sharpening |
| [`techniques/Bloom.comp.glsl`](../../shaders/techniques/Bloom.comp.glsl) | Bloom downsample/upsample pyramid |
| [`Exposure.glsl`](../../shaders/techniques/displaytransform/Exposure.glsl), [`ExposureGather.comp.glsl`](../../shaders/pass/composite/ExposureGather.comp.glsl) | Weighted log-luminance histogram and camera adaptation |
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

The current shaderpack runs these passes in order.

| Order | Pass | Purpose |
| --- | --- | --- |
| 1 | [`TAAPrepare`](../../shaders/pass/composite/TAAPrepare.comp.glsl) | Meters unexposed post-DOF scene color and encodes the temporal input |
| 2 | [`TAAResolve`](../../shaders/pass/composite/TAAResolve.comp.glsl) | Uses `history_taa`, motion/reprojection data, and current-frame input to write `transient_taaOutput` |
| 3 | [`FXAA`](../../shaders/pass/composite/FXAA.comp.glsl) | Saves the unexposed scene-linear TAA history and writes `transient_fxaaOutput` |
| 4 | [`RCAS`](../../shaders/pass/composite/RCAS.comp.glsl) | Sharpens, decodes to scene linear, and applies the common camera EV |

TAAPrepare and TAAResolve also run with TAA disabled. Iris enables FXAA and RCAS only for TAA. On both frame parities, TAAResolve finishes reading the previous `history_taa` tile before FXAA writes the new scene-linear history to the same tile, at (width, 2 × height) in the RGBA16F image.

TAA settings cover enable, jitter, current/history filters, and CAS sharpness. Current/previous-jitter custom uniforms
are generated from the R2 frame sequence in [`scripts/shaders.properties`](../../scripts/shaders.properties);
changing the sampling sequence requires updating reprojection conventions as well.

### 2.1 时域重建策略对照

2.1 原生 TAA 与超分的设计由重建阶段写轮换的未锐化历史，撤下上表的 FXAA 兜底。输出锐化直接使用重建的颜色、覆盖率与矩。自动曝光沿用重建前的场景测光，Bloom 读取按相机 EV 缩放的未锐化场景均值。可信轮廓的两侧分别经过 DRT 与 Look，再按同一输出足迹的覆盖率混合。[覆盖率与矩参与的锐化](temporal-reconstruction.md#覆盖率与矩参与的锐化) · [亚像素灯带的显示混色](temporal-reconstruction.md#亚像素灯带的显示混色)

## Bloom

`SETTING_BLOOM_SHAPE` offers Standard (0, default), Diamond Four (1), Horizontal Streak (2), Cross Four (3),
Camera Six (4), Camera Ten (5), Cinema Star (6), Prismatic Streak (7), 67 (8), and Trollface (9).
Standard is an eight-scale, long-tailed Gaussian approximation with a concentrated core. Each scale contributes
1/8 of the normalized filter; it is not a single Gaussian. The fixed eight levels preserve the broad halo that
was lost in the rejected four-level prototype.

The filters use offline-fitted offsets and nonnegative weights in
[`BloomTaps.glsl`](../../shaders/techniques/BloomTaps.glsl). Fitting accounts for decimation phases and bilinear
sampling, following the [DPF approach](https://huakaifugui2.github.io/CZ-DPF-Paper/). Runtime uses fixed RGB filters
and a fixed pyramid depth. The four-scale fitting proxy calibrates reusable taps; it does not set the runtime depth.

| Order | Pass/stage | Purpose |
| --- | --- | --- |
| 1 | [`Bloom`](../../shaders/techniques/Bloom.comp.glsl), downsample levels 1–8 | Five bilinear reads per output texel. The first level reads the decoded linear main image and applies Bloom-only highlight compression. |
| 2 | [`BloomShape`](../../shaders/pass/composite/BloomShape.comp.glsl), optional | Initializes the selected tap table in workgroup shared memory, then samples raw level 1 in a bounded loop and writes a separate half-resolution image. Standard and Bloom Off disable this pass. |
| 3 | [`Bloom`](../../shaders/techniques/Bloom.comp.glsl), reconstruct levels 8–2 | Eight bilinear reads per output texel. The deepest contribution and each finer downsample receive weight 1/8; the invocation reads the finer texel that it then overwrites. |
| 4 | [`PostComposite`](../../shaders/pass/composite/PostComposite.comp.glsl) | Fuses the last upsample, adds the optional shape with centered bilinear reconstruction, and applies Bloom intensity and underwater boost. |

[`programs.main.kts`](../../scripts/programs.main.kts) owns this order and its enable conditions.
[`Bloom.glsl`](../../shaders/techniques/Bloom.glsl) owns common addressing and sampling. Downsample and shape reads
zero-extend their bilinear footprint at screen edges while keeping atlas reads inside the input tile. Upsampling
clamps to the valid input mip. This prevents a long streak from repeatedly copying a bright edge texel.

- `transient_bloom`: frame-local RGBA16F atlas with ceil-sized padded mip tiles; reconstructed in place from coarse
  to fine.
- `transient_bloomShape`: frame-local RGBA16F logical image; only its half-resolution region is written and read.
  It lives from the shape pass through post composition. The allocation in
  [`shadesmith.json`](../../shaders/shadesmith.json) reuses a slot after FXAA/RCAS have finished with it, without
  increasing the current physical atlas dimensions. It remains a separate logical resource from the pyramid.

Each shape adds energy over the full Standard core and tail. Camera and Prismatic shapes spread each input channel
with its own weights. The two novelty kernels align their visual bounds to the light and are most legible around
small emitters. Larger emitting surfaces broaden the shapes and can fill fine gaps. Parameters and custom
PNG/function input are maintained by
[`fit-bloom-shape.py`](../../scripts/fit-bloom-shape.py); see [kernel fitting](../development/offline-tools.md#bloom-kernel-fitting).

Highlight compression belongs to the first Bloom downsample, after AA decoding. RGB and luma modes retain their
existing curves. AA uses fixed log encoding with an exact zero; decoded writes in
[`RCAS`](../../shaders/pass/composite/RCAS.comp.glsl) and the TAA-off path of
[`TAAResolve`](../../shaders/pass/composite/TAAResolve.comp.glsl) clamp to the RGBA16F limit of 65504.

### Bloom validation and limits

Shader snapshot `0ed64baf9a70…` rendered all ten modes and Bloom Off at 1280×720 in Small lamps night (EV 9.45),
outdoor point lights (EV 11.07), and Craft close lights (EV 7.34). Camera and world time were fixed within each
scene, with 192 warmup frames per mode. Camera Six produced finite, nonzero shape output with TAA disabled.
Bloom Off suppressed the first downsample and shape passes with Trollface selected.

On RTX 3080 Ti in Small lamps night, two rounds bracketed the forward and reverse shape orders with Standard.
Each window used 96 warmup frames and 64 sampled frames. GPU times are in milliseconds.

| Mode | Shape pass p50 | Bloom + complete PostComposite p50 / p95 |
| --- | ---: | ---: |
| Standard | — | 0.191 / 0.201 |
| Diamond Four | 0.254 | 0.455 / 0.471 |
| Horizontal Streak | 0.094 | 0.289 / 0.296 |
| Cross Four | 0.193 | 0.386 / 0.396 |
| Camera Six | 0.357 | 0.553 / 1.114 |
| Camera Ten | 0.399 | 0.593 / 1.234 |
| Cinema Star | 0.263 | 0.460 / 0.831 |
| Prismatic Streak | 0.179 | 0.375 / 0.397 |
| 67 | 0.199 | 0.395 / 0.411 |
| Trollface | 0.311 | 0.507 / 0.992 |

The combined column sums the aligned program samples per frame before calculating quantiles. The unchanged
volumetric-composite reference pass had a 0.049 ms median in every mode.

Independent CPU impulse models support a near-radial standard kernel: at radii 12, 24, 48 and 96 pixels, the
eight-phase mean fourfold angular component was about 1.0–1.9% of ring brightness, versus 2.6–3.2% for the previous
COD filter. Dyadic sampling still has pixel-grid and phase error; this is an approximation, not exact rotational
symmetry. Odd viewport sizes, boundary energy and phase centroids have CPU coverage; odd-size runtime, underwater
rendering and other GPUs remain unmeasured. Shape fitting error describes the added raw-level-1 branch, not the
complete Bloom result.

## Camera exposure

[`TAAPrepare`](../../shaders/pass/composite/TAAPrepare.comp.glsl) accumulates a 256-bin weighted histogram of post-DOF scene luminance over 40 stops starting at −24 EV. Distance, emissive and center weights feed workgroup-local bins and then the frame SSBO. [`ExposureGather`](../../shaders/pass/composite/ExposureGather.comp.glsl) trims each tail by cumulative weight, retains partial boundary bins and computes the weighted mean log luminance. The target gray rises from the night setting to the daylight setting. The reducer clamps the resulting EV, updates temporal adaptation, then leaves the histogram empty for the next frame.

Default post-exposure gray is 8/255 (0.031) for the darkest scenes and 40/255 (0.157) for daylight, in scene-linear units before Bloom and the DRT. The camera blends between these targets using metered log luminance from −13 to −2 EV and clamps its gain to −6.5 through +10 EV. The bright target sets the level for the retained log average. Individual sunlit surfaces can be much brighter.

Reprojection reads unexposed scene-linear history from the previous frame. FXAA stores the new history before spatial filtering. RCAS applies the camera multiplier after reconstruction. The TAA-disabled path applies the same multiplier in TAAResolve. Bloom and all four DRTs therefore see the same camera exposure. The histogram debug view shows the retained frame's bin weights, target EV and camera EV.

## Display transform

[`PostComposite`](../../shaders/pass/composite/PostComposite.comp.glsl) adds Bloom and calls [`DRT`](../../shaders/techniques/displaytransform/DRT.glsl). The default is Skibidi v41 with neutral Look and Display P3 target primaries. SDR grading retains the primary calibration and HSL mixer, converts to the chosen output primaries and transfer function, then applies the common Look to encoded RGB. Final output dithers SDR to eight bits.

| DRT | Input and reference | Target gamut |
| --- | --- | --- |
| Skibidi v41 | Linear AP0. The source model and SDR/SDR-style HDR curves come from drt-bench `280aa5e31fdfb412bea24f9746b9417fedfd67d5`. | Native radial allocation in the chosen D65 RGB space. |
| Blender AgX | Blender 5.0.0 `a37564c4df7a9b604809e5192918400e46a0205f` OCIO transforms from ACES2065-1 to sRGB AgX or Rec.2100-HLG AgX HDR 1000 nits. The 57³ LUTs use tetrahedral interpolation. | View-defined primaries. |
| OpenDRT | Standard v1.1.0 from the same drt-bench revision, via XYZ. | Uses the selected display basis for its output rendering. |
| Psychov30 | RenoDX `6d563066956190c8b9a2c7a907398bfac3629a3c` test30, via linear BT.709 with white-point adaptation. | Uses the existing 709 or 2020 operator. Other targets use the native SDR or HDR default. |

For Skibidi, preset D65 and user-entered primary coordinates feed the same gamut matrix. An invalid triangle falls back to Display P3. Rec.709 uses the pinned v41 scalar fields. For other targets [`SkibidiGamut`](../../shaders/pass/setup/SkibidiGamut.comp.glsl) generates the per-hue boundary seeds, red cusp and EO12 fit when Iris runs setup. Its coefficients and weighted bins occupy 670,728 bytes after the 16 KiB global block in SSBO 0. The persistent 145 × 81 tile begins at (0, 256) in the fixed R32F image. Expanding that image from 256 × 264 to 256 × 345 costs 82,944 bytes. One 256-lane workgroup carries the four setup phases. Each workgroup barrier follows a group memory barrier. The fitter reads the newly stored boundary fields through coherent image loads with bilinear interpolation. The steady-state display pass uses the filtered sampler. This allows the existing frame passes to use concurrent compute.

When HDR Mod supplies both presence and enable macros, DRTs read game paper white and peak in nits. The Look uses signed sRGB relative to game paper white, including pixels above white. At handoff AP scales by game paper white divided by UI white, then applies signed sRGB encoding in BT.709 primaries. The mod consumes floating-point output and handles monitor primaries and transfer. The SDR path applies whenever either macro is absent. [HDR Mod shaderpack interface](https://github.com/rrtt217/Minecraft-HDR-Mod/wiki/For-Shaderpack-Devs)

### 2.1 validation record

The fixed-input comparison uses the three ACES sample frames 0017, 0061 and 0067 plus gray, primary and signed probes, at −2, 0 and +2 EV in AP0, neutral Look, sRGB output and Rec.709 target. Across these exposures the largest 8-bit channel difference was one code for Skibidi and Blender AgX, and zero for OpenDRT and Psychov30. Blender uses the user's chosen 5.0 view. Rebuild its textures and shaders with OpenColorIO 2.6.0 and [the AgX generator](../../scripts/generate-agx.py). The Blender color configuration credits Troy Sobotka, Zijun Eary Zhou, Mark Faderbauer and Sakari Kapanen. [Pinned Blender configuration](https://github.com/blender/blender/blob/a37564c4df7a9b604809e5192918400e46a0205f/release/datafiles/colormanagement/config.ocio)

In an isolated GL dispatch, preset P3 and manual (680,320), (265,690), (150,60) in thousandths produced identical tables and SDR/HDR sample values on llvmpipe. The single-workgroup P3 bake differed from the four-dispatch bake by at most 0.00000346 in the EO12 table. All 54 packaged GLSL variants compiled, including manual exposure, TAA off, custom target, and absent, disabled and enabled HDR interfaces. The 65 × 37 weighted-histogram fixture yielded +7.355 EV for the dark uniform scene and the same scene with a small bright emitter, −0.456 EV for bright snow, and +6.668 EV for a dark cave with a bright exit. These values describe the synthetic fixture.

Minecraft 1.21.11, Iris 1.10.6-vibris.3, HDR Mod 2.3.4 and RTX 3080 Ti ran the 1280 × 720 night arcade scene with manual EV 0 and Skibidi targeting Display P3. Both captures used the four-dispatch implementation in candidate `f4213de6508a678531ba8fa0a7da6d30c684d533` with a diagnostic strip inserted before the production DRT. The recovered game paper white, peak and UI white were 203, 1000 and 284 nits respectively. Neutral gray rose monotonically to 998.7 nits. The closest sampled code to paper white decoded to 203.54 nits. In 162 colored highlight samples above paper white the median largest-channel change from neutral to Punchy was 708.3 nits. These are shader handoff values reconstructed from RGBA16F. The SSBO readback verified manual EV 0 in both jobs and a 687,112-byte allocation at binding 0. Against software GL on the same chart and uniform values, the largest differences in signed encoded RGB were 0.000981 for neutral and 0.001961 for Punchy. For every DRT, the installed-but-disabled and missing-mod shader variants gave identical software output for both Looks. The generated pack has the same 16 physical image declarations as its 2.1 base.

On `b384fe2ae24a39329034a8d9c71d6215b32ea819`, Iris dispatched the single-workgroup gamut bake. Its standalone GPU time on the RTX 3080 Ti was 6.66 ms for Display P3 on setup. With the same chart, EV 0, neutral Look and 203/1000 nit scene settings, changing only the mod's UI white from 284 to 160 nits gave 998.125 and 997.500 nits at the brightest gray. The same gray patch near paper white measured 205.469 nits in both window captures. Across 480 gray and colored samples the median largest-channel change was 0.039 nits and the maximum was 1.875 nits. These values come from FP16 Windows Graphics Capture of the dedicated game window on a 10-bit PQ/Rec.2020 DXGI output. They measure the composited signal. The capture and patch coordinate matching are recorded in `compositor-result.json` and `compare-compositor.py`.

The same game also displayed the fixed EV 0 diagnostic chart on the flatroom preset with HDR Mod missing (140 loaded mods) and installed but disabled (142). Both composited windows reached 283.906 nits at the brightest gray. Against the 284-nit SDR reference the 95th-percentile absolute channel errors across the chart were 1.75 and 2.41 nits respectively. The largest difference between the two window captures was 4.22 nits. Both window captures use the HDR measurement's patch coordinates. Vibris timed out waiting for terrain stability on the missing-mod texture dump. With the mod installed and disabled, a settled-frame RGBA16F readback on `c651b6c0` agreed with the software SDR chart to 0.000489 in encoded RGB at manual EV 0. See `sdr-window-results.json`, the retained FP16 windows and both loader records.

In the same Iris version, Skibidi, Blender AgX, OpenDRT, Psychov30 and Skibidi with Bloom intensity raised by four stops all returned the identical 32-bit target EV (+2.6045022) and metered log luminance (−4.1524277). This test fixed the linear input and weights at the histogram entrance. The game loaded each DRT and SSBO 0 recorded the result. In separate unpatched 720p scenes, the torch-free deep wall in Dump Taiga adapted to +6.60 EV (target +6.67), the daylight cave exit to +0.30 EV and the small-lamp gallery to −2.03 EV. The dark wall's central 10th/50th/90th percentiles in encoded SDR luma were 0.210/0.314/0.434. Its surface pattern remained visible.

The default low/high trims are 3%/5% by weight. In the cave exit the weighted mean log luminance was −1.54 without trimming, −1.71 at 3%/5%, and −1.84 at 5%/10%. The same choices gave +0.73, +0.63 and +0.48 in the small-lamp gallery. On the deep wall, 3%/5% changed the untrimmed result by 0.024 stop. In the initial 96/255 daylight calibration, daytime snow reached the −3 EV floor and 29.70% of the frame had encoded luma at or above 0.99. Testing a −5 EV floor allowed snow to settle at −4.18 EV, leaving 0.0054% at or above 0.99. Median sunlit snow was 0.959 and shaded snow measured 0.754. The comparison uses the same camera, sun time and snow patches. See `snow-contrast.json`, the `*-scene.json` histogram readbacks and the matching scene PNGs.

At 96/255 with a −5 EV floor (`c75a8dc61`), daylight surfaces in the 2.1 SDR scene gallery looked washed out even after the camera settled. At 40/255 with a −6.5 EV floor (`0c2a23af3`), the noon courtyard went from −2.07 to −3.34 EV. The area at or above 0.95 encoded luma fell from 15.58% to 0.58%. Median encoded luma was 0.53 previously, 0.35 after tuning, and 0.27 in the earlier default AgX rendering (`792ffe98a`). In the snow preset, sunlit and shaded patches measured 0.887 and 0.570, previously 0.957 and 0.759. The midnight courtyard moved from +7.06 to +6.62 EV.

The six matched 1280 × 720 SDR pairs use Minecraft 1.21.11, Iris 1.10.6-vibris.3, Skibidi, neutral Look, and automatic exposure. They cover day and night in the courtyard, snow, the cave mouth, an indoor mall and a mountain lake. Each pair has the same scene and preset hashes.

The main checkout retains Vela run `c23818a2-c6d0-4fac-99a9-53e6566c3e36` in `builds/exposure-target-20261123/exposure-review.zip` (SHA-256 `f8a19d862694e1c495898411588162f8d58ebf124c835fca7052228763927129`). The archive contains the photographs, effective settings and camera readbacks, four recorded histograms replayed through both shader revisions, and the earlier courtyard reference. Open `comparison.html` from the archive for the six pairs.

<!-- vela:evidence {"run":"c23818a2-c6d0-4fac-99a9-53e6566c3e36","path":"exposure-review.zip"} -->

### GPU timings

On the 720p Dump Taiga snow preset, eight runs compared the former default on `792ffe98a` with the 2.1 default on `c651b6c0`. Each direction included 80 warmup frames and 96 measured frames. Values below are the medians of the per-run medians in milliseconds. Matched PresentMon windows each contained 92 frames after a 40 ms guard at both ends.

| GPU measurement | Earlier default | 2.1 default |
| --- | ---: | ---: |
| TAA preparation, including 2.1 metering | 0.0123 | 0.0174 |
| Display transform and Bloom combine | 0.0369 | 0.0934 |
| Exposure mip or histogram reduction | 0.0174 | 0.0164 |
| Selected post-processing programs, summed per frame | 0.5210 | 0.5978 |
| Terrain solid sentinel | 0.1116 | 0.1111 |
| Whole-frame GPU time | 21.414 | 21.554 |
| Whole-frame GPU busy time | 17.844 | 17.950 |

A separate 16-run forward/reverse matrix used the same frozen shader snapshot at manual EV 0. The display program took 0.0942 ms for Skibidi, 0.0369 for Blender AgX, 0.0389 for OpenDRT and 0.0338 for Psychov30. The program includes the final Bloom combine. Raw case receipts, frozen snapshot identities, PresentMon samples and the reducer are in `native-profiles.zip`. The independent gamut dispatch on the RTX 3080 Ti measured neutral narrow-gamut linear SDR channels from −0.00000003 to 1.00023 after transport and conversion back to the target. Punchy expanded that range to −0.312 to 1.234. See `gpu-geometry.json` for HDR and the preset and custom P3 results.

The AgX choice image, input identifiers and baseline script are in project Vela run `16da9e19-35a7-47c4-a356-a2884c76c57f`. Run `d2481554-868d-4549-9ccc-efb659257cee` contains the numerical comparison, generated tables, histogram tests, game chart, RGBA16F/SSBO readbacks and game receipts. The fit provenance files are `ordered.py` (SHA-256 `c5c571797f8df7293b7617edfc79febaac41f8aa6d5bb51215ff0b3f76bf7268`) and `exposure_curves.py` (`1aa03bb96c213137b76a5785cf00547c57cd81532b58083342376d0ba21ee7ef`). Raw files are under `.vela/local/runs/<run UUID>/` in the main project checkout. The hashed game texture and buffer captures for the exposure tests are in `native-captures.zip`. The disabled-mod chart is in `sdr-verified-strip.npy`.

### Skibidi 色域 LUT 的重载成本

2.1 候选 `b384fe2ae` 在 RTX 3080 Ti、Minecraft 1.21.11、Iris `1.10.6-vibris.3` 上启用默认 Skibidi 和 Display P3。相同街机场景的两次整包 `shader.reload` 用了 21.31 s 和 17.03 s。独立 OpenGL 时间查询在程序编译链接后开始，P3 制表单次派发用 6.66 ms。旧四阶段实验由宿主显式同步，四次派发合计 0.134 ms。AP 开启并发计算，合并阶段的候选用一个 256 线程工作组同步表的生产与读取。重设工作组数量前，需解决阶段间图像和 SSBO 的可见性。原始查询在项目运行 `d2481554-868d-4549-9ccc-efb659257cee/nvidia-{single,fixed}.log`。

候选实验仍按显示器原色在 setup 中制表。192 个色相、每个色相 15 个饱和度各自取 79 个亮度样本，同一方向的样本共用 11 个插值系数。枚举这些方向仅访问 7,203 个网格节点中的 4,168 个，原型在格线附近多留 12 个节点。需要拟合的控制点由 79,233 减至 45,980，227,520 个训练样本全部保留。将四邻点的系数按方向各读一次，源码中的控制点读取由 10,010,880 次减至 126,720 次。P3、Rec.2020、手填 P3 和窄色域在 llvmpipe 上逐项对照，每个目标共 9,417 个有效 LUT 表值，最大绝对差为 `1.67e-6`。`builds/skibidi-setup-20261122/check_fit.py` 可重做软件对照，`software.json` 保存四组结果。

**2.1 保留现行 dense 制表。** 在 RTX 3080 Ti、NVIDIA 617.42、无头 OpenGL 4.6 上，按方向拟合的 rays 将默认 P3 的制表派发从 6.77 ms 缩短到 4.59 ms。两种程序在初始化或重设目标时各派发一次。首次观测的 P3 驱动链接却从 599 ms 增到 755 ms，单次派发节省的约 2 ms 无法抵消这次链接增加的 156 ms。

| 目标 | GLSL 编译，dense / rays（ms） | 首次观测的链接，dense / rays（ms） | 重开进程后的链接，dense / rays（ms） | GPU 派发，dense / rays（ms） |
| --- | ---: | ---: | ---: | ---: |
| Display P3 | 36.91 / 8.62 | 598.93 / 755.25 | 2.39 / 4.36 | 6.77 / 4.59 |
| Rec.2020 | 9.05 / 11.20 | 807.78 / 1070.44 | 3.36 / 2.99 | 7.00 / 4.52 |
| 手填 P3 | 10.21 / 8.69 | 866.97 / 1114.45 | 2.94 / 3.15 | 6.19 / 4.16 |
| 手填窄色域 | 8.79 / 8.79 | 918.85 / 1156.79 | 3.57 / 5.20 | 7.70 / 4.75 |

编译与链接分别计至驱动返回状态。首个 P3 编译包含该进程首次使用编译器的开销。表中派发为两个进程对冻结源码做正反序配对的 GPU 时间中位数，每种实现、每个目标各 18 次，另存了首次派发时间。两个额外进程先预热编译器，再只改图像 uniform 的显式位置，运行时仍绑定同一图像。交替先后顺序后，两个新位置下 rays 的首次链接均比 dense 慢，四个目标的差值为 121–301 ms。以上时间只覆盖 setup 程序。

每个目标读回 9,417 个有效 FP32 表项。对同一目标，dense 和 rays 的结果逐位相同，重复派发和显式位置对照也得到相同结果。另将预设 P3 与手填相同原色单独比较，两种实现在该卡上都出现了最大 `8.29e-6` 的表项差异。原始逐次计时、四目标 LUT、冻结输入及复核脚本在 Vela 运行 `465e2b57-b291-4cb8-a246-e073f2f3db30` 的 `skibidi-setup-ab.zip`。复核时可在解包目录执行 `python summarize.py <f2dbfadcc 检出目录>`。

<!-- vela:evidence {"run":"465e2b57-b291-4cb8-a246-e073f2f3db30","path":"skibidi-setup-ab.zip"} -->

2026-11-22 17:00（洛杉矶）关闭的专用实例运行云实验 `031e136d5` 加 `c66921dd` 补丁。冻结源码中的 `setup23` 清空体素数据，包内没有 Skibidi 程序。Vibris 重试同一快照后，单独记录到重载 64.30 s、应用场景 16.64 s。云分支早先在同卡回放另一快照 `7f787bf60`，五个程序的驱动链接合计 73.79 s。此次中断缺逐程序计时，具体调用仍待定位。源包和重试回执在云检出 `.vibris/jobs/100ff5af-0088-47b8-b646-64ef2fd0c924/`，现场摘录在主检出 `builds/skibidi-setup-20261122/incident.json`。

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
