# Rendering Pipeline Dependencies

[`scripts/programs.main.kts`](../../scripts/programs.main.kts) owns the compute order, grouped entries, enable
conditions, and indirect dispatch. The module pages describe the resource contracts behind that order.
For Iris's stage semantics, see the [Iris Shaders documentation](https://github.com/IrisShaders/docs).

| Dependency | Why the order matters |
| --- | --- |
| Setup → Begin → shadow/geometry | Setup initializes global state and clears atlases by format. Begin prepares current and previous camera state, jitter, frame counters, allocations, and atmosphere LUTs before the frame consumes them. |
| Shadow geometry → Prepare → shadow sampling and voxel fallback | Prepare evaluates shadow-water normals and builds/propagates the voxel tree. Screen lighting uses the shadow data, while GI and other tracing paths use the voxel data. See [shadows](shadows.md) and [water](water-translucency.md). |
| G-buffer → scene preparation → direct lighting and GI | The [geometry paths](geometry-materials.md) supply depth, normals, material, and translucency. Optional Voxy merge precedes consumers. Hi-Z enables screen tracing. Denoiser edge classification and dilation precede history reprojection. Volumetric depth layers come from edge classification. |
| Caustics → direct lighting and cumulus render → accumulation → volume composition | Caustic photon tracing consumes screen pixel size and must remap its result before lighting samples it. The cloud history/upscale stage produces scattering and transmittance for later [volume composition](atmosphere-clouds.md). |
| Direct lighting, probe and radiance cache → ReSTIR → denoising | The probe gathers reprojected data and projects current-frame results. RC data is available before initial sample shading. ReSTIR produces diffuse/specular estimates, resolves deferred voxel visibility, and supplies [denoiser inputs](global-illumination.md#gi-denoising). |
| Air and water volumes → translucent back composite → SST → translucent composite | Air and water scattering use the prepared depth layers. The back composite provides the scene color for translucency tracing. Final composition combines the layers, with optional depth-break correction afterward. See [water and translucency](water-translucency.md). |
| Composed HDR color → DOF and AA → Bloom → post composition and exposure → final | DOF captures its input after translucency. AA resolves the color consumed by Bloom. Post composition adds Bloom and prepares exposure inputs. [The final entry](../../shaders/pass/composite/Final.frag.glsl) applies exposure and the display transform. See [post-processing](post-processing.md). |
| Current screen importance → RTWSM collapse/blur/get warp/write warp → next-frame shadows | RTWSM has independent resources alongside post-processing. Preserve the cross-frame producer/consumer order even when some programs share a dispatch slot with post passes. |

Root raster wrappers connect the geometry and final entries. The compute generator handles numbered `.csh` wrappers.
The shader pack sets `allowConcurrentCompute=true` in [`scripts/shaders.properties`](../../scripts/shaders.properties).
Entries with the same number and different suffixes may execute together. For example, `PostComposite` can share a
number with `IMapBlur` because they use independent data. Air and water can share slots with the corresponding
denoiser blur passes when spatial denoising is enabled. Their standalone branches preserve the volume work when it
is disabled.

See [compute pass registration](../development/generators.md#add-a-compute-pass-controlled-by-a-setting) for source
ownership and the generation command.
