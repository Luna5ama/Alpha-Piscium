/*
    References:
        [MCA23] Aldridge, Graham. "Screen Space Shadows". Siggraph 2023.
            Apache License 2.0. Copyright 2023 Sony Interactive Entertainment.
            https://www.bendstudio.com/blog/inside-bend-screen-space-shadows/

        You can find full license texts in /licenses
*/
#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_vote : enable
#extension GL_KHR_shader_subgroup_clustered : enable
#extension GL_KHR_shader_subgroup_ballot : enable

#include "/techniques/HiZCheck.glsl"
#include "/util/Coords.glsl"
#include "/util/Material.glsl"

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(rgba8) uniform writeonly image2D uimg_rgba8;

const uint BEND_SSS_PARAMS_BASE = 128u;

// Bend Studio SSS Implementation
#define WAVE_SIZE 64
#define SAMPLE_COUNT 60
#define HARD_SHADOW_SAMPLES 4
#define FADE_OUT_SAMPLES 8
#define READ_COUNT ((SAMPLE_COUNT + WAVE_SIZE - 1) / WAVE_SIZE + 2)

shared float DepthData[READ_COUNT * WAVE_SIZE];
shared bool LdsEarlyOut;

struct DispatchParameters {
	// Visual configuration:
	// These values will require manual tuning.
	// All shadow computation is performed in non-linear depth space (not in world space), so tuned value choices will depend on scene depth distribution (as determined by the Projection Matrix setup).

	float SurfaceThickness;				// This is the assumed thickness of each pixel for shadow-casting, measured as a percentage of the difference in non-linear depth between the sample and FarDepthValue.
										// Recommended starting value: 0.005 (0.5%)

	float BilinearThreshold;			// Percentage threshold for determining if the difference between two depth values represents an edge, and should not perform interpolation.
										// To tune this value, set 'DebugOutputEdgeMask' to true to visualize where edges are being detected.
										// Recommended starting value: 0.02 (2%)

	float ShadowContrast;				// A contrast boost is applied to the transition in/out of shadow.
										// Recommended starting value: 2 or 4. Values >= 1 are valid.

	bool IgnoreEdgePixels;				// If an edge is detected, the edge pixel will not contribute to the shadow.
										// If a very flat surface is being lit and rendered at an grazing angles, the edge detect may incorrectly detect multiple 'edge' pixels along that flat surface.
										// In these cases, the grazing angle of the light may subsequently produce aliasing artefacts in the shadow where these incorrect edges were detected.
										// Setting this value to true would mean that those pixels would not cast a shadow, however it can also thin out otherwise valid shadows, especially on foliage edges.
										// Recommended starting value: false, unless typical scenes have numerous large flat surfaces, in which case true.

	bool UsePrecisionOffset;			// A small offset is applied to account for an imprecise depth buffer (recommend off)


	bool BilinearSamplingOffsetMode;	// There are two modes to compute bilinear samples for shadow depth:
										// true = sampling points for pixels are offset to the wavefront shared ray, shadow depths and starting depths are the same. Can project more jagged/aliased shadow lines in some cases.
										// false = sampling points for pixels are not offset and start from pixel centers. Shadow depths are biased based on depth gradient across the current pixel bilinear sample. Has more issues in back-face / grazing areas.
										// Both modes have subtle visual differences, which may / may not exaggerate depth buffer aliasing that gets projected in to the shadow.
										// Evaluating the visual difference between each mode is recommended, then hard-coding the mode used to optimize the shader.
										// Recommended starting value: false

	// Debug views
	// bool DebugOutputEdgeMask;			// Use this to visualize edges, for tuning the 'BilinearThreshold' value.
	// bool DebugOutputThreadIndex;		// Debug output to visualize layout of compute threads
	// bool DebugOutputWaveIndex;			// Debug output to visualize layout of compute wavefronts, useful to sanity check the Light Coordinate is being computed correctly.

	// Culling / Early out:
	vec2 DepthBounds;					// Depth Bounds (min, max) for the on-screen volume of the light. Typically (0,1) for directional lights. Only used when 'UseEarlyOut' is true.

	bool UseEarlyOut;					// Set to true to early-out when depth values are not within [DepthBounds] - otherwise DepthBounds is unused
										// [Optionally customize the 'EarlyOutPixel()' function to perform your own early-out logic, e.g. skipping pixels that a shadow map indicates are already fully occluded]
										// This can dramatically reduce cost when only a small portion of the pixels need a shadow term (e.g., cull out sky pixels), however it does have some overhead (~15%) in worst-case where nothing early-outs
										// Note; Early-out is most efficient when WAVE_SIZE matches the hardware wavefront size - otherwise cross wave communication is required.

    vec4 LightCoordinate;
    ivec2 WaveOffset;
    float FarDepthValue;
    float NearDepthValue;
    vec2 InvDepthTextureSize;
};

float GetScreenDepth(ivec2 texel) {
    if (any(greaterThanEqual(uvec2(texel), uvec2(uval_mainImageSizeI)))) {
        return 0.0;
    }
    float viewZ = texelFetch(usam_gbufferSolidViewZ, texel, 0).r;
    if (viewZ <= -65536.0) return 0.0; // Far

    return coords_viewZToReversedZ(viewZ, nearPlane);
}

vec2 GetScreenDepthPair(ivec2 texel, ivec2 offset) {
    ivec2 neighbor = texel + offset;
    bool inBounds = all(lessThan(uvec2(texel), uvec2(uval_mainImageSizeI))) &&
        all(lessThan(uvec2(neighbor), uvec2(uval_mainImageSizeI)));
    if (!inBounds) {
        return vec2(GetScreenDepth(texel), GetScreenDepth(neighbor));
    }

    ivec2 gatherTexel = min(texel, neighbor);
    vec2 gatherUV = (vec2(gatherTexel) + 1.0) * uval_mainImageSizeRcp;
    vec4 gatheredViewZ = textureGather(usam_gbufferSolidViewZ, gatherUV, 0);
    bool horizontal = offset.x != 0;
    bool positive = horizontal ? offset.x > 0 : offset.y > 0;
    vec2 viewZ = horizontal ? gatheredViewZ.wz : gatheredViewZ.wx;
    if (!positive) {
        viewZ = viewZ.yx;
    }

    vec2 depth = nearPlane / -viewZ;
    return mix(vec2(0.0), depth, greaterThan(viewZ, vec2(-65536.0)));
}

void ComputeWavefrontExtents(DispatchParameters params, ivec3 groupID, uint laneID, out vec2 outDeltaXY, out vec2 outPixelXY, out float outPixelDistance, out bool outMajorAxisX) {
    ivec2 xy = groupID.yz * WAVE_SIZE + params.WaveOffset;

    vec2 light_xy = floor(params.LightCoordinate.xy) + 0.5;
    vec2 light_xy_fraction = params.LightCoordinate.xy - light_xy;
    bool reverse_direction = params.LightCoordinate.w > 0.0;

    ivec2 sign_xy = ivec2(sign(xy));
    bool horizontal = abs(xy.x + sign_xy.y) < abs(xy.y - sign_xy.x);

    ivec2 axis;
    axis.x = horizontal ? sign_xy.y : 0;
    axis.y = horizontal ? 0 : -sign_xy.x;

    xy = axis * groupID.x + xy;
    vec2 xy_f = vec2(xy);

    bool x_axis_major = abs(xy_f.x) > abs(xy_f.y);
    float major_axis = x_axis_major ? xy_f.x : xy_f.y;
    float major_axis_start = abs(major_axis);

    float ma_light_frac = x_axis_major ? light_xy_fraction.x : light_xy_fraction.y;
    ma_light_frac = major_axis > 0.0 ? -ma_light_frac : ma_light_frac;

    vec2 start_xy = xy_f + light_xy;
    vec2 xy_delta = (xy_f - light_xy_fraction) * (float(WAVE_SIZE) / (major_axis_start + ma_light_frac));

    float thread_step = float(laneID ^ (reverse_direction ? 0u : uint(WAVE_SIZE - 1)));

    vec2 pixel_xy = fma(xy_delta, vec2(-thread_step / float(WAVE_SIZE)), start_xy);
    float pixel_distance = major_axis_start - thread_step + ma_light_frac;

    outPixelXY = pixel_xy;
    outPixelDistance = pixel_distance;
    outDeltaXY = xy_delta;
    outMajorAxisX = x_axis_major;
}

void WriteScreenSpaceShadow(DispatchParameters params, ivec3 groupID, uint laneID) {
    vec2 xy_delta;
    vec2 pixel_xy;
    float pixel_distance;
    bool x_axis_major;

    ComputeWavefrontExtents(params, groupID, laneID, xy_delta, pixel_xy, pixel_distance, x_axis_major);

    float direction = -params.LightCoordinate.w;
    float z_sign = params.NearDepthValue > params.FarDepthValue ? -1.0 : 1.0;

    bool skip_pixel = false;
    vec2 write_xy = floor(pixel_xy);

    vec2 readXY0 = floor(pixel_xy);
    float minorAxis0 = x_axis_major ? pixel_xy.y : pixel_xy.x;
    float bilinear0 = fract(minorAxis0) - 0.5;
    int bias0 = bilinear0 > 0.0 ? 1 : -1;
    ivec2 offsetXY0 = ivec2(x_axis_major ? 0 : bias0, x_axis_major ? bias0 : 0);
    float sampleDistance0 = pixel_distance;
    pixel_xy += xy_delta * direction;

    vec2 readXY1 = floor(pixel_xy);
    float minorAxis1 = x_axis_major ? pixel_xy.y : pixel_xy.x;
    float bilinear1 = fract(minorAxis1) - 0.5;
    int bias1 = bilinear1 > 0.0 ? 1 : -1;
    ivec2 offsetXY1 = ivec2(x_axis_major ? 0 : bias1, x_axis_major ? bias1 : 0);
    float sampleDistance1 = pixel_distance + float(WAVE_SIZE) * direction;
    pixel_xy += xy_delta * direction;

    vec2 readXY2 = floor(pixel_xy);
    float minorAxis2 = x_axis_major ? pixel_xy.y : pixel_xy.x;
    float bilinear2 = fract(minorAxis2) - 0.5;
    int bias2 = bilinear2 > 0.0 ? 1 : -1;
    ivec2 offsetXY2 = ivec2(x_axis_major ? 0 : bias2, x_axis_major ? bias2 : 0);
    float sampleDistance2 = pixel_distance + float(WAVE_SIZE * 2) * direction;

    vec2 depthPair0 = GetScreenDepthPair(ivec2(readXY0), offsetXY0);
    float samplingDepth0 = depthPair0.x;

    // Early out logic
    if (params.UseEarlyOut) {
        bool in_bounds = (samplingDepth0 < params.DepthBounds.y && samplingDepth0 > params.DepthBounds.x);
        skip_pixel = !in_bounds;
        bool wave_active = subgroupAny(!skip_pixel);

        if (gl_SubgroupSize == WAVE_SIZE) {
            if (!wave_active) return;
        } else {
            // Fallback for non-matching WaveSize
            LdsEarlyOut = true;
            barrier();
            if (wave_active) LdsEarlyOut = false;
            barrier();
            if (LdsEarlyOut) return;
        }
    }

    vec2 depthPair1 = GetScreenDepthPair(ivec2(readXY1), offsetXY1);
    vec2 depthPair2 = GetScreenDepthPair(ivec2(readXY2), offsetXY2);

    float d20 = depthPair0.y;
    float depthThicknessScale0 = abs(params.FarDepthValue - samplingDepth0);
    bool usePointFilter0 = abs(samplingDepth0 - d20) > depthThicknessScale0 * params.BilinearThreshold && transient_edgeMask_fetch(ivec2(readXY0)).r < 1.0;
    float edgeDepth0 = params.IgnoreEdgePixels ? 1e20 : samplingDepth0;
    float shadowDepth0 = samplingDepth0 + abs(samplingDepth0 - d20) * z_sign;
    float shadowingDepth0 = usePointFilter0 ? edgeDepth0 : shadowDepth0;

    float samplingDepth1 = depthPair1.x;
    float d21 = depthPair1.y;
    float depthThicknessScale1 = abs(params.FarDepthValue - samplingDepth1);
    bool usePointFilter1 = abs(samplingDepth1 - d21) > depthThicknessScale1 * params.BilinearThreshold && transient_edgeMask_fetch(ivec2(readXY1)).r < 1.0;
    float edgeDepth1 = params.IgnoreEdgePixels ? 1e20 : samplingDepth1;
    float shadowDepth1 = samplingDepth1 + abs(samplingDepth1 - d21) * z_sign;
    float shadowingDepth1 = usePointFilter1 ? edgeDepth1 : shadowDepth1;

    float samplingDepth2 = depthPair2.x;
    float d22 = depthPair2.y;
    float depthThicknessScale2 = abs(params.FarDepthValue - samplingDepth2);
    bool usePointFilter2 = abs(samplingDepth2 - d22) > depthThicknessScale2 * params.BilinearThreshold && transient_edgeMask_fetch(ivec2(readXY2)).r < 1.0;
    float edgeDepth2 = params.IgnoreEdgePixels ? 1e20 : samplingDepth2;
    float shadowDepth2 = samplingDepth2 + abs(samplingDepth2 - d22) * z_sign;
    float shadowingDepth2 = usePointFilter2 ? edgeDepth2 : shadowDepth2;

    // Write LDS
    DepthData[laneID] = (shadowingDepth0 - params.LightCoordinate.z) / sampleDistance0;

    float storedDepth1 = (shadowingDepth1 - params.LightCoordinate.z) / sampleDistance1;
    storedDepth1 = sampleDistance1 > 0.0 ? storedDepth1 : 1e10;
    DepthData[uint(WAVE_SIZE) + laneID] = storedDepth1;

    float storedDepth2 = (shadowingDepth2 - params.LightCoordinate.z) / sampleDistance2;
    storedDepth2 = sampleDistance2 > 0.0 ? storedDepth2 : 1e10;
    DepthData[uint(WAVE_SIZE * 2) + laneID] = storedDepth2;

    barrier();

    if (skip_pixel) return; // But wait, other threads might need us? No, we wrote LDS already.

    ivec2 writeTexel = ivec2(write_xy);
    uvec4 packedGBufferData1 = texelFetch(usam_gbufferSolidData1, writeTexel, 0);
    uint materialID = packedGBufferData1.a >> 16;
    float resourceSSS = unpackUnorm4x8(packedGBufferData1.g).b;
    #if defined(MC_TEXTURE_FORMAT_LAB_PBR) && SETTING_PBR_MATERIAL == 1 || SETTING_PBR_MATERIAL == 2
    uint packedGBufferData2 = texelFetch(usam_gbufferSolidData2, writeTexel, 0).r;
    bool forceBuiltInPBR = bool(bitfieldExtract(packedGBufferData2, 26, 1));
    #else
    bool forceBuiltInPBR = true;
    #endif
    float sssFactor = material_decodeSSS(materialID, resourceSSS, forceBuiltInPBR);
    float start_depth = samplingDepth0;
    if (sssFactor > 0.0) {
        start_depth = coords_reversedZToViewZ(start_depth, nearPlane);
        float jitterR = rand_stbnVec1(writeTexel, frameCounter);
        start_depth += jitterR * pow(sssFactor, 0.25) * SETTING_SSS_DEPTH_RANGE;
        start_depth = coords_viewZToReversedZ(start_depth, nearPlane);
    }

    if (params.UsePrecisionOffset) start_depth = mix(start_depth, params.FarDepthValue, -1.0 / 65535.0);

    uint sample_index = laneID + 1u;
    vec4 shadow_value = vec4(1.0);
    float hard_shadow = 1.0;

    float depthScaleFactor = min(sampleDistance0 + direction, 1.0 / params.SurfaceThickness) / max(depthThicknessScale0, 1e-6);
    float depth_scale = depthScaleFactor * sampleDistance0;
    start_depth = (start_depth - params.LightCoordinate.z) * depthScaleFactor - z_sign;

    // Hard samples
    for (int i = 0; i < HARD_SHADOW_SAMPLES; i++) {
        float depth_delta = abs(start_depth - DepthData[sample_index + i] * depth_scale);
        hard_shadow = min(hard_shadow, depth_delta);
    }

    // Soft samples
    for (int i = HARD_SHADOW_SAMPLES; i < SAMPLE_COUNT - FADE_OUT_SAMPLES; i++) {
        float depth_delta = abs(start_depth - DepthData[sample_index + i] * depth_scale);
        shadow_value[i & 3] = min(shadow_value[i & 3], depth_delta);
    }

    // Fade out
    for (int i = SAMPLE_COUNT - FADE_OUT_SAMPLES; i < SAMPLE_COUNT; i++) {
        float depth_delta = abs(start_depth - DepthData[sample_index + i] * depth_scale);
        float fade = float(i + 1 - (SAMPLE_COUNT - FADE_OUT_SAMPLES)) / float(FADE_OUT_SAMPLES + 1) * 0.75;
        shadow_value[i & 3] = min(shadow_value[i & 3], depth_delta + fade);
    }

    // Contrast
    shadow_value = clamp(shadow_value * params.ShadowContrast + (1.0 - params.ShadowContrast), 0.0, 1.0);
    hard_shadow = clamp(hard_shadow * params.ShadowContrast + (1.0 - params.ShadowContrast), 0.0, 1.0);

    float result = dot(shadow_value, vec4(0.25));
    result = min(hard_shadow, result);

    // Store
    transient_bendShadow_store(writeTexel, vec4(result));
}

void main() {
    uint linearGroupID = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;

    uint totalDispatches = indirectComputeData[BEND_SSS_PARAMS_BASE + 4u];
    uint dataPtr = BEND_SSS_PARAMS_BASE + 8u;

    bool found = false;
    uint wcY = 0u;
    uint wcZ = 0u;
    uint woX = 0u;
    uint woY = 0u;
    uint groupStart = 0u;

    for (uint i = 0u; i < totalDispatches; i++) {
        uint start = indirectComputeData[dataPtr + i * 6u + 4u];
        uint count = indirectComputeData[dataPtr + i * 6u + 5u];
        if (linearGroupID >= start && linearGroupID < start + count) {
            wcY = indirectComputeData[dataPtr + i * 6u + 0u];
            wcZ = indirectComputeData[dataPtr + i * 6u + 1u];
            woX = indirectComputeData[dataPtr + i * 6u + 2u];
            woY = indirectComputeData[dataPtr + i * 6u + 3u];
            groupStart = start;
            found = true;
            break;
        }
    }

    if (!found) return;

    uint localIndex = linearGroupID - groupStart;
    ivec3 groupID;
    groupID.x = int(localIndex % 64u);
    uint rem = localIndex / 64u;
    groupID.y = int(rem % wcY);
    groupID.z = int(rem / wcY);

    DispatchParameters params;
    // params.SurfaceThickness = 0.005; // Default
    params.SurfaceThickness = 0.01; // Works better with grass
    params.BilinearThreshold = 0.02;
    params.ShadowContrast = 8.0;
    params.IgnoreEdgePixels = false;
    params.UsePrecisionOffset = false;
    params.BilinearSamplingOffsetMode = false;
    // params.DebugOutputEdgeMask = false;
    params.DepthBounds = vec2(0.0, 1.0);
    params.UseEarlyOut = true;
    params.LightCoordinate = vec4(
        uintBitsToFloat(indirectComputeData[BEND_SSS_PARAMS_BASE + 0u]),
        uintBitsToFloat(indirectComputeData[BEND_SSS_PARAMS_BASE + 1u]),
        uintBitsToFloat(indirectComputeData[BEND_SSS_PARAMS_BASE + 2u]),
        uintBitsToFloat(indirectComputeData[BEND_SSS_PARAMS_BASE + 3u])
    );
    params.WaveOffset = ivec2(woX, woY);
    params.FarDepthValue = 0.0;
    params.NearDepthValue = 1.0; // Reversed Z
    params.InvDepthTextureSize = 1.0 / uval_mainImageSize;

    WriteScreenSpaceShadow(params, groupID, gl_LocalInvocationIndex);
}
