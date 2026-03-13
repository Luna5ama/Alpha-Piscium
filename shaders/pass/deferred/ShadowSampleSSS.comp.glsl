#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_vote : enable
#extension GL_KHR_shader_subgroup_clustered : enable
#extension GL_KHR_shader_subgroup_ballot : enable

#include "/techniques/HiZCheck.glsl"
#include "/util/Coords.glsl"
#include "/util/GBufferData.glsl"
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
    float SurfaceThickness;
    float BilinearThreshold;
    float ShadowContrast;
    bool IgnoreEdgePixels;
    bool UsePrecisionOffset;
    bool BilinearSamplingOffsetMode;
    bool DebugOutputEdgeMask;
    vec2 DepthBounds;
    bool UseEarlyOut;
    vec4 LightCoordinate;
    ivec2 WaveOffset;
    float FarDepthValue;
    float NearDepthValue;
    vec2 InvDepthTextureSize;
};

float GetScreenDepth(ivec2 texel) {
    if (any(lessThan(texel, ivec2(0))) || any(greaterThanEqual(texel, uval_mainImageSizeI))) {
        return 0.0; // Far plane (usually 1.0 or 0.0 depending on setup, standard is 1.0 for far?)
        // Wait, Minecraft default projection: Near=0.05, Far=???. Standard GL: -1 to 1 NDC. Depth 0..1.
        // If usam_gbufferViewZ is -65536 (sky), we should return far depth.
    }
    float viewZ = texelFetch(usam_gbufferViewZ, texel, 0).r;
    if (viewZ <= -65536.0) return 1.0; // Far

    // Project viewZ to screen depth [0,1]
    vec4 clip = global_camProj * vec4(0.0, 0.0, viewZ, 1.0);
    return (clip.z / clip.w) * 0.5 + 0.5;
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
    float major_axis_end = abs(major_axis) - float(WAVE_SIZE);

    float ma_light_frac = x_axis_major ? light_xy_fraction.x : light_xy_fraction.y;
    ma_light_frac = major_axis > 0.0 ? -ma_light_frac : ma_light_frac;

    vec2 start_xy = xy_f + light_xy;
    vec2 end_xy = mix(params.LightCoordinate.xy, start_xy, (major_axis_end + ma_light_frac) / (major_axis_start + ma_light_frac));

    vec2 xy_delta = start_xy - end_xy;

    float thread_step = float(laneID ^ (reverse_direction ? 0u : uint(WAVE_SIZE - 1)));

    vec2 pixel_xy = mix(start_xy, end_xy, thread_step / float(WAVE_SIZE));
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

    float sampling_depth[READ_COUNT];
    float shadowing_depth[READ_COUNT];
    float depth_thickness_scale[READ_COUNT];
    float sample_distance[READ_COUNT];

    float direction = -params.LightCoordinate.w;
    float z_sign = params.NearDepthValue > params.FarDepthValue ? -1.0 : 1.0;

    bool is_edge = false;
    bool skip_pixel = false;
    vec2 write_xy = floor(pixel_xy);

    for (int i = 0; i < READ_COUNT; i++) {
        vec2 read_xy = floor(pixel_xy);
        float minor_axis = x_axis_major ? pixel_xy.y : pixel_xy.x;
        float bilinear = fract(minor_axis) - 0.5;

        // Manual bilinear gather
        int bias = bilinear > 0.0 ? 1 : -1;
        ivec2 offset_xy = ivec2(x_axis_major ? 0 : bias, x_axis_major ? bias : 0);

        float d1 = GetScreenDepth(ivec2(read_xy));
        float d2 = GetScreenDepth(ivec2(read_xy) + offset_xy);

        depth_thickness_scale[i] = abs(params.FarDepthValue - d1);

        bool use_point_filter = abs(d1 - d2) > depth_thickness_scale[i] * params.BilinearThreshold;
        if (i == 0) is_edge = use_point_filter;

        sampling_depth[i] = d1;

        float edge_depth = params.IgnoreEdgePixels ? 1e20 : d1;
        float shadow_depth = d1 + abs(d1 - d2) * z_sign;
        shadowing_depth[i] = use_point_filter ? edge_depth : shadow_depth;

        sample_distance[i] = pixel_distance + (float(WAVE_SIZE) * float(i)) * direction;

        pixel_xy += xy_delta * direction;
    }

    // Early out logic
    if (params.UseEarlyOut) {
        bool in_bounds = (sampling_depth[0] < params.DepthBounds.y && sampling_depth[0] > params.DepthBounds.x);
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

    // Write LDS
    for (int i = 0; i < READ_COUNT; i++) {
        float stored_depth = (shadowing_depth[i] - params.LightCoordinate.z) / sample_distance[i];
        if (i != 0) {
            stored_depth = sample_distance[i] > 0.0 ? stored_depth : 1e10;
        }
        uint idx = uint(i * WAVE_SIZE) + laneID;
        DepthData[idx] = stored_depth;
    }

    barrier();

    if (skip_pixel) return; // But wait, other threads might need us? No, we wrote LDS already.

    // Check bounds of write_xy
    if (any(lessThan(write_xy, vec2(0))) || any(greaterThanEqual(write_xy, uval_mainImageSize))) return;

    // Check ViewZ valid
    ivec2 writeTexel = ivec2(write_xy);
    float viewZ = texelFetch(usam_gbufferViewZ, writeTexel, 0).r;
    if (viewZ <= -65536.0) {
        imageStore(uimg_rgba8, writeTexel, vec4(1.0, 0.0, 1.0, 0.0));
        return;
    }

    float start_depth = sampling_depth[0];
    if (params.UsePrecisionOffset) start_depth = mix(start_depth, params.FarDepthValue, -1.0 / 65535.0);

    start_depth = (start_depth - params.LightCoordinate.z) / sample_distance[0];

    uint sample_index = laneID + 1u;
    vec4 shadow_value = vec4(1.0);
    float hard_shadow = 1.0;

    float depth_scale = min(sample_distance[0] + direction, 1.0 / params.SurfaceThickness) * sample_distance[0] / max(depth_thickness_scale[0], 1e-6);

    start_depth = start_depth * depth_scale - z_sign;

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

    // Material logic (Aux) from original code
    GBufferData gData = gbufferData_init();
    gbufferData1_unpack(texelFetch(usam_gbufferData1, writeTexel, 0), gData);
    gbufferData2_unpack(texelFetch(usam_gbufferData2, writeTexel, 0), gData);
    Material material = material_decode(gData);
    float auxEarlyOut = float(material.sss < 0.0001);
    float auxDispatchMask = 1.0;

    // Store
    imageStore(uimg_rgba8, writeTexel, vec4(result, 0.0, auxEarlyOut, auxDispatchMask));
}

void main() {
    uint linearGroupID = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;

    uint totalDispatches = indirectComputeData[BEND_SSS_PARAMS_BASE + 5u];
    uint dataPtr = BEND_SSS_PARAMS_BASE + 8u;

    bool found = false;
    uint wcY, wcZ, woX, woY, groupStart;

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
    params.SurfaceThickness = 0.005; // Default
    params.BilinearThreshold = uintBitsToFloat(indirectComputeData[BEND_SSS_PARAMS_BASE + 4u]);
    params.ShadowContrast = 4.0;
    params.IgnoreEdgePixels = false;
    params.UsePrecisionOffset = false;
    params.BilinearSamplingOffsetMode = false;
    params.DebugOutputEdgeMask = false;
    params.DepthBounds = vec2(0.0, 1.0);
    params.UseEarlyOut = true;
    params.LightCoordinate = vec4(
        uintBitsToFloat(indirectComputeData[BEND_SSS_PARAMS_BASE + 0u]),
        uintBitsToFloat(indirectComputeData[BEND_SSS_PARAMS_BASE + 1u]),
        uintBitsToFloat(indirectComputeData[BEND_SSS_PARAMS_BASE + 2u]),
        uintBitsToFloat(indirectComputeData[BEND_SSS_PARAMS_BASE + 3u])
    );
    params.WaveOffset = ivec2(woX, woY);
    params.FarDepthValue = 1.0;
    params.NearDepthValue = 0.0; // Assuming standard 0-1 depth
    params.InvDepthTextureSize = 1.0 / uval_mainImageSize;

    WriteScreenSpaceShadow(params, groupID, gl_LocalInvocationIndex);
}
