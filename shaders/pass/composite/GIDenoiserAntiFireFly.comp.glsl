/*
    References:
        [KOZ21] Kozlowski, Pawel. Cheblokov, Tim. "ReLAX: A Denoiser Tailored to Work with the ReSTIR Algorithm". GTC 2021.
            https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s32759/
*/

#extension GL_KHR_shader_subgroup_ballot : enable

#include "/techniques/gi/Common.glsl"
#include "/techniques/HiZCheck.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform restrict image2D uimg_rgba16f;

// Shared memory with padding for 3x3 tap
// Each work group is 16x16, need +1 padding on each side for 3x3 taps
shared vec4 shared_diff[18][18];
shared vec4 shared_spec[18][18];

uvec2 groupOriginTexelPos = gl_WorkGroupID.xy << 4u;

void loadSharedDataRCRS(uint index) {
    if (index < 324u) { // 18 * 18 = 324
        uvec2 sharedXY = uvec2(index % 18u, index / 18u);
        ivec2 srcXY = ivec2(groupOriginTexelPos) + ivec2(sharedXY) - 1;
        srcXY = clamp(srcXY, ivec2(0), ivec2(uval_mainImageSize - 1));

        vec4 d = transient_gi1AntiFireFlyInput_fetch(srcXY);
        vec4 s = transient_gi3AntiFireFlyInput_fetch(srcXY);

        shared_diff[sharedXY.y][sharedXY.x] = d;
        shared_spec[sharedXY.y][sharedXY.x] = s;
    }
}

void updateMinMax(ivec2 samplePos, vec3 sampleColor, inout vec2 minMaxLum, inout ivec4 minMaxPos) {
    float sampleLum = colors2_colorspaces_luma(COLORS2_WORKING_COLORSPACE, sampleColor);
    if (sampleLum < minMaxLum.x) {
        minMaxLum.x = sampleLum;
        minMaxPos.xy = samplePos;
    }
    if (sampleLum > minMaxLum.y) {
        minMaxLum.y = sampleLum;
        minMaxPos.zw = samplePos;
    }
}

void updateMinMaxDiffSpec(ivec2 samplePos, inout vec2 minMaxLumDiff, inout ivec4 minMaxPosDiff, inout vec2 minMaxLumSpec, inout ivec4 minMaxPosSpec) {
    vec3 sampleDiffColor = shared_diff[samplePos.y][samplePos.x].rgb;
    vec3 sampleSpecColor = shared_spec[samplePos.y][samplePos.x].rgb;
    updateMinMax(samplePos, sampleDiffColor, minMaxLumDiff, minMaxPosDiff);
    updateMinMax(samplePos, sampleSpecColor, minMaxLumSpec, minMaxPosSpec);
}

void antiFireFlyRCRS(ivec2 texelPos) {
    // Load shared tiles (two calls to cover 324 entries with 256 threads)
    loadSharedDataRCRS(gl_LocalInvocationIndex);
    loadSharedDataRCRS(gl_LocalInvocationIndex + 256u);
    barrier();

    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        ivec2 localPos = ivec2(gl_LocalInvocationID.xy) + 1; // +1 for padding

        vec2 minMaxLumDiff = vec2(FLT_MAX, 0.0);
        ivec4 minMaxPosDiff = ivec4(localPos, localPos);
        vec2 minMaxLumSpec = vec2(FLT_MAX, 0.0);
        ivec4 minMaxPosSpec = ivec4(localPos, localPos);

        updateMinMaxDiffSpec(localPos + ivec2(-1, -1), minMaxLumDiff, minMaxPosDiff, minMaxLumSpec, minMaxPosSpec);
        updateMinMaxDiffSpec(localPos + ivec2(0, -1), minMaxLumDiff, minMaxPosDiff, minMaxLumSpec, minMaxPosSpec);
        updateMinMaxDiffSpec(localPos + ivec2(1, -1), minMaxLumDiff, minMaxPosDiff, minMaxLumSpec, minMaxPosSpec);

        updateMinMaxDiffSpec(localPos + ivec2(-1, 0), minMaxLumDiff, minMaxPosDiff, minMaxLumSpec, minMaxPosSpec);
        updateMinMaxDiffSpec(localPos + ivec2(1, 0), minMaxLumDiff, minMaxPosDiff, minMaxLumSpec, minMaxPosSpec);

        updateMinMaxDiffSpec(localPos + ivec2(-1, 1), minMaxLumDiff, minMaxPosDiff, minMaxLumSpec, minMaxPosSpec);
        updateMinMaxDiffSpec(localPos + ivec2(0, 1), minMaxLumDiff, minMaxPosDiff, minMaxLumSpec, minMaxPosSpec);
        updateMinMaxDiffSpec(localPos + ivec2(1, 1), minMaxLumDiff, minMaxPosDiff, minMaxLumSpec, minMaxPosSpec);

        vec4 centerDiff = shared_diff[localPos.y][localPos.x];
        vec4 centerSpec = shared_spec[localPos.y][localPos.x];

        // Fixes vignette-like fade in
        float viewHistoryLength = history_taa_fetch(texelPos).a;
        float antiFireFlyMix = linearStep(1.0, 4.0, viewHistoryLength);

        vec4 newDiff = centerDiff;
        float centerDiffLum = colors2_colorspaces_luma(COLORS2_WORKING_COLORSPACE, newDiff.rgb);
        if (centerDiffLum < minMaxLumDiff.x) {
            newDiff.rgb = shared_diff[minMaxPosDiff.y][minMaxPosDiff.x].rgb;
        }
        if (centerDiffLum > minMaxLumDiff.y) {
            newDiff.rgb = shared_diff[minMaxPosDiff.w][minMaxPosDiff.z].rgb;
        }

        newDiff.rgb = mix(centerDiff.rgb, newDiff.rgb, antiFireFlyMix);
        transient_gi1Reprojected_store(texelPos, newDiff);

        vec4 newSpec = centerSpec;
        float centerSpecLum = colors2_colorspaces_luma(COLORS2_WORKING_COLORSPACE, newSpec.rgb);
        if (centerSpecLum < minMaxLumSpec.x) {
            newSpec.rgb = shared_spec[minMaxPosSpec.y][minMaxPosSpec.x].rgb;
        }
        if (centerSpecLum > minMaxLumSpec.y) {
            newSpec.rgb = shared_spec[minMaxPosSpec.w][minMaxPosSpec.z].rgb;
        }

        newSpec.rgb = mix(centerSpec.rgb, newSpec.rgb, antiFireFlyMix);
        transient_gi3Reprojected_store(texelPos, newSpec);
    }
}

void main() {
    if (hiz_groupGroundCheckSubgroup(gl_WorkGroupID.xy, 4)) {
        ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
        antiFireFlyRCRS(texelPos);
    }
}

