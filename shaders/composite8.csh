#version 460 compatibility

#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_vote : enable

#include "/denoiser/Update.glsl"
#include "/util/Coords.glsl"
#include "/util/Rand.glsl"
#include "/util/Interpo.glsl"
#include "/util/Material.glsl"
#include "/util/Dither.glsl"

layout(local_size_x = 8, local_size_y = 8) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

uniform sampler2D usam_temp4;
uniform usampler2D usam_tempRGBA32UI;
uniform usampler2D usam_packedZN;
uniform usampler2D usam_gbufferData32UI;
uniform usampler2D usam_tempRG32UI;

layout(rgba16f) uniform writeonly image2D uimg_temp2;
layout(rgba16f) uniform writeonly image2D uimg_temp3;
layout(rgba8) uniform writeonly image2D uimg_temp6;
layout(rgba32ui) uniform writeonly uimage2D uimg_svgfHistory;

shared vec4 shared_moments[12][12][2];
shared vec4 shared_momentsV[8][12][2];

uvec2 groupOriginTexelPos = gl_WorkGroupID.xy << 3u;
ivec2 texelPos = ivec2(groupOriginTexelPos) + ivec2(gl_LocalInvocationID.xy);

void loadSharedData(uint index) {
    if (index < 144) {
        uvec2 sharedXY = uvec2(index % 12, index / 12);
        ivec2 srcXY = ivec2(groupOriginTexelPos) + ivec2(sharedXY) - 2;
        srcXY = clamp(srcXY, ivec2(0), ivec2(global_mainImageSize - 1));

        vec3 fastColor = colors_LogLuvToSRGB(unpackUnorm4x8(texelFetch(usam_tempRGBA32UI, srcXY, 0).y));
        vec3 directColor = texelFetch(usam_temp4, srcXY, 0).xyz;
        vec3 inputColor = fastColor + directColor;

        inputColor = colors_SRGBToYCoCg(inputColor);
        vec3 moment1 = inputColor;
        vec3 moment2 = inputColor * inputColor;

        shared_moments[sharedXY.y][sharedXY.x][0] = vec4(moment1, colors_sRGB_luma(directColor));
        shared_moments[sharedXY.y][sharedXY.x][1] = vec4(moment2, colors_sRGB_luma(fastColor));
    }
}

void updateMoments0(uvec2 originXY, ivec2 offset, inout vec4 moment1, inout vec4 moment2) {
    ivec2 sampleXY = ivec2(originXY) + offset;
    moment1 += shared_moments[sampleXY.y][sampleXY.x][0];
    moment2 += shared_moments[sampleXY.y][sampleXY.x][1];
}

void sampleV(uint index) {
    if (index < 96) {
        uvec2 writeSharedXY = uvec2(index % 12, index / 12);
        uvec2 readSharedXY = writeSharedXY;
        readSharedXY.y += 2;
        vec4 moment1 = vec4(0.0);
        vec4 moment2 = vec4(0.0);
        updateMoments0(readSharedXY, ivec2(0, -2), moment1, moment2);
        updateMoments0(readSharedXY, ivec2(0, -1), moment1, moment2);
        updateMoments0(readSharedXY, ivec2(0, 0), moment1, moment2);
        updateMoments0(readSharedXY, ivec2(0, 1), moment1, moment2);
        updateMoments0(readSharedXY, ivec2(0, 2), moment1, moment2);
        moment1 /= 5.0;
        moment2 /= 5.0;
        shared_momentsV[writeSharedXY.y][writeSharedXY.x][0] = moment1;
        shared_momentsV[writeSharedXY.y][writeSharedXY.x][1] = moment2;
    }
}

void updateMoments1(uvec2 originXY, ivec2 offset, inout vec4 moment1, inout vec4 moment2) {
    ivec2 sampleXY = ivec2(originXY) + offset;
    moment1 += shared_momentsV[sampleXY.y][sampleXY.x][0];
    moment2 += shared_momentsV[sampleXY.y][sampleXY.x][1];
}

float computeGeometryWeight(vec3 centerPos, vec3 centerNormal, float sampleViewZ, uint sampleNormal, vec2 sampleScreenPos, float a) {
    vec3 sampleViewPos = coords_toViewCoord(sampleScreenPos, sampleViewZ, gbufferProjectionInverse);
    vec3 sampleViewGeomNormal = coords_octDecode11(unpackSnorm2x16(sampleNormal));

    float normalWeight = pow4(dot(centerNormal, sampleViewGeomNormal));

    vec3 posDiff = centerPos - sampleViewPos;
    float planeDist1 = pow2(dot(posDiff, centerNormal));
    float planeDist2 = pow2(dot(posDiff, sampleViewGeomNormal));
    float maxPlaneDist = max(planeDist1, planeDist2);
    float planeWeight = a / (a + maxPlaneDist);

    return planeWeight;
}

void bilateralBilinearSample(
vec2 gatherTexelPos, vec4 baseWeights,
vec3 centerPos, vec3 centerNormal,
inout vec3 colorSum, inout float weightSum
) {
    vec2 originScreenPos = (gatherTexelPos * 2.0) * global_mainImageSizeRcp;
    vec2 gatherUV = gatherTexelPos * global_mainImageSizeRcp;
    vec2 gatherUV2Y = gatherUV * vec2(1.0, 0.5);

    vec4 bilateralWeights = vec4(1.0);
    uvec4 prevNs = textureGather(usam_packedZN, gatherUV2Y, 0);
    vec4 prevZs = uintBitsToFloat(textureGather(usam_packedZN, gatherUV2Y, 1));
    float a = 0.0001 * max(abs(centerPos.z), 0.1);
    bilateralWeights.x *= computeGeometryWeight(centerPos, centerNormal, prevZs.x, prevNs.x, global_mainImageSizeRcp * vec2(-1.0, 1.0) + originScreenPos, a);
    bilateralWeights.y *= computeGeometryWeight(centerPos, centerNormal, prevZs.y, prevNs.y, global_mainImageSizeRcp * vec2(1.0, 1.0) + originScreenPos, a);
    bilateralWeights.z *= computeGeometryWeight(centerPos, centerNormal, prevZs.z, prevNs.z, global_mainImageSizeRcp * vec2(1.0, -1.0) + originScreenPos, a);
    bilateralWeights.w *= computeGeometryWeight(centerPos, centerNormal, prevZs.w, prevNs.w, global_mainImageSizeRcp * vec2(-1.0, -1.0) + originScreenPos, a);

    vec4 interpoWeights = baseWeights * bilateralWeights;
    weightSum += interpoWeights.x + interpoWeights.y + interpoWeights.z + interpoWeights.w;

    uvec4 packedColorData1 = textureGather(usam_tempRG32UI, gatherUV, 0);
    uvec4 packedCOlorData2 = textureGather(usam_tempRG32UI, gatherUV, 1);
    vec4 colorData11 = unpackHalf4x16(packedColorData1.xy);
    vec4 colorData12 = unpackHalf4x16(packedColorData1.zw);

    vec4 colorData21 = unpackHalf4x16(packedCOlorData2.xy);
    vec4 colorData22 = unpackHalf4x16(packedCOlorData2.zw);

    vec4 colorRs = vec4(colorData11.x, colorData11.z, colorData12.x, colorData12.z);
    colorSum.r += dot(interpoWeights, colorRs);
    vec4 colorGs = vec4(colorData11.y, colorData11.w, colorData12.y, colorData12.w);
    colorSum.g += dot(interpoWeights, colorGs);
    vec4 colorBs = vec4(colorData21.x, colorData21.z, colorData22.x, colorData22.z);
    colorSum.b += dot(interpoWeights, colorBs);
}

const float _BOOST_ADD = exp2(-SETTING_DENOISER_VARIANCE_BOOST_ADD_FACTOR);

void main() {
    loadSharedData(gl_LocalInvocationIndex);
    loadSharedData(gl_LocalInvocationIndex + 64);
    loadSharedData(gl_LocalInvocationIndex + 128);
    barrier();

    sampleV(gl_LocalInvocationIndex);
    sampleV(gl_LocalInvocationIndex + 64);
    barrier();

    if (all(lessThan(texelPos, global_mainImageSizeI))) {
        vec3 currColor = vec3(0.0);

        float viewZ = uintBitsToFloat(texelFetch(usam_packedZN, texelPos + ivec2(0, global_mainImageSizeI.y), 0).g);

        if (viewZ != -65536.0) {
            vec2 texelPos2x2F = vec2(texelPos) * 0.5;

            vec2 screenPos = (vec2(texelPos) + 0.5) * global_mainImageSizeRcp;
            vec3 viewPos = coords_toViewCoord(screenPos, viewZ, gbufferProjectionInverse);

            GBufferData gData;
            gbufferData1_unpack(texelFetch(usam_gbufferData32UI, texelPos, 0), gData);

            vec3 colorSum = vec3(0.0);
            float weightSum = 0.0;

            vec2 centerPixel = texelPos2x2F - 0.5;
            vec2 centerPixelOrigin = floor(centerPixel);
            vec2 gatherTexelPos = centerPixelOrigin + 1.0;
            vec2 pixelPosFract = centerPixel - centerPixelOrigin;

            vec2 bilinearWeights2 = pixelPosFract;
            vec4 blinearWeights4;
            blinearWeights4.yz = bilinearWeights2.xx;
            blinearWeights4.xw = 1.0 - bilinearWeights2.xx;
            blinearWeights4.xy *= bilinearWeights2.yy;
            blinearWeights4.zw *= 1.0 - bilinearWeights2.yy;

            bilateralBilinearSample(
                texelPos2x2F, blinearWeights4,
                viewPos, gData.geometryNormal,
                colorSum, weightSum
            );

            if (weightSum > 0.01) {
                colorSum /= weightSum;
                currColor += colorSum;
            }
        }

        vec3 directColor = texelFetch(usam_temp4, texelPos, 0).xyz;
        vec3 currTotalColor = currColor;
        currTotalColor += directColor;

        vec3 mean;
        vec3 stddev;
        float directLum;
        float giLum;
        {
            uvec2 readSharedXY = gl_LocalInvocationID.xy;
            readSharedXY.x += 2;
            vec4 moment1 = vec4(0.0);
            vec4 moment2 = vec4(0.0);
            updateMoments1(readSharedXY, ivec2(-2, 0), moment1, moment2);
            updateMoments1(readSharedXY, ivec2(-1, 0), moment1, moment2);
            updateMoments1(readSharedXY, ivec2(0, 0), moment1, moment2);
            updateMoments1(readSharedXY, ivec2(1, 0), moment1, moment2);
            updateMoments1(readSharedXY, ivec2(2, 0), moment1, moment2);
            moment1 /= 5.0;
            moment2 /= 5.0;
            vec3 variance = max(moment2.rgb - pow2(moment1.rgb), 0.0);

            mean = moment1.rgb;
            stddev = sqrt(variance);
            directLum = moment1.a;
            giLum = moment2.a;
        }

        float directWeighting = exp2(-directLum / giLum);

        uvec4 packedData = texelFetch(usam_tempRGBA32UI, texelPos, 0);
        vec3 prevColor;
        vec3 prevFastColor;
        vec2 prevMoments;
        float prevHLen;
        svgf_unpack(packedData, prevColor, prevFastColor, prevMoments, prevHLen);

        float clampingThreshold = directWeighting * SETTING_DENOISER_FAST_HISTORY_CLAMPING_THRESHOLD;
        vec3 aabbMin = mean - stddev * clampingThreshold;
        vec3 aabbMax = mean + stddev * clampingThreshold;
        vec3 clipFastColor = colors_SRGBToYCoCg(prevFastColor + directColor);
        aabbMin = min(aabbMin, clipFastColor);
        aabbMax = max(aabbMax, clipFastColor);
        vec3 prevColorYCoCg = colors_SRGBToYCoCg(prevColor);
        vec3 prevColorYCoCgClamped = clamp(prevColorYCoCg, aabbMin, aabbMax);
        float clippingWeight = linearStep(
            SETTING_DENOISER_MAX_FAST_ACCUM,
            SETTING_DENOISER_MAX_FAST_ACCUM * 2.0,
            prevHLen
        );
        prevColorYCoCg = mix(prevColorYCoCg, prevColorYCoCgClamped, clippingWeight);
        vec3 prevColorClamped = colors_YCoCgToSRGB(prevColorYCoCg);

        float moment2Correction = pow2(colors_sRGB_luma(prevColorClamped)) - pow2(colors_sRGB_luma(prevColor));
        prevMoments.y += moment2Correction;
        prevMoments.y = max(prevMoments.y, 0.0);

        prevColor = prevColorClamped;

        vec3 newColor;
        vec3 newFastColor;
        vec2 newMoments;
        float newHLen;
        svgf_updateHistory(
            currTotalColor, currColor,
            prevColor, prevFastColor, prevMoments, prevHLen,
            newColor, newFastColor, newMoments, newHLen
        );

//        vec3 meanSRGB = colors_YCoCgToSRGB(mean);
//        imageStore(uimg_temp3, texelPos, vec4(exp2(-max(directLum / giLum, 0.0))));

        {
            float variance = max(newMoments.g - newMoments.r * newMoments.r, 0.0);
            float boostFrameV = linearStep(1.0 + SETTING_DENOISER_VARIANCE_BOOST_FRAMES, 1.0, newHLen);
            float boostV = _BOOST_ADD + variance * SETTING_DENOISER_VARIANCE_BOOST_MULTIPLY;
            variance += boostV * pow(boostFrameV, SETTING_DENOISER_VARIANCE_BOOST_DECAY);

            vec4 filterInput = vec4(newColor, variance);
            filterInput = dither_fp16(filterInput, rand_IGN(texelPos, frameCounter));
            imageStore(uimg_temp2, texelPos, filterInput);
        }

        {
            uvec4 packedOutData = uvec4(0u);
            svgf_pack(packedOutData, newColor, newFastColor, newMoments, newHLen);
            imageStore(uimg_svgfHistory, svgf_texelPos1(texelPos), packedOutData);
        }
    }
}