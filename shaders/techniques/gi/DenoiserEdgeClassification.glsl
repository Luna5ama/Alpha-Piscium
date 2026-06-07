#include "/techniques/gi/Common.glsl"
#include "/util/Coords.glsl"
#include "/util/Math.glsl"

layout(rgba8) uniform writeonly restrict image2D uimg_rgba8;

shared uvec2 shared_data[18][18];

uvec2 groupOriginTexelPos = gl_WorkGroupID.xy << 4u;

void loadSharedData(uint index) {
    if (index < 324) {
        uvec2 sharedXY = uvec2(index % 18, index / 18);
        ivec2 srcXY = ivec2(groupOriginTexelPos) + ivec2(sharedXY) - 1;
        srcXY = clamp(srcXY, ivec2(0), ivec2(uval_mainImageSize - 1));

        float viewZ = texelFetch(usam_gbufferSolidViewZ, srcXY, 0).r;

        uvec4 gbufferData1 = texelFetch(usam_gbufferSolidData1, srcXY, 0);
        vec3 geomNormal = coords_octDecode11(unpackSnorm4x8(gbufferData1.r).xy);

        uvec2 packedData = uvec2(0u);
        packedData.x = floatBitsToUint(viewZ);
        packedData.y = packSnorm4x8(vec4(geomNormal, 0.0));

        shared_data[sharedXY.y][sharedXY.x] = packedData;
    }
}

struct SampleData {
    vec3 geomNormal;
    float viewZ;
};

SampleData loadSampleData(ivec2 texelPos) {
    SampleData sData;
    uvec2 packedData = shared_data[texelPos.y][texelPos.x];
    sData.viewZ = uintBitsToFloat(packedData.x);
    sData.geomNormal = normalize(unpackSnorm4x8(packedData.y).xyz);
    return sData;
}

void classifyGIDenoiserEdges() {
    loadSharedData(gl_LocalInvocationIndex);
    loadSharedData(gl_LocalInvocationIndex + 256);
    barrier();

    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        ivec2 localPos = ivec2(gl_LocalInvocationID.xy) + 1;

        SampleData centerData = loadSampleData(localPos);

        vec2 centerScreenPos = coords_texelToUV(texelPos, uval_mainImageSizeRcp);
        vec3 centerViewPos = coords_toViewCoord(centerScreenPos, centerData.viewZ, global_camProjInverse);
        vec3 centerWorldPos = coords_pos_viewToWorld(centerViewPos, gbufferModelViewInverse);

        float glazingAngleFactor = saturate(dot(centerData.geomNormal, -normalize(centerWorldPos)));
        float geomDepthThreshold = exp2(mix(-10.0, -16.0, glazingAngleFactor)) * max(4.0, pow2(centerData.viewZ));

        float weightSum = 0.0;

        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx != 0 || dy != 0) {
                    ivec2 sampleLocalPos = localPos + ivec2(dx, dy);

                    SampleData sampleData = loadSampleData(sampleLocalPos);

                    ivec2 sampleGlobalPos = texelPos + ivec2(dx, dy);
                    vec2 sampleScreenPos = coords_texelToUV(sampleGlobalPos, uval_mainImageSizeRcp);
                    vec3 sampleViewPos = coords_toViewCoord(sampleScreenPos, sampleData.viewZ, global_camProjInverse);
                    vec3 sampleWorldPos = coords_pos_viewToWorld(sampleViewPos, gbufferModelViewInverse);

                    float planeDistance = gi_planeDistance(centerWorldPos, centerData.geomNormal, sampleWorldPos, sampleData.geomNormal);

                    float geomDepthWeight = float(planeDistance < geomDepthThreshold);

                    float geomNormalDot = saturate(dot(centerData.geomNormal, sampleData.geomNormal));
                    float geomNormalWeight = pow2(geomNormalDot);

                    float weight = geomDepthWeight * geomNormalWeight;
                    weightSum += weight;
                }
            }
        }

        weightSum /= 8.0;

        transient_edgeMask_store(texelPos, vec4(weightSum));
    }
}
