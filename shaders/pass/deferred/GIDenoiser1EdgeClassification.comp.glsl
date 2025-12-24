#include "/techniques/gi/Common.glsl"
#include "/util/Coords.glsl"
#include "/util/Math.glsl"
#include "/util/GBufferData.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform writeonly restrict image2D uimg_temp1;
layout(rgba8) uniform writeonly restrict image2D uimg_rgba8;

// Shared memory with padding for 3x3 tap
// Each work group is 16x16, need +2 padding on each side for 3x3 taps
shared uvec3 shared_data[18][18];

const float BASE_GEOM_DEPTH_WEIGHT = exp2(-16.0);
const float BASE_NORMAL_WEIGHT_DECAY = 4.0;
const float BASE_NORMAL_WEIGHT = 8.0;

uvec2 groupOriginTexelPos = gl_WorkGroupID.xy << 4u;

void loadSharedData(uint index) {
    if (index < 324) { // 18 * 18 = 324
        uvec2 sharedXY = uvec2(index % 18, index / 18);
        ivec2 srcXY = ivec2(groupOriginTexelPos) + ivec2(sharedXY) - 1;
        srcXY = clamp(srcXY, ivec2(0), ivec2(uval_mainImageSize - 1));

        // Load viewZ
        float viewZ = texelFetch(usam_gbufferViewZ, srcXY, 0).r;

        // Load geometry normal
        GBufferData gData = gbufferData_init();
        gbufferData1_unpack_world(texelFetch(usam_gbufferData1, srcXY, 0), gData);

        uvec3 packedData = uvec3(0u);
        packedData.x = floatBitsToUint(viewZ);
        packedData.y = packSnorm4x8(vec4(gData.geomNormal, 0.0));
        packedData.z = packSnorm4x8(vec4(gData.normal, 0.0));

        shared_data[sharedXY.y][sharedXY.x] = packedData;
    }
}

struct SampleData {
    vec3 geomNormal;
    vec3 normal;
    float viewZ;
};

SampleData loadSampleData(ivec2 texelPos) {
    SampleData sData;
    uvec3 packedData = shared_data[texelPos.y][texelPos.x];
    sData.viewZ = uintBitsToFloat(packedData.x);
    sData.geomNormal = normalize(unpackSnorm4x8(packedData.y).xyz);
    sData.normal = normalize(unpackSnorm4x8(packedData.z).xyz);
    return sData;
}

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);

    // Load shared data using flattened index (18*18 = 324 elements, 256 threads)
    loadSharedData(gl_LocalInvocationIndex);
    loadSharedData(gl_LocalInvocationIndex + 256);
    barrier();

    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        // Local position in shared memory (with +1 offset for padding)
        ivec2 localPos = ivec2(gl_LocalInvocationID.xy) + 1;

        SampleData centerData = loadSampleData(localPos);

        vec2 centerScreenPos = coords_texelToUV(texelPos, uval_mainImageSizeRcp);
        vec3 centerViewPos = coords_toViewCoord(centerScreenPos, centerData.viewZ, global_camProjInverse);
        vec3 centerWorldPos = coords_pos_viewToWorld(centerViewPos, gbufferModelViewInverse);

        float planeWeight = rcp(exp2(SETTING_DENOISER_REPROJ_GEOMETRY_EDGE_WEIGHT)) * max(abs(centerData.viewZ), 0.1);

        float glazingAngleFactor = saturate(dot(centerData.geomNormal, -normalize(centerWorldPos)));
        float geomDepthThreshold = exp2(mix(-10.0, -16.0, glazingAngleFactor)) * max(4.0, pow2(centerData.viewZ));

        float weightSum = 0.0;

        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0) continue;

                ivec2 sampleLocalPos = localPos + ivec2(dx, dy);

                // Load sample data
                SampleData sampleData = loadSampleData(sampleLocalPos);

                // Convert to view space positions
                ivec2 sampleGlobalPos = texelPos + ivec2(dx, dy);
                vec2 sampleScreenPos = coords_texelToUV(sampleGlobalPos, uval_mainImageSizeRcp);
                vec3 sampleViewPos = coords_toViewCoord(sampleScreenPos, sampleData.viewZ, global_camProjInverse);
                vec3 sampleWorldPos = coords_pos_viewToWorld(sampleViewPos, gbufferModelViewInverse);

                // Calculate plane distance (geometry weight)
                float planeDistance = gi_planeDistance(centerWorldPos, centerData.geomNormal, sampleWorldPos, sampleData.geomNormal);

                float geomDepthWeight = float(planeDistance < geomDepthThreshold);

                float geomNormalDot = saturate(dot(centerData.geomNormal, sampleData.geomNormal));
                float geomNormalWeight = pow2(geomNormalDot);

                float weight = geomDepthWeight * geomNormalWeight;
                weightSum += weight;
            }
        }

        weightSum /= 8.0;

        transient_edgeMaskTemp_store(texelPos, vec4(weightSum));
    }
}

