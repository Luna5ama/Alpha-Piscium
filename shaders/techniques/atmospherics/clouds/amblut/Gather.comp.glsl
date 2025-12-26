#extension GL_KHR_shader_subgroup_arithmetic : enable

#include "Common.glsl"
#include "/util/Coords.glsl"
#include "/util/Rand.glsl"

/*const*/
#define ATMOSPHERE_RAYMARCHING_SKY a
/*const*/
#include "/techniques/atmospherics/air/Raymarching.glsl"

layout(local_size_x = 32, local_size_y = 32) in;
const ivec3 workGroups = ivec3(AMBIENT_IRRADIANCE_LUT_SIZE, AMBIENT_IRRADIANCE_LUT_SIZE, 1);

shared vec3 shared_inSctrSum[16];

layout(rgba16f) uniform restrict image3D uimg_cloudsAmbLUT;

void main() {
    int layerIndex = clouds_amblut_currLayerIndex();
    uint clearFlag = uint(global_historyResetFactor < 1.0);
    clearFlag &= uint(gl_WorkGroupID.x != layerIndex);
    clearFlag &= uint(gl_WorkGroupID.x < AMBIENT_IRRADIANCE_LUT_LAYERS);
    clearFlag &= uint(gl_WorkGroupID.y == 0u);
    if (bool(clearFlag)) {
        ivec3 texelPos3D = ivec3(uvec3(gl_LocalInvocationID.xy, gl_WorkGroupID.x));
        vec4 result = imageLoad(uimg_cloudsAmbLUT, texelPos3D);
        result.a *= global_historyResetFactor;
        imageStore(uimg_cloudsAmbLUT, texelPos3D, result);
    }

    vec2 jitter = rand_r2Seq2(gl_LocalInvocationIndex + gl_WorkGroupSize.x * frameCounter);
    ivec2 texelPos = ivec2(gl_WorkGroupID.xy);
    vec2 viewDirUV = (vec2(texelPos) + jitter) / vec2(AMBIENT_IRRADIANCE_LUT_SIZE);
    vec3 viewDir = coords_equirectanglarBackward(viewDirUV);

    vec2 rayDirSpherical = ssbo_ambLUTWorkingBuffer.rayDir[gl_LocalInvocationIndex];
    float phi = rayDirSpherical.x;
    float theta = rayDirSpherical.y;
    float cosPhi = cos(phi);
    float sinPhi = sin(phi);
    float cosTheta = cos(theta);
    float sinTheta = sin(theta);
    vec3 rayDir = vec3(cosTheta * sinPhi, cosPhi, sinTheta * sinPhi);
    vec3 inSctr = ssbo_ambLUTWorkingBuffer.inSctr[gl_LocalInvocationIndex];

    float cosLightTheta = dot(viewDir, rayDir);
    float phase =  phasefunc_CornetteShanks(cosLightTheta, -SETTING_CLOUDS_AMB_BACKSCATTER_FACTOR);

    vec3 phasedInSctr = inSctr * phase * SPHERE_SOLID_ANGLE / float(SAMPLE_COUNT);
    vec3 subgroupSum1 = subgroupAdd(phasedInSctr);
    if (subgroupElect()) {
        shared_inSctrSum[gl_SubgroupID] = subgroupSum1;
    }
    barrier();

    if (gl_SubgroupID == 0 && gl_SubgroupInvocationID < gl_NumSubgroups) {
        vec3 subgroupSum2 = shared_inSctrSum[gl_SubgroupInvocationID];
        subgroupSum2 = subgroupAdd(subgroupSum2);
        if (subgroupElect()) {
            vec3 currResult = subgroupSum2;
            ivec3 texelPos3D = ivec3(texelPos, layerIndex);
            vec4 prevResult = imageLoad(uimg_cloudsAmbLUT, texelPos3D);
            vec4 newResult;
            prevResult.a *= global_historyResetFactor;
            newResult.a = min(prevResult.a + 1.0, 16.0);
            newResult.rgb = mix(prevResult.rgb, currResult, 1.0 / newResult.a);

            imageStore(uimg_cloudsAmbLUT, texelPos3D, newResult);
        }
    }
}
