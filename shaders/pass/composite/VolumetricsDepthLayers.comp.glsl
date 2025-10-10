#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_KHR_shader_subgroup_ballot : enable

#include "/util/FullScreenComp.glsl"
#include "/techniques/textile/CSR32F.glsl"
#include "/techniques/textile/CSRG32F.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rg32f) uniform writeonly image2D uimg_csrg32f;

shared vec4 shared_dilateData[16];

const float EPS = 0.2;

// layer 1: air behind translucent
// inscattering: 3x16f
// transmittance: 3x10u
// layer 2: water
// inscattering: 3x16f
// transmittance: 3x10u
// layer 3: air in front of translucent
// inscattering: 3x16f
// transmittance: 3x10u

void main() {
    vec2 layer1 = vec2(-FLT_MAX);
    vec2 layer2 = vec2(-FLT_MAX);
    vec2 layer3 = vec2(-FLT_MAX);
    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        float solid = texelFetch(colortex10, texelPos, 0).r;
        float nearWaterOg = -texelFetch(usam_csr32f, csr32f_tile1_texelToTexel(texelPos), 0).r;
        float nearTranslucentOg = -texelFetch(usam_csr32f, csr32f_tile3_texelToTexel(texelPos), 0).r;
        float farWaterOg = -texelFetch(usam_csr32f, csr32f_tile2_texelToTexel(texelPos), 0).r;
        float farTranslucentOg = -texelFetch(usam_csr32f, csr32f_tile4_texelToTexel(texelPos), 0).r;
        if (isEyeInWater == 1) {
            nearWaterOg = 0.0;
        }

        float farTranslucent = abs(farTranslucentOg - farWaterOg) > EPS * abs(min(farTranslucentOg, farWaterOg)) ? farTranslucentOg : 0.0f;
        float farWater = abs(farWaterOg - nearWaterOg) > EPS * abs(min(nearWaterOg, farWaterOg)) ? farWaterOg : 0.0f;

        layer1.y = solid;
        layer1.x = min(farWater, farTranslucent);
        layer1.x = layer1.x < 0.0 ? layer1.x : layer1.y;

        layer2.y = layer1.x;
        layer2.x = nearWaterOg > -65536.0 ? nearWaterOg : layer2.y;

        layer3.y = max(layer2.x, nearTranslucentOg);
        layer3.x = 0.0;

        layer1 = layer1.x > layer1.y + EPS ? layer1 : vec2(-FLT_MAX);
        layer2 = layer2.x > layer2.y + EPS ? layer2 : vec2(-FLT_MAX);
    }

    // Storing results:
    // x: layer1End (solid)
    // y: layer1Start
    //
    // layer2End = layer1Start
    // z: layer2Start
    //
    // w: layer3End
    // layer3Start = 0.0
    vec4 result = vec4(layer1, layer2);
    vec4 dilated = subgroupMax(result);
    if (subgroupElect()) {
        shared_dilateData[gl_SubgroupID] = dilated;
    }
    barrier();
    if (gl_SubgroupID == 0) {
        vec4 partialMin = gl_SubgroupInvocationID < gl_NumSubgroups ? shared_dilateData[gl_SubgroupInvocationID] : vec4(-FLT_MAX);
        vec4 totalMin = subgroupMax(partialMin);
        shared_dilateData[0] = totalMin;
    }
    barrier();

    bvec4 dilateCond = equal(result, vec4(-FLT_MAX));
    result = mix(result, shared_dilateData[0], dilateCond);

    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        result = abs(result) * (vec2(bvec2(any(dilateCond.xy), any(dilateCond.zw))) * 2.0 - 1.0).xxyy;
        imageStore(uimg_csrg32f, csrg32f_tile1_texelToTexel(texelPos), result.xyxy);
        imageStore(uimg_csrg32f, csrg32f_tile2_texelToTexel(texelPos), result.zwzw);
        imageStore(uimg_csrg32f, csrg32f_tile3_texelToTexel(texelPos), layer3.xyxy);
    }
}