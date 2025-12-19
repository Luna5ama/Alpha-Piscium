#extension GL_KHR_shader_subgroup_ballot : enable

#define GLOBAL_DATA_MODIFIER \

#define MATERIAL_TRANSLUCENT a

#include "/techniques/atmospherics/LocalComposite.glsl"
#include "/techniques/textile/CSR32F.glsl"
#include "/util/FullScreenComp.glsl"
#include "/util/GBufferData.glsl"
#include "/util/Material.glsl"
#include "/util/Coords.glsl"
#include "/util/Fresnel.glsl"
#include "/util/BSDF.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform restrict image2D uimg_main;
layout(rgba8) uniform writeonly image2D uimg_rgba8;

void main() {
    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        vec4 outputColor = texelFetch(usam_main, texelPos, 0);

//        vec3 albedo = colors2_material_toWorkSpace(transient_solidAlbedo_fetch(texelPos).rgb);
//        vec4 glintColorData = texelFetch(usam_temp4, texelPos, 0);
//        if (any(greaterThan(glintColorData.xyz, vec3(0.0)))) {
//            vec3 glintColor = colors2_material_toWorkSpace(glintColorData.rgb);
//            glintColor = pow(glintColor, vec3(SETTING_EMISSIVE_ARMOR_GLINT_CURVE));
//            float baseColorLuma = colors2_colorspaces_luma(COLORS2_COLORSPACES_SRGB, albedo.rgb);
//            albedo.rgb += glintColor.rgb * glintColorData.a * (1.0 + baseColorLuma * 12.0) * 8.0;
//        }

//        vec3 giRadiance = transient_atrous1_fetch(texelPos).rgb;
//        outputColor.rgb += giRadiance.rgb * albedo;
        ScatteringResult sctrResult = atmospherics_localComposite(0, texelPos);
        outputColor.rgb = scatteringResult_apply(sctrResult, outputColor.rgb);

        if (isEyeInWater != 1) {
            ScatteringResult sctrResult = atmospherics_localComposite(1, texelPos);
            outputColor.rgb = scatteringResult_apply(sctrResult, outputColor.rgb);
        }

        ivec2 waterNearDepthTexelPos = csr32f_tile1_texelToTexel(texelPos);
        ivec2 waterFarDepthTexelPos = csr32f_tile2_texelToTexel(texelPos);

        ivec2 translucentNearDepthTexelPos = csr32f_tile3_texelToTexel(texelPos);
        ivec2 translucentFarDepthTexelPos = csr32f_tile4_texelToTexel(texelPos);

        float waterStartViewZ = -texelFetch(usam_csr32f, waterNearDepthTexelPos, 0).r;
        float translucentStartViewZ = -texelFetch(usam_csr32f, translucentNearDepthTexelPos, 0).r;

        float startViewZ = max(translucentStartViewZ, waterStartViewZ);
        float solidViewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;

        if (startViewZ > -65536.0 && startViewZ > solidViewZ) {
            GBufferData gData = gbufferData_init();
            gbufferData1_unpack(texelFetch(usam_gbufferData1, texelPos, 0), gData);
            gbufferData2_unpack(texelFetch(usam_gbufferData2, texelPos, 0), gData);
            vec3 translucentTransmittance = texelFetch(usam_translucentColor, texelPos, 0).rgb;
            outputColor.rgb *= mix(translucentTransmittance / gData.albedo, vec3(0.0), lessThan(gData.albedo, vec3(0.001)));
        }

        imageStore(uimg_main, texelPos, outputColor);
    }
}