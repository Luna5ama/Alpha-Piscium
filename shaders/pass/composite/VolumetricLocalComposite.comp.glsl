#extension GL_KHR_shader_subgroup_ballot : enable

#define GLOBAL_DATA_MODIFIER \

#include "/techniques/atmospherics/LocalComposite.glsl"
#include "/techniques/textile/CSRGBA16F.glsl"
#include "/util/FullScreenComp.glsl"
#include "/util/Colors2.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) restrict uniform image2D uimg_main;

void main() {
    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        vec4 outputColor = texelFetch(usam_main, texelPos, 0);

        vec3 albedo = colors2_material_idt(texelFetch(usam_temp5, texelPos, 0).rgb);
        vec4 glintColorData = texelFetch(usam_temp4, texelPos, 0);
        if (any(greaterThan(glintColorData.xyz, vec3(0.0)))) {
            vec3 glintColor = colors2_material_idt(glintColorData.rgb);
            glintColor = pow(glintColor, vec3(SETTING_EMISSIVE_ARMOR_GLINT_CURVE));
            float baseColorLuma = colors2_colorspaces_luma(COLORS2_COLORSPACES_SRGB, albedo.rgb);
            albedo.rgb += glintColor.rgb * glintColorData.a * (1.0 + baseColorLuma * 12.0) * 8.0;
        }

        vec3 giRadiance = texelFetch(usam_csrgba16f, csrgba16f_temp1_texelToTexel(texelPos), 0).rgb;
        outputColor.rgb += giRadiance.rgb * albedo;
        ScatteringResult sctrResult = atmospherics_localComposite(texelPos);
        outputColor.rgb = scatteringResult_apply(sctrResult, outputColor.rgb);

        imageStore(uimg_main, texelPos, outputColor);
    }
}