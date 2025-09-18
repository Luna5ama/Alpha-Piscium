#include "/techniques/textile/CSRGBA16F.glsl"
#include "/util/FullScreenComp.glsl"
#include "/util/GBufferData.glsl"
#include "/util/Material.glsl"
#include "/util/Coords.glsl"
#include "/util/Fresnel.glsl"
#include "/util/BSDF.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform restrict image2D uimg_main;
layout(rgba16f) uniform writeonly image2D uimg_csrgba16f;

void main() {
    if (all(lessThan(texelPos, global_mainImageSizeI))) {
        vec4 outputColor = texelFetch(usam_main, texelPos, 0);

        ivec2 farDepthTexelPos = texelPos;
        ivec2 nearDepthTexelPos = texelPos;
        farDepthTexelPos.y += global_mainImageSizeI.y;
        nearDepthTexelPos += global_mainImageSizeI;

        float startViewZ = -texelFetch(usam_translucentDepthLayers, nearDepthTexelPos, 0).r;
        //            float endViewZ = -texelFetch(usam_translucentDepthLayers, farDepthTexelPos, 0).r;
        //            float startViewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;

        if (startViewZ > -65536.0) {
            vec2 screenPos = coords_texelToUV(texelPos, global_mainImageSizeRcp);
            vec3 startViewPos = coords_toViewCoord(screenPos, startViewZ, global_camProjInverse);

            GBufferData gData = gbufferData_init();
            gbufferData1_unpack(texelFetch(usam_gbufferData1, texelPos, 0), gData);
            gbufferData2_unpack(texelFetch(usam_gbufferData2, texelPos, 0), gData);

            Material material = material_decode(gData);
            material.roughness *= 0.5;

            vec3 viewDir = normalize(-startViewPos);
            vec3 localViewDir = normalize(material.tbnInv * viewDir);

            vec2 noiseV = rand_stbnVec2(texelPos, frameCounter);
            float pdfRatio = 1.0;
            bsdf_SphericalCapBoundedWithPDFRatio(noiseV, localViewDir, vec2(material.roughness), pdfRatio);

            vec4 sstData1 = texelFetch(usam_temp1, texelPos, 0);
            vec4 sstData2 = texelFetch(usam_temp2, texelPos, 0);
            vec3 refractColor = sstData1.xyz;
            vec3 reflectColor = sstData2.xyz;

            float MDotV = sstData1.w;
            float NDotV = dot(gData.normal, viewDir);
            float NDotL = sstData2.w;

            float fresnelTransmittance = fresnel_dielectricDielectric_transmittance(MDotV, AIR_IOR, material.hardCodedIOR);
            float fresnelReflectance = fresnel_dielectricDielectric_reflection(MDotV, AIR_IOR, material.hardCodedIOR);
            float g1 = bsdf_smithG1(NDotV, material.roughness);
            float g2 = bsdf_smithG2(NDotV, NDotL, material.roughness);

            float reflectance = max(fresnelReflectance * pdfRatio * (g2 / g1), 0.0);
            vec3 reflectanceAlbedo = reflectance * material.albedo;

            vec3 translucentColor = vec3(0.0);
            translucentColor += fresnelTransmittance * gData.albedo * refractColor;
            translucentColor += reflectance * reflectanceAlbedo * reflectColor;
            outputColor.rgb = translucentColor;
        }

        imageStore(uimg_main, texelPos, outputColor);

        #ifdef SETTING_DOF
        float viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;
        viewZ = max(startViewZ, viewZ);
        outputColor.a = abs(viewZ);
        imageStore(uimg_csrgba16f, csrgba16f_temp1_texelToTexel(texelPos), outputColor);
        #endif
    }
}