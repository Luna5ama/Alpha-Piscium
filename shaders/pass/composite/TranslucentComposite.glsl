#include "/techniques/SST.glsl"
#include "/util/FullScreenComp.glsl"
#include "/util/GBufferData.glsl"
#include "/util/Material.glsl"
#include "/util/Coords.glsl"
#include "/util/Colors.glsl"
#include "/util/Fresnel.glsl"
#include "/util/BSDF.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform restrict image2D uimg_main;
layout(rgba16f) uniform writeonly image2D uimg_csrgba16f;

void main() {
    if (all(lessThan(texelPos, global_mainImageSizeI))) {
        vec4 outputColor = texelFetch(usam_temp1, texelPos, 0);

        ivec2 farDepthTexelPos = texelPos;
        ivec2 nearDepthTexelPos = texelPos;
        farDepthTexelPos.y += global_mainImageSizeI.y;
        nearDepthTexelPos += global_mainImageSizeI;

        float startViewZ = -texelFetch(usam_translucentDepthLayers, nearDepthTexelPos, 0).r;
        //            float endViewZ = -texelFetch(usam_translucentDepthLayers, farDepthTexelPos, 0).r;
        //            float startViewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;

        if (startViewZ > -65536.0) {
            vec2 screenPos = coords_texelToUV(texelPos, global_mainImageSizeRcp);
            vec2 ndcPos = screenPos * 2.0 - 1.0;
            float edgeFactor = smoothstep(1.0, 0.0, dot(ndcPos, ndcPos));
            vec3 startViewPos = coords_toViewCoord(screenPos, startViewZ, global_camProjInverse);

            GBufferData gData = gbufferData_init();
            gbufferData1_unpack(texelFetch(usam_gbufferData1, texelPos, 0), gData);
            gbufferData2_unpack(texelFetch(usam_gbufferData2, texelPos, 0), gData);

            Material material = material_decode(gData);

            vec3 viewDir = normalize(-startViewPos);

            vec3 localViewDir = normalize(material.tbnInv * viewDir);

            vec2 noiseV = rand_stbnVec2(texelPos, frameCounter);
            float pdfRatio = 1.0;
            vec3 tangentMicroNormal = bsdf_SphericalCapBoundedWithPDFRatio(noiseV, localViewDir, vec2(material.roughness), pdfRatio);
            vec3 microNormal = normalize(material.tbn * tangentMicroNormal);

            float rior = AIR_IOR / material.hardCodedIOR;
            vec3 refractDir = refract(-viewDir, microNormal, rior);
            vec3 reflectDir = reflect(-viewDir, microNormal);

            SSTResult refractResult = sst_trace(startViewPos, refractDir);
            vec2 refractCoord = refractResult.hit ? refractResult.hitScreenPos.xy : coords_viewToScreen(startViewPos + refractDir * edgeFactor, global_camProj).xy;
            vec3 refractColor = texture(usam_temp1, refractCoord).rgb;

            SSTResult reflectResult = sst_trace(startViewPos, reflectDir);
            vec3 reflectColor = vec3(0.0);
            if (reflectResult.hit) {
                reflectColor = texture(usam_temp1, reflectResult.hitScreenPos.xy).rgb;
            }

            float MDotV = dot(microNormal, -viewDir);
            float NDotV = dot(gData.normal, -viewDir);
            float NDotL = saturate(dot(gData.normal, reflectDir));

            float fresnelTransmittance = fresnel_dielectricDielectric_transmittance(MDotV, AIR_IOR, material.hardCodedIOR);
            vec3 translucentTransmittance = texelFetch(usam_translucentColor, texelPos, 0).rgb;
            float fresnelReflectance = fresnel_dielectricDielectric_reflection(MDotV, AIR_IOR, material.hardCodedIOR);
            float g1 = bsdf_smithG1(NDotV, material.roughness);
            float g2 = bsdf_smithG2(NDotV, NDotL, material.roughness);

            vec3 translucentColor = vec3(0.0);
            translucentColor += fresnelTransmittance * translucentTransmittance * refractColor;
            translucentColor += fresnelReflectance * pdfRatio * translucentTransmittance * reflectColor;
            outputColor.rgb = translucentColor;
        }

        imageStore(uimg_main, texelPos, outputColor);

        #ifdef SETTING_DOF
        float viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;
        outputColor.a = abs(viewZ);
        imageStore(uimg_csrgba16f, csrgba16f_temp1_texelToTexel(texelPos), outputColor);
        #endif
    }
}