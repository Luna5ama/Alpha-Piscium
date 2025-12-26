#extension GL_KHR_shader_subgroup_ballot : enable

#define GLOBAL_DATA_MODIFIER \

#define MATERIAL_TRANSLUCENT a

#include "/techniques/atmospherics/air/lut/API.glsl"
#include "/techniques/atmospherics/LocalComposite.glsl"
#include "/techniques/textile/CSR32F.glsl"
#include "/techniques/Lighting.glsl"
#include "/util/FullScreenComp.glsl"
#include "/util/GBufferData.glsl"
#include "/util/Material.glsl"
#include "/util/Coords.glsl"
#include "/util/Fresnel.glsl"
#include "/util/BSDF.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform restrict writeonly image2D uimg_temp4;
layout(rgba16f) uniform restrict image2D uimg_main;

void main() {
    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        imageStore(uimg_temp4, texelPos, vec4(0.0));
        vec4 outputColor = texelFetch(usam_main, texelPos, 0);

        ivec2 waterNearDepthTexelPos = csr32f_tile1_texelToTexel(texelPos);
        ivec2 waterFarDepthTexelPos = csr32f_tile2_texelToTexel(texelPos);

        ivec2 translucentNearDepthTexelPos = csr32f_tile3_texelToTexel(texelPos);
        ivec2 translucentFarDepthTexelPos = csr32f_tile4_texelToTexel(texelPos);

        float waterStartViewZ = -texelFetch(usam_csr32f, waterNearDepthTexelPos, 0).r;
        float translucentStartViewZ = -texelFetch(usam_csr32f, translucentNearDepthTexelPos, 0).r;
        //            float endViewZ = -texelFetch(usam_csr32f, farDepthTexelPos, 0).r;
        //            float startViewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;
        float startViewZ = max(translucentStartViewZ, waterStartViewZ);


        float solidViewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;

        if (startViewZ > -65536.0 && startViewZ > solidViewZ) {
            vec2 screenPos = coords_texelToUV(texelPos, uval_mainImageSizeRcp);
            vec3 startViewPos = coords_toViewCoord(screenPos, startViewZ, global_camProjInverse);

            GBufferData gData = gbufferData_init();
            gbufferData1_unpack(texelFetch(usam_gbufferData1, texelPos, 0), gData);
            gbufferData2_unpack(texelFetch(usam_gbufferData2, texelPos, 0), gData);

            Material material = material_decode(gData);

            vec3 viewDir = normalize(-startViewPos);
            vec3 localViewDir = normalize(material.tbnInv * viewDir);

            vec2 noiseV = rand_stbnVec2(texelPos, frameCounter);
            float pdfRatio = 1.0;
            bsdf_SphericalCapBoundedWithPDFRatio(noiseV, localViewDir, vec2(material.roughness), pdfRatio);

            vec4 sstData1 = transient_translucentRefraction_fetch(texelPos);
            vec4 sstData2 = transient_translucentReflection_fetch(texelPos);
            vec3 refractColor = sstData1.xyz;
            vec3 reflectColor = sstData2.xyz;

            float MDotV = sstData1.w;
            float NDotV = dot(gData.normal, viewDir);
            float NDotL = sstData2.w;

            vec2 iors = mix(vec2(AIR_IOR, material.hardCodedIOR), vec2(material.hardCodedIOR, AIR_IOR), bvec2(isEyeInWater == 1));
            float fresnelTransmittance = fresnel_dielectricDielectric_transmittance(MDotV, iors.x, iors.y);
            float fresnelReflectance = fresnel_dielectricDielectric_reflection(MDotV, iors.x, iors.y);
            float g1 = bsdf_smithG1(NDotV, material.roughness);
            float g2 = bsdf_smithG2(NDotV, NDotL, material.roughness);

            float reflectance = max(fresnelReflectance * pdfRatio * (g2 / g1), 0.0);

            vec3 translucentColor = vec3(0.0);
            translucentColor += fresnelTransmittance * gData.albedo * refractColor;
            translucentColor += reflectance * reflectColor;

            // TODO: Cleanup
            {
                AtmosphereParameters atmosphere = getAtmosphereParameters();
                material.albedo = max(material.albedo, 0.0001);
                vec3 feetPlayerPos = (gbufferModelViewInverse * vec4(startViewPos, 1.0)).xyz;
                vec3 worldPos = feetPlayerPos + cameraPosition;
                vec3 atmPos = atmosphere_viewToAtm(atmosphere, startViewPos);
                atmPos.y = max(atmPos.y, atmosphere.bottom + 0.1);
                float viewAltitude = length(atmPos);
                vec3 upVector = atmPos / viewAltitude;
                const vec3 earthCenter = vec3(0.0, 0.0, 0.0);

                vec4 shadowPos = global_shadowProjPrev * global_shadowRotationMatrix * global_shadowView * vec4(feetPlayerPos, 1.0);

                vec3 sampleTexCoord = shadowPos.xyz / shadowPos.w;
                sampleTexCoord = sampleTexCoord * 0.5 + 0.5;
                sampleTexCoord.xy = rtwsm_warpTexCoord(usam_rtwsm_imap, sampleTexCoord.xy);
                float shadowV = rtwsm_sampleShadowDepth(shadowtex1HW, sampleTexCoord, 0.0);
                vec3 V = normalize(-startViewPos);

                vec3 shadow = vec3(shadowV);
                float shadowIsSun = float(all(equal(sunPosition, shadowLightPosition)));

                float cosSunZenith = dot(uval_sunDirWorld, vec3(0.0, 1.0, 0.0));
                vec3 tSun = atmospherics_air_lut_sampleTransmittance(atmosphere, cosSunZenith, viewAltitude);
                tSun *= float(raySphereIntersectNearest(atmPos, uval_sunDirWorld, earthCenter + PLANET_RADIUS_OFFSET * upVector, atmosphere.bottom) < 0.0);
                vec3 sunShadow = mix(vec3(1.0), shadow, shadowIsSun);
                vec4 sunIrradiance = vec4(SUN_ILLUMINANCE * tSun * sunShadow, colors2_colorspaces_luma(COLORS2_WORKING_COLORSPACE, sunShadow));
                LightingResult sunLighting = directLighting2(material, sunIrradiance, V, uval_sunDirView, gData.normal, material.hardCodedIOR);

                float cosMoonZenith = dot(uval_moonDirWorld, vec3(0.0, 1.0, 0.0));
                vec3 tMoon = atmospherics_air_lut_sampleTransmittance(atmosphere, cosMoonZenith, viewAltitude);
                tMoon *= float(raySphereIntersectNearest(atmPos, uval_moonDirWorld, earthCenter + PLANET_RADIUS_OFFSET * upVector, atmosphere.bottom) < 0.0);
                vec3 moonShadow = mix(shadow, vec3(1.0), shadowIsSun);
                vec4 moonIrradiance = vec4(MOON_ILLUMINANCE * tMoon * moonShadow, colors2_colorspaces_luma(COLORS2_WORKING_COLORSPACE, moonShadow));
                LightingResult moonLighting = directLighting2(material, moonIrradiance, V, uval_moonDirView, gData.normal, material.hardCodedIOR);

                LightingResult combinedLighting = lightingResult_add(sunLighting, moonLighting);

                translucentColor += combinedLighting.specular;
            }

            outputColor.rgb = translucentColor;
        }

        if (isEyeInWater == 1) {
            ScatteringResult sctrResult = atmospherics_localComposite(1, texelPos);
            outputColor.rgb = scatteringResult_apply(sctrResult, outputColor.rgb);
        }
        ScatteringResult sctrResult = atmospherics_localComposite(2, texelPos);
        outputColor.rgb = scatteringResult_apply(sctrResult, outputColor.rgb);

        imageStore(uimg_main, texelPos, outputColor);
    }
}