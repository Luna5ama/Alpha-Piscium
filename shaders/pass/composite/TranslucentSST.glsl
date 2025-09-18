#include "/techniques/atmospherics/air/lut/API.glsl"
#include "/techniques/EnvProbe.glsl"
#include "/techniques/SST.glsl"
#include "/util/FullScreenComp.glsl"
#include "/util/GBufferData.glsl"
#include "/util/Material.glsl"
#include "/util/Fresnel.glsl"
#include "/util/BSDF.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform restrict writeonly image2D uimg_temp1;
layout(rgba16f) uniform restrict writeonly image2D uimg_temp2;

float edgeReductionFactor(vec2 screenPos) {
    const float SQUIRCLE_M = 4.0;
    vec2 ndcPos = screenPos * 2.0 - 1.0;
    vec2 squircle = pow(smoothstep(0.5, 0.95, abs(ndcPos)), vec2(SQUIRCLE_M));
    return saturate(1.0 - (squircle.x + squircle.y));
}

void main() {
    sst_init();

    if (all(lessThan(texelPos, global_mainImageSizeI))) {
        ivec2 farDepthTexelPos = texelPos;
        ivec2 nearDepthTexelPos = texelPos;
        farDepthTexelPos.y += global_mainImageSizeI.y;
        nearDepthTexelPos += global_mainImageSizeI;

        float startViewZ = -texelFetch(usam_translucentDepthLayers, nearDepthTexelPos, 0).r;

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
            vec3 tangentMicroNormal = bsdf_SphericalCapBoundedWithPDFRatio(noiseV, localViewDir, vec2(material.roughness), pdfRatio);
            vec3 microNormal = normalize(material.tbn * tangentMicroNormal);

            float rior = AIR_IOR / material.hardCodedIOR;
            vec3 refractDir = refract(-viewDir, microNormal, rior);
            vec3 reflectDir = reflect(-viewDir, microNormal);

            SSTResult refractResult = sst_trace(startViewPos, refractDir);
            vec2 refractCoord = refractResult.hit ? refractResult.hitScreenPos.xy : coords_viewToScreen(startViewPos + refractDir * edgeReductionFactor(screenPos), global_camProj).xy;
            float refractDepth = texture(usam_gbufferViewZ, refractCoord).r;
            if (refractDepth > startViewZ) {
                refractCoord =  coords_viewToScreen(startViewPos + refractDir * 0.1 / (refractDepth - startViewZ), global_camProj).xy;
            }
            vec3 refractColor = texture(usam_main, refractCoord).rgb;
            float MDotV = saturate(dot(microNormal, viewDir));
            imageStore(uimg_temp1, texelPos, vec4(refractColor, MDotV));

            SSTResult reflectResult = sst_trace(startViewPos, reflectDir);
            vec3 reflectDirWorld = coords_dir_viewToWorld(reflectDir);
            vec2 envUV = coords_mercatorForward(reflectDirWorld);
            ivec2 envTexel = ivec2(envUV * (ENV_PROBE_SIZE - 1));
            EnvProbeData envData = envProbe_decode(texelFetch(usam_envProbe, envTexel, 0));
            vec3 reflectColor = envData.radiance.rgb;
            if (envProbe_isSky(envData)) {
                AtmosphereParameters atmosphere = getAtmosphereParameters();
                SkyViewLutParams skyParams = atmospherics_air_lut_setupSkyViewLutParams(atmosphere, reflectDirWorld);
                reflectColor = atmospherics_air_lut_sampleSkyViewLUT(atmosphere, skyParams, 0.0).inScattering;
            }
            if (reflectResult.hit) {
                reflectColor = mix(reflectColor, texture(usam_main, reflectResult.hitScreenPos.xy).rgb, edgeReductionFactor(reflectResult.hitScreenPos.xy));
            }

            float NDotL = saturate(dot(gData.normal, reflectDir));
            imageStore(uimg_temp2, texelPos, vec4(reflectColor, NDotL));
        }
    }
}