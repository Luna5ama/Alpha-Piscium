#define MATERIAL_TRANSLUCENT a
#define SST_DEBUG_PASS a

#include "/techniques/atmospherics/air/lut/API.glsl"
#include "/techniques/EnvProbe.glsl"
#include "/techniques/textile/CSR32F.glsl"
#include "/techniques/SST2.glsl"
#include "/util/Celestial.glsl"
#include "/util/FullScreenComp.glsl"
#include "/util/GBufferData.glsl"
#include "/util/Material.glsl"
#include "/util/MaterialIDConst.glsl"
#include "/util/Fresnel.glsl"
#include "/util/BSDF.glsl"
#include "/util/Sampling.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform restrict writeonly image2D uimg_rgba16f;

float sst_edgeReductionFactor(vec2 screenPos, float squirclePow, vec2 edgeStart, vec2 edgeEnd) {
    vec2 ndcPos = abs(screenPos * 2.0 - 1.0);
    vec2 squircle = pow(smoothstep(edgeStart, edgeEnd, ndcPos), vec2(squirclePow));
    return saturate(1.0 - (squircle.x + squircle.y));
}

void main() {
    sst_init(0.005);

    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        ivec2 waterNearDepthTexelPos = csr32f_tile1_texelToTexel(texelPos);
        ivec2 waterFarDepthTexelPos = csr32f_tile2_texelToTexel(texelPos);

        ivec2 translucentNearDepthTexelPos = csr32f_tile3_texelToTexel(texelPos);
        ivec2 translucentFarDepthTexelPos = csr32f_tile4_texelToTexel(texelPos);

        float waterStartViewZ = -texelFetch(usam_csr32f, waterNearDepthTexelPos, 0).r;
        float translucentStartViewZ = -texelFetch(usam_csr32f, translucentNearDepthTexelPos, 0).r;

        float startViewZ = max(translucentStartViewZ, waterStartViewZ);

        if (startViewZ > -65536.0) {
            vec2 screenPos = coords_texelToUV(texelPos, uval_mainImageSizeRcp);
            vec3 startViewPos = coords_toViewCoord(screenPos, startViewZ, global_camProjInverse);

            GBufferData gData = gbufferData_init();
            gbufferData1_unpack(texelFetch(usam_gbufferData1, texelPos, 0), gData);
            gbufferData2_unpack(texelFetch(usam_gbufferData2, texelPos, 0), gData);

            Material material = material_decode(gData);
            bool isWater = gData.materialID == MATERIAL_ID_WATER;

            vec3 viewDir = normalize(-startViewPos);

            vec3 localViewDir = normalize(material.tbnInv * viewDir);

            vec2 noiseV = rand_stbnVec2(texelPos, frameCounter);
            float pdfRatio = 1.0;
            vec3 tangentMicroNormal = bsdf_SphericalCapBoundedWithPDFRatio(noiseV, localViewDir, vec2(material.roughness), pdfRatio);
            vec3 microNormal = normalize(material.tbn * tangentMicroNormal);

            vec2 ndcPos = abs(screenPos * 2.0 - 1.0);

            float inWaterDisable = float(isEyeInWater == 1);
            float riorFixWeight = min(max2(ndcPos), 1.0 - pow2(saturate(dot(vec3(0.0, 0.0, 1.0), gData.geomNormal))));
            riorFixWeight = saturate(riorFixWeight - inWaterDisable) * float(isWater);
            float rior = AIR_IOR / mix(material.hardCodedIOR, 1.0, riorFixWeight);
            float cosThetaSign = 1.0;
            if (isEyeInWater == 1) {
                rior = rcp(rior);
                cosThetaSign = -1.0;
            }
            vec3 refIncidentDir = -viewDir;
            vec3 refractDir;

            if (isEyeInWater == 1 || !isWater) {
                refractDir = refract(refIncidentDir, microNormal, rior);
            } else {
                #ifdef SETTING_WATER_REFRACT_APPROX
                refractDir = refract(refIncidentDir, (gData.geomNormal - gData.normal), rior);
                #else
                refractDir = refract(refIncidentDir, microNormal, rior);
                float refractFixWeight = pow4(smoothstep(-0.5, 0.5, dot(refractDir, gData.geomNormal)));
                refractDir = mix(refractDir, refract(refIncidentDir, gData.geomNormal, rior), refractFixWeight);
                #endif
            }
            refractDir = normalize(refractDir);

            vec3 reflectDir = reflect(refIncidentDir, microNormal);
            if (dot(reflectDir, gData.geomNormal) < 0.0) {
                reflectDir = reflect(reflectDir, gData.geomNormal);
            }
            if (isWater) {
                reflectDir = mix(reflectDir, reflect(refIncidentDir, gData.geomNormal), pow2(linearStep(0.5, -0.3, dot(reflectDir, gData.geomNormal))));
            }
            reflectDir = normalize(reflectDir);

            const float SQRT_2 = 1.41421356237;
            const float SQRT_1_2 = 0.7071067812;
            vec2 nv = SQRT_2 * noiseV - SQRT_1_2;
            nv = pow3(nv) + 0.5;
            AtmosphereParameters atmosphere = getAtmosphereParameters();

            SSTRay refractRay = sstray_setup(texelPos, startViewPos, refractDir);
            sst_trace(refractRay, 128u);
            vec3 refractColor = vec3(0.0);

            if (isEyeInWater == 1) {
                vec3 refractDirWorld = coords_dir_viewToWorld(refractDir);
                if (refractDirWorld.y > 0.0) {
                    SkyViewLutParams skyParams = atmospherics_air_lut_setupSkyViewLutParams(atmosphere, refractDirWorld);
                    refractColor = atmospherics_air_lut_sampleSkyViewLUT(atmosphere, skyParams, 0.0).inScattering;
                    if (refractRay.currT < 0.0 && refractRay.currT > -1.0) {
                        vec3 refractHitScreen = refractRay.pRayStart + refractRay.pRayDir * (refractRay.pRayVecLen * abs(refractRay.currT));
                        vec2 refractCoord = refractHitScreen.xy + (global_taaJitter * uval_mainImageSizeRcp);
                        float refractDepth = texture(usam_gbufferViewZ, refractCoord).r;
                        refractColor = mix(refractColor, sampling_catmullBicubic5Tap(usam_main, saturate(refractCoord) * uval_mainImageSize, 0.0, uval_mainImageSizeRcp).rgb, sst_edgeReductionFactor(refractHitScreen.xy, 2.0, vec2(0.0), vec2(1.5)));
                    }
                }
            } else {
                bool refractHit = refractRay.currT < 0.0 && refractRay.currT > -1.0;
                vec3 refractHitScreen = refractHit
                    ? refractRay.pRayStart + refractRay.pRayDir * (refractRay.pRayVecLen * abs(refractRay.currT))
                    : vec3(screenPos, 0.0);
                vec2 refractCoord = refractHit ? (refractHitScreen.xy + (global_taaJitter * uval_mainImageSizeRcp)) : screenPos;
                float refractDepth = texture(usam_gbufferViewZ, refractCoord).r;
                if (refractDepth > startViewZ) {
                    refractCoord = screenPos;
                }
                refractColor = sampling_catmullBicubic5Tap(usam_main, saturate(refractCoord) * uval_mainImageSize, 0.0, uval_mainImageSizeRcp).rgb;
            }

            float MDotV = dot(microNormal, viewDir);
            transient_translucentRefraction_store(texelPos, vec4(refractColor, MDotV));

            SSTRay reflectRay = sstray_setup(texelPos, startViewPos, reflectDir);
            sst_trace(reflectRay, 128u);
            vec3 reflectDirWorld = coords_dir_viewToWorld(reflectDir);
            reflectDirWorld = rand_sampleInCone(reflectDirWorld, 0.005, noiseV);
            vec2 envSliceUV = vec2(-1.0);
            vec2 envSliceID = vec2(-1.0);
            coords_cubeMapForward(reflectDirWorld, envSliceUV, envSliceID);
            ivec2 envTexel = ivec2((envSliceUV + envSliceID) * ENV_PROBE_SIZE);
            EnvProbeData envData = envProbe_decode(texelFetch(usam_envProbe, envTexel, 0));
            vec3 reflectColor = envData.radiance.rgb;
            if (envProbe_isSky(envData) && reflectDirWorld.y > 0.0) {
                AtmosphereParameters atmosphere = getAtmosphereParameters();
                SkyViewLutParams skyParams = atmospherics_air_lut_setupSkyViewLutParams(atmosphere, reflectDirWorld);
                reflectColor = atmospherics_air_lut_sampleSkyViewLUT(atmosphere, skyParams, 0.0).inScattering;
            }
            if (reflectRay.currT < 0.0 && reflectRay.currT > -1.0) {
                vec3 reflectHitScreen = reflectRay.pRayStart + reflectRay.pRayDir * (reflectRay.pRayVecLen * abs(reflectRay.currT));
                vec2 sampleCoord = saturate(reflectHitScreen.xy + (global_taaJitter * uval_mainImageSizeRcp));
                vec3 hitGeomNormal = transient_geomViewNormal_sample(sampleCoord).rgb;
                vec3 hitColor = sampling_catmullBicubic5Tap(usam_main, sampleCoord * uval_mainImageSize, 0.0, uval_mainImageSizeRcp).rgb;
                float mixFactor = sst_edgeReductionFactor(reflectHitScreen.xy, 4.0, vec2(0.5), vec2(0.95));
                hitGeomNormal = normalize(hitGeomNormal * 2.0 - 1.0);
                float hitDot = dot(-reflectDir, hitGeomNormal);
                float reflectDepth = texture(usam_gbufferViewZ, sampleCoord).r;
                bool skyReflected = reflectDepth < -far;
                mixFactor *= saturate(float(hitDot > 0.0) + float(skyReflected));

                if (isEyeInWater == 1) {
                    if (!skyReflected) {
                        reflectColor = mix(reflectColor, hitColor, mixFactor);
                    }
                } else {
                    reflectColor = mix(reflectColor, hitColor, mixFactor);
                }
            }

            float NDotL = dot(gData.normal, reflectDir);
            transient_translucentReflection_store(texelPos, vec4(reflectColor, NDotL));
        }
    }
}