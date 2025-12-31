#define MATERIAL_TRANSLUCENT a

#include "/techniques/atmospherics/air/lut/API.glsl"
#include "/techniques/EnvProbe.glsl"
#include "/techniques/textile/CSR32F.glsl"
#include "/techniques/SST.glsl"
#include "/util/Celestial.glsl"
#include "/util/FullScreenComp.glsl"
#include "/util/GBufferData.glsl"
#include "/util/Material.glsl"
#include "/util/Fresnel.glsl"
#include "/util/BSDF.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform restrict writeonly image2D uimg_rgba16f;

float edgeReductionFactor(vec2 screenPos) {
    const float SQUIRCLE_M = 4.0;
    vec2 ndcPos = screenPos * 2.0 - 1.0;
    vec2 squircle = pow(smoothstep(0.5, 0.95, abs(ndcPos)), vec2(SQUIRCLE_M));
    return saturate(1.0 - (squircle.x + squircle.y));
}

float edgeReductionFactor2(vec2 screenPos) {
    const float SQUIRCLE_M = 2.0;
    vec2 ndcPos = screenPos * 2.0 - 1.0;
    vec2 squircle = pow(saturate(1.5 * abs(ndcPos)), vec2(SQUIRCLE_M));
    return saturate(1.0 - (squircle.x + squircle.y));
}

// from https://github.com/GameTechDev/TAA
vec4 BicubicSampling56(sampler2D samplerV, vec2 inHistoryUV, vec2 resolution) {
    vec2 inHistoryST = inHistoryUV * resolution;
    const vec2 rcpResolution = rcp(resolution);
    const vec2 fractional = fract(inHistoryST - 0.5);
    const vec2 uv = (floor(inHistoryST - 0.5) + vec2(0.5f, 0.5f)) * rcpResolution;

    // 5-tap bicubic sampling (for Hermite/Carmull-Rom filter) -- (approximate from original 16->9-tap bilinear fetching)
    const vec2 t = vec2(fractional);
    const vec2 t2 = vec2(fractional * fractional);
    const vec2 t3 = vec2(fractional * fractional * fractional);
    const float s = float(0.0);
    const vec2 w0 = -s * t3 + float(2.f) * s * t2 - s * t;
    const vec2 w1 = (float(2.f) - s) * t3 + (s - float(3.f)) * t2 + float(1.f);
    const vec2 w2 = (s - float(2.f)) * t3 + (3 - float(2.f) * s) * t2 + s * t;
    const vec2 w3 = s * t3 - s * t2;
    const vec2 s0 = w1 + w2;
    const vec2 f0 = w2 / (w1 + w2);
    const vec2 m0 = uv + f0 * rcpResolution;
    const vec2 tc0 = uv - 1.f * rcpResolution;
    const vec2 tc3 = uv + 2.f * rcpResolution;

    const vec4 A = vec4(texture(samplerV, vec2(m0.x, tc0.y)));
    const vec4 B = vec4(texture(samplerV, vec2(tc0.x, m0.y)));
    const vec4 C = vec4(texture(samplerV, vec2(m0.x, m0.y)));
    const vec4 D = vec4(texture(samplerV, vec2(tc3.x, m0.y)));
    const vec4 E = vec4(texture(samplerV, vec2(m0.x, tc3.y)));
    const vec4 color = (float(0.5f) * (A + B) * w0.x + A * s0.x + float(0.5f) * (A + B) * w3.x) * w0.y + (B * w0.x + C * s0.x + D * w3.x) * s0.y + (float(0.5f) * (B + E) * w0.x + E * s0.x + float(0.5f) * (D + E) * w3.x) * w3.y;
    return color;
}

void main() {
    sst_init();

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
            bool isWater = gData.materialID == 3u;

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

            SSTResult refractResult = sst_trace(startViewPos, refractDir, 0.005);
            vec3 refractColor = vec3(0.0);

            if (isEyeInWater == 1) {
                vec3 refractDirWorld = coords_dir_viewToWorld(refractDir);
                if (refractDirWorld.y > 0.0) {
                    SkyViewLutParams skyParams = atmospherics_air_lut_setupSkyViewLutParams(atmosphere, refractDirWorld);
                    refractColor = atmospherics_air_lut_sampleSkyViewLUT(atmosphere, skyParams, 0.0).inScattering;
                    if (refractResult.hit) {
                        vec2 refractCoord = refractResult.hitScreenPos.xy + (global_taaJitter * uval_mainImageSizeRcp);
                        float refractDepth = texture(usam_gbufferViewZ, refractCoord).r;
                        refractColor = mix(refractColor, BicubicSampling56(usam_main, saturate(refractCoord), uval_mainImageSize).rgb, edgeReductionFactor2(refractResult.hitScreenPos.xy));
                    }
                }
            } else {
                vec2 refractCoord = refractResult.hit ? (refractResult.hitScreenPos.xy + (global_taaJitter * uval_mainImageSizeRcp)) : screenPos;
                float refractDepth = texture(usam_gbufferViewZ, refractCoord).r;
                if (refractDepth > startViewZ) {
                    refractCoord = screenPos;
                }
                refractColor = BicubicSampling56(usam_main, saturate(refractCoord), uval_mainImageSize).rgb;
            }

            //            vec3 refractColor = texture(usam_main, refractCoord).rgb;

            float MDotV = dot(microNormal, viewDir);
            transient_translucentRefraction_store(texelPos, vec4(refractColor, MDotV));

            SSTResult reflectResult = sst_trace(startViewPos, reflectDir, 0.005);
            vec3 reflectDirWorld = coords_dir_viewToWorld(reflectDir);
            reflectDirWorld = rand_sampleInCone(reflectDirWorld, 0.005, noiseV);
            vec2 envSliceUV = vec2(-1.0);
            vec2 envSliceID = vec2(-1.0);
            coords_cubeMapForward(reflectDirWorld, envSliceUV, envSliceID);
            ivec2 envTexel = ivec2((envSliceUV + envSliceID) * ENV_PROBE_SIZE);
            EnvProbeData envData = envProbe_decode(texelFetch(usam_envProbe, envTexel, 0));
            vec3 reflectColor = envData.radiance.rgb * RCP_PI;
            if (envProbe_isSky(envData) && reflectDirWorld.y > 0.0) {
                AtmosphereParameters atmosphere = getAtmosphereParameters();
                SkyViewLutParams skyParams = atmospherics_air_lut_setupSkyViewLutParams(atmosphere, reflectDirWorld);
                reflectColor = atmospherics_air_lut_sampleSkyViewLUT(atmosphere, skyParams, 0.0).inScattering;
            }
            if (reflectResult.hit) {
                vec2 sampleCoord = saturate(reflectResult.hitScreenPos.xy + (global_taaJitter * uval_mainImageSizeRcp));
                vec3 hitGeomNormal = transient_geomViewNormal_sample(sampleCoord).rgb;
                vec3 hitColor = BicubicSampling56(usam_main, sampleCoord, uval_mainImageSize).rgb;
                float mixFactor = edgeReductionFactor(reflectResult.hitScreenPos.xy);
                hitGeomNormal = normalize(hitGeomNormal * 2.0 - 1.0);
                float hitDot = dot(-reflectDir, hitGeomNormal);
                mixFactor *= float(hitDot > 0.0);

                if (isEyeInWater == 1) {
                    float reflectDepth = texture(usam_gbufferViewZ, sampleCoord).r;
                    if (reflectDepth > -far) {
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