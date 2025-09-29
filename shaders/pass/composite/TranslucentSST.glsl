#define MATERIAL_TRANSLUCENT a

#include "/techniques/atmospherics/air/lut/API.glsl"
#include "/techniques/EnvProbe.glsl"
#include "/techniques/SST.glsl"
#include "/util/Celestial.glsl"
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

float edgeReductionFactor2(vec2 screenPos) {
    const float SQUIRCLE_M = 1.0;
    vec2 ndcPos = screenPos * 2.0 - 1.0;
    vec2 squircle = pow(linearStep(0.5, 0.98, abs(ndcPos)), vec2(SQUIRCLE_M));
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

            vec3 viewDir = normalize(-startViewPos);

            vec3 localViewDir = normalize(material.tbnInv * viewDir);

            vec2 noiseV = rand_stbnVec2(texelPos, frameCounter);
            float pdfRatio = 1.0;
            vec3 tangentMicroNormal = bsdf_SphericalCapBoundedWithPDFRatio(noiseV, localViewDir, vec2(material.roughness), pdfRatio);
            vec3 microNormal = normalize(material.tbn * tangentMicroNormal);

            vec2 ndcPos = abs(screenPos * 2.0 - 1.0);

            float rior = AIR_IOR / mix(material.hardCodedIOR, 1.0, min(max2(ndcPos), 1.0 - pow2(saturate(dot(vec3(0.0, 0.0, 1.0), gData.geomNormal)))));
//            float rior = AIR_IOR / mix(material.hardCodedIOR, 1.0, 1.0 - saturate(dot(vec3(0.0, 0.0, 1.0), gData.geomNormal)));
            vec3 refractDir = refract(-viewDir, microNormal, rior);
            vec3 reflectDir = reflect(-viewDir, microNormal);

            SSTResult refractResult = sst_trace(startViewPos, refractDir, 0.01);
            vec2 refractCoord = refractResult.hit ? (refractResult.hitScreenPos.xy + (global_taaJitter * global_mainImageSizeRcp)) : screenPos;
            float refractDepth = texture(usam_gbufferViewZ, refractCoord).r;
            if (refractDepth > startViewZ) {
                refractCoord = screenPos;
            }

            vec3 refractColor = BicubicSampling56(usam_main, refractCoord, global_mainImageSize).rgb;
            //            vec3 refractColor = texture(usam_main, refractCoord).rgb;
            if (gData.materialID == 3u) {
                float refractViewZ = texture(usam_gbufferViewZ, refractCoord).r;
                vec3 refractViewPos = coords_toViewCoord(refractCoord, refractViewZ, global_camProjInverse);
                float refractDistance = distance(startViewPos, refractViewPos);
                refractDistance = min(refractDistance, far);
                vec3 scatteringCoeff = -log(gData.albedo);
                vec3 extinctionCoeff = scatteringCoeff * vec3(1.3, 1.1, 1.05) * 1.5;
                vec3 opticalDepth = extinctionCoeff * refractDistance;
                vec3 transmittance = exp(-opticalDepth);
                refractColor *= transmittance;
                vec3 sampleInSctr = scatteringCoeff * refractDistance;
                vec3 sampleInSctrInt = (sampleInSctr - sampleInSctr * transmittance) / extinctionCoeff;


                AtmosphereParameters atmosphere = getAtmosphereParameters();
                float shadowIsSun = float(all(equal(sunPosition, shadowLightPosition)));
                vec3 atmPos = atmosphere_viewToAtm(atmosphere, refractViewPos);
                atmPos.y = max(atmPos.y, atmosphere.bottom + 0.1);
                float viewAltitude = length(atmPos);
                vec3 upVector = atmPos / viewAltitude;
                const vec3 earthCenter = vec3(0.0, 0.0, 0.0);

                float cosSunZenith = dot(uval_sunDirWorld, vec3(0.0, 1.0, 0.0));
                vec3 tSun = atmospherics_air_lut_sampleTransmittance(atmosphere, cosSunZenith, viewAltitude);
                tSun *= float(raySphereIntersectNearest(atmPos, uval_sunDirWorld, earthCenter + PLANET_RADIUS_OFFSET * upVector, atmosphere.bottom) < 0.0);
                vec3 sunIrradiance = SUN_ILLUMINANCE * tSun * phasefunc_Rayleigh(dot(uval_sunDirView, refractDir));

                float cosMoonZenith = dot(uval_moonDirWorld, vec3(0.0, 1.0, 0.0));
                vec3 tMoon = atmospherics_air_lut_sampleTransmittance(atmosphere, cosMoonZenith, viewAltitude);
                tMoon *= float(raySphereIntersectNearest(atmPos, uval_moonDirWorld, earthCenter + PLANET_RADIUS_OFFSET * upVector, atmosphere.bottom) < 0.0);
                vec3 moonIrradiance = MOON_ILLUMINANCE * tMoon *  phasefunc_Rayleigh(dot(uval_moonDirView, refractDir));

                vec3 totalInSctr = (sunIrradiance + moonIrradiance) * sampleInSctrInt * 0.02;
                refractColor += totalInSctr;
            }

            float MDotV = dot(microNormal, viewDir);
            imageStore(uimg_temp1, texelPos, vec4(refractColor, MDotV));

            SSTResult reflectResult = sst_trace(startViewPos, reflectDir, 0.02);
            vec3 reflectDirWorld = coords_dir_viewToWorld(reflectDir);
            vec2 envSliceUV = vec2(-1.0);
            vec2 envSliceID = vec2(-1.0);
            coords_cubeMapForward(reflectDirWorld, envSliceUV, envSliceID);
            ivec2 envTexel = ivec2((envSliceUV + envSliceID) * ENV_PROBE_SIZE);
            EnvProbeData envData = envProbe_decode(texelFetch(usam_envProbe, envTexel, 0));
            vec3 reflectColor = envData.radiance.rgb;
            if (envProbe_isSky(envData)) {
                AtmosphereParameters atmosphere = getAtmosphereParameters();
                SkyViewLutParams skyParams = atmospherics_air_lut_setupSkyViewLutParams(atmosphere, reflectDirWorld);
                reflectColor = atmospherics_air_lut_sampleSkyViewLUT(atmosphere, skyParams, 0.0).inScattering;
            }
            if (reflectResult.hit) {
                vec2 sampleCoord = reflectResult.hitScreenPos.xy + (global_taaJitter * global_mainImageSizeRcp);
                reflectColor = mix(reflectColor, BicubicSampling56(usam_main, sampleCoord, global_mainImageSize).rgb, edgeReductionFactor(reflectResult.hitScreenPos.xy));
            }

            float NDotL = dot(gData.normal, reflectDir);
            imageStore(uimg_temp2, texelPos, vec4(reflectColor, NDotL));
        }
    }
}