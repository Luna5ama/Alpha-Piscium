#version 460 compatibility

#include "/util/GBufferData.glsl"
#include "/util/Material.glsl"
#include "/util/Rand.glsl"
#include "/util/Hash.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform restrict image2D uimg_temp3;
layout(rgba16f) uniform restrict image2D uimg_rgba16f;
layout(rgba32ui) uniform restrict uimage2D uimg_rgba32ui;
#include "/techniques/SSGI.glsl"

#if USE_REFERENCE
void main() {

}
#else
void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    sst_init();

    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        vec4 ssgiOut = vec4(0.0, 0.0, 0.0, -1.0);
        ReSTIRReservoir temporalReservoir = restir_initReservoir(texelPos);
        if (RANDOM_FRAME < MAX_FRAMES){
            if (RANDOM_FRAME >= 0) {
                float viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).x;
                vec2 screenPos = coords_texelToUV(texelPos, uval_mainImageSizeRcp);
                vec3 viewPos = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);

                GBufferData gData = gbufferData_init();
                gbufferData1_unpack(texelFetch(usam_gbufferData1, texelPos, 0), gData);
                gbufferData2_unpack(texelFetch(usam_gbufferData2, texelPos, 0), gData);
                Material material = material_decode(gData);

                uvec3 baseRandKey = uvec3(texelPos, RANDOM_FRAME);

                temporalReservoir = restir_loadReservoir(texelPos, 0);

                const uint MAX_AGE = 100u;

                float wSum = 0.0;
                vec4 prevSample = vec4(0.0);
                vec3 prevHitNormal = vec3(0.0);

                if (restir_isReservoirValid(temporalReservoir)) {
                    vec3 prevSampleDirView = temporalReservoir.Y.xyz;

                    float prevHitDistance = temporalReservoir.Y.w;
                    if (prevHitDistance > 0.0){
                        vec3 prevHitViewPos = viewPos + prevSampleDirView * prevHitDistance;
                        vec3 prevHitScreenPos = coords_viewToScreen(prevHitViewPos, global_camProj);
                        ivec2 prevHitTexelPos = ivec2(prevHitScreenPos.xy * uval_mainImageSize);

                        vec3 prevHitRadiance = transient_giRadianceInput_fetch(prevHitTexelPos).rgb;
                        float brdf = saturate(dot(gData.normal, prevSampleDirView)) / PI;
                        prevSample = vec4(prevHitRadiance, brdf);

                        GBufferData prevGData = gbufferData_init();
                        gbufferData1_unpack(texelFetch(usam_gbufferData1, prevHitTexelPos, 0), prevGData);
                        prevHitNormal = prevGData.normal;
                    }

//                    float prevHitDistance;
//                    prevSample = ssgiEvalF(viewPos, gData, prevSampleDirView, prevHitDistance);
//                    prevPHat = length(prevSample);
                } else {
                    temporalReservoir.m = 0u;
                }

                float prevPHat = length(prevSample.xyz * prevSample.w);
                wSum = max(0.0, temporalReservoir.avgWY) * float(temporalReservoir.m) * prevPHat;

                #if SPATIAL_REUSE_FEEDBACK
                if (temporalReservoir.m < SPATIAL_REUSE_FEEDBACK) {
                    ReSTIRReservoir prevSpatialReservoir =  restir_loadReservoir(texelPos, 1);

                    vec3 prevSpatialSampleDirView = prevSpatialReservoir.Y.xyz;
                    float prevSpatialHitDistance = prevSpatialReservoir.Y.w;

                    vec3 prevSpatialHitViewPos = viewPos + prevSpatialSampleDirView * prevSpatialHitDistance;
                    vec3 prevSpatialHitScreenPos = coords_viewToScreen(prevSpatialHitViewPos, global_camProj);
                    ivec2 prevSpatialHitTexelPos = ivec2(prevSpatialHitScreenPos.xy * uval_mainImageSize);
                    vec3 prevSpatialHitRadiance = transient_giRadianceInput_fetch(prevSpatialHitTexelPos).rgb;
                    float brdf = saturate(dot(gData.normal, prevSpatialSampleDirView)) / PI;
                    float prevSpatialPHat = length(prevSpatialHitRadiance * brdf);

                    float prevSpatialWi = max(prevSpatialReservoir.avgWY * prevSpatialPHat * float(prevSpatialReservoir.m), 0.0);
                    float reservoirRand2 = hash_uintToFloat(hash_44_q3(uvec4(baseRandKey, 456546u)).w);

                    if (restir_isReservoirValid(prevSpatialReservoir)) {
                        if (restir_updateReservoir(
                            temporalReservoir,
                            wSum,
                            prevSpatialReservoir.Y,
                            prevSpatialWi,
                            prevSpatialReservoir.m,
                            prevSpatialReservoir.age,
                            reservoirRand2
                        )) {
                            prevPHat = prevSpatialPHat;
                            prevSample = vec4(prevSpatialHitRadiance, brdf);
                            GBufferData prevSpatialHitGData = gbufferData_init();
                            gbufferData1_unpack(texelFetch(usam_gbufferData1, prevSpatialHitTexelPos, 0), prevSpatialHitGData);
                            prevHitNormal = prevSpatialHitGData.normal;
                        }
                    }
                }
                #endif
                if (temporalReservoir.age >= MAX_AGE) {
                    temporalReservoir = restir_initReservoir(texelPos);
                }

                {
//                    vec2 rand2 = hash_uintToFloat(hash_44_q3(uvec4(baseRandKey, 12312745u)).zw);
//                    vec4 sampleDirTangentAndPdf = rand_sampleInCosineWeightedHemisphere(rand2);
//                    float samplePdf = sampleDirTangentAndPdf.w;
//                    vec3 sampleDirView = normalize(material.tbn * sampleDirTangentAndPdf.xyz);
//                    vec4 ssgiData = uintBitsToFloat(imageLoad(uimg_csrgba32ui, csrgba32ui_temp4_texelToTexel(texelPos)));
//                    float hitDistance = ssgiData.w;
//                    vec3 initalSample = ssgiData.xyz;


//                    InitialSampleData initialSample = initialSampleData_init();
//                    {
//                        uvec3 baseRandKey = uvec3(texelPos, RANDOM_FRAME);
//                        vec2 rand2 = hash_uintToFloat(hash_44_q3(uvec4(baseRandKey, 12312745u)).zw);
//                        //                vec2 rand2 = rand_stbnVec2(texelPos, RANDOM_FRAME);
//                        //                vec4 sampleDirTangentAndPdf = rand_sampleInCosineWeightedHemisphere(rand2);
//                        vec4 sampleDirTangentAndPdf = rand_sampleInHemisphere(rand2);
//                        vec3 sampleDirView = normalize(material.tbn * sampleDirTangentAndPdf.xyz);
//
//                        //                ivec2 stbnPos = texelPos + ivec2(rand_r2Seq2(RANDOM_FRAME / 64u) * vec2(128, 128));
//                        //                vec3 sampleDirTangent = rand_stbnUnitVec3Cosine(stbnPos, RANDOM_FRAME);
//                        //                vec3 sampleDirView = normalize(material.tbn * sampleDirTangent);
//
//                        vec4 ssgiOut = vec4(0.0);
//                        //                sharedData[gl_LocalInvocationIndex] = sampleDirView;
//                        vec4 resultStuff = ssgiEvalF2(viewPos, sampleDirView);
//
//                        initialSample.hitRadiance = resultStuff.xyz;
//                        initialSample.directionAndLength.xyz = sampleDirView;
//                        initialSample.directionAndLength.w = resultStuff.w;
//                    }

                    InitialSampleData initialSample = initialSampleData_unpack(transient_restir_initialSample_load(texelPos));
                    vec3 hitRadiance = initialSample.hitRadiance;
                    vec3 sampleDirView = initialSample.directionAndLength.xyz;
                    float hitDistance = initialSample.directionAndLength.w;


                    float brdf = saturate(dot(gData.normal, sampleDirView)) / PI;
                    vec3 initalSample = brdf * hitRadiance;

//                    float samplePdf = saturate(dot(gData.normal, sampleDirView)) / PI;
                    float samplePdf = brdf;
//                    float samplePdf = 1.0 / (2.0 * PI);

                    float newPHat = length(initalSample);
                    float newWi = samplePdf <= 0.0 ? 0.0 : newPHat / samplePdf;

                    float reservoirRand1 = hash_uintToFloat(hash_44_q3(uvec4(baseRandKey, 547679546u)).w);

                    float reservoirPHat = prevPHat;
                    vec4 finalSample = prevSample;
                    vec3 hitNormal = prevHitNormal;
                    if (restir_updateReservoir(temporalReservoir, wSum, vec4(sampleDirView, hitDistance), newWi, 1u, 0u, reservoirRand1)) {
                        reservoirPHat = newPHat;
                        finalSample = vec4(hitRadiance, brdf);

                        vec3 hitViewPos = viewPos + sampleDirView * hitDistance;
                        vec3 hitScreenPos = coords_viewToScreen(hitViewPos, global_camProj);
                        ivec2 hitTexelPos = ivec2(hitScreenPos.xy * uval_mainImageSize);
                        GBufferData hitGData = gbufferData_init();
                        gbufferData1_unpack(texelFetch(usam_gbufferData1, hitTexelPos, 0), hitGData);
                        hitNormal = hitGData.normal;
                    }
                    float avgWSum = wSum / float(temporalReservoir.m);
                    temporalReservoir.avgWY = reservoirPHat <= 0.0 ? 0.0 : (avgWSum / reservoirPHat);
                    temporalReservoir.m = clamp(temporalReservoir.m, 0u, 20u);
                    ssgiOut = vec4(finalSample.xyz * finalSample.w * temporalReservoir.avgWY, temporalReservoir.Y.w);

                    SpatialSampleData spatialSample = spatialSampleData_init();
                    spatialSample.hitRadiance = finalSample.xyz;
                    spatialSample.geomNormal = gData.geomNormal;
                    spatialSample.normal = gData.normal;
                    spatialSample.hitNormal = hitNormal;
                    transient_restir_spatialInput_store(texelPos, spatialSampleData_pack(spatialSample));

                    temporalReservoir.age++;
                }
            }
        }
        transient_ssgiOut_store(texelPos, ssgiOut);
        restir_storeReservoir(texelPos, temporalReservoir, 0);
    }
}
#endif