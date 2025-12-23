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
uint selectWeighted(vec4 weights, float rand) {
    vec4 prefixSum;
    prefixSum.x = weights.x;
    prefixSum.y = prefixSum.x + weights.y;
    prefixSum.z = prefixSum.y + weights.z;
    prefixSum.w = prefixSum.z + weights.w;

    float total = prefixSum.w;
    float threshold = rand * total;

    vec4 cmp = step(prefixSum, vec4(threshold));
    return uint(dot(cmp, vec4(1.0)));
}

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    sst_init();

    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        vec4 ssgiOut = vec4(0.0, 0.0, 0.0, -1.0);
        ReSTIRReservoir temporalReservoir = restir_initReservoir(texelPos);
        if (RANDOM_FRAME < MAX_FRAMES && RANDOM_FRAME >= 0){
            float viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).x;
            vec2 screenPos = coords_texelToUV(texelPos, uval_mainImageSizeRcp);
            vec3 viewPos = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);

            GBufferData gData = gbufferData_init();
            gbufferData1_unpack(texelFetch(usam_gbufferData1, texelPos, 0), gData);
            gbufferData2_unpack(texelFetch(usam_gbufferData2, texelPos, 0), gData);
            Material material = material_decode(gData);

            uvec3 baseRandKey = uvec3(texelPos, RANDOM_FRAME);

            float wSum = 0.0;
            vec4 prevSample = vec4(0.0);
            vec3 prevHitNormal = vec3(0.0);

            {
                uvec4 reprojInfoData = transient_gi_diffuse_reprojInfo_load(texelPos);
                ReprojectInfo reprojInfo = reprojectInfo_unpack(reprojInfoData);

                if (reprojInfo.historyResetFactor > 0.0) {
                    vec2 curr2PrevTexelPos = reprojInfo.curr2PrevScreenPos * uval_mainImageSize;
                    vec2 centerPixel = curr2PrevTexelPos - 0.5;
                    vec2 gatherOrigin = floor(centerPixel);
                    vec2 gatherTexelPos = gatherOrigin + 1.0;
                    vec2 pixelPosFract = fract(centerPixel);

                    vec2 bilinearWeights2 = pixelPosFract;
                    vec4 blinearWeights4;
                    blinearWeights4.yz = bilinearWeights2.xx;
                    blinearWeights4.xw = 1.0 - bilinearWeights2.xx;
                    blinearWeights4.xy *= bilinearWeights2.yy;
                    blinearWeights4.zw *= 1.0 - bilinearWeights2.yy;
                    vec4 finalWeights = blinearWeights4 * reprojInfo.bilateralWeights;

                    uint selectedIndex = selectWeighted(finalWeights, hash_uintToFloat(hash_44_q3(uvec4(baseRandKey, 987654u)).x));

                    vec4 texelPosXs = vec4(gatherTexelPos.x) + vec4(-0.5, 0.5, 0.5, -0.5);
                    vec4 texelPosYs = vec4(gatherTexelPos.y) + vec4(0.5, 0.5, -0.5, -0.5);
                    vec2 selectedTexelPos = vec2(texelPosXs[selectedIndex], texelPosYs[selectedIndex]);
                    ivec2 prevTexelPos = ivec2(selectedTexelPos);
                    imageStore(uimg_temp3, texelPos, vec4(reprojInfo.bilateralWeights));

                    ReSTIRReservoir prevTemporalReservoir = restir_reservoir_unpack(history_restir_reservoirTemporal_load(prevTexelPos));
                    if (restir_isReservoirValid(prevTemporalReservoir)) {
                        vec3 prevSampleDirView = prevTemporalReservoir.Y.xyz;
                        prevSampleDirView = coords_dir_viewToWorldPrev(prevSampleDirView);
                        prevSampleDirView = coords_dir_worldToView(prevSampleDirView);
                        prevTemporalReservoir.Y.xyz = prevSampleDirView;

                        float prevHitDistance = prevTemporalReservoir.Y.w;
                        if (prevHitDistance > 0.0){
                            vec3 prevHitViewPos = viewPos + prevSampleDirView * prevHitDistance;
                            vec3 prevHitScreenPos = coords_viewToScreen(prevHitViewPos, global_camProj);
                            ivec2 prevHitTexelPos = ivec2(prevHitScreenPos.xy * uval_mainImageSize);

                            vec3 prevHitRadiance = sampleIrradiance(prevHitTexelPos, -prevSampleDirView);
                            float brdf = saturate(dot(gData.normal, prevSampleDirView)) / PI;
                            prevSample = vec4(prevHitRadiance, brdf);


                            // TODO: retrace for temporal resampling
                            //                    float prevHitDistance;
                            //                    prevSample = ssgiEvalF(viewPos, gData, prevSampleDirView, prevHitDistance);
                            //                    prevPHat = length(prevSample);
                            GBufferData prevGData = gbufferData_init();
                            gbufferData1_unpack(texelFetch(usam_gbufferData1, prevHitTexelPos, 0), prevGData);
                            prevHitNormal = prevGData.normal;
                        }

                        temporalReservoir = prevTemporalReservoir;
                    }
                }
            }

            float prevPHat = length(prevSample.xyz * prevSample.w);
            wSum = max(0.0, temporalReservoir.avgWY) * float(temporalReservoir.m) * prevPHat;

            #if SPATIAL_REUSE_FEEDBACK
            if (temporalReservoir.m < SPATIAL_REUSE_FEEDBACK) {
                ReSTIRReservoir prevSpatialReservoir = restir_reservoir_unpack(history_restir_reservoirSpatial_load(texelPos));

                vec3 prevSpatialSampleDirView = prevSpatialReservoir.Y.xyz;
                prevSpatialSampleDirView = coords_dir_viewToWorldPrev(prevSpatialSampleDirView);
                prevSpatialSampleDirView = coords_dir_worldToView(prevSpatialSampleDirView);
                prevSpatialReservoir.Y.xyz = prevSpatialSampleDirView;

                float prevSpatialHitDistance = prevSpatialReservoir.Y.w;

                vec3 prevSpatialHitViewPos = viewPos + prevSpatialSampleDirView * prevSpatialHitDistance;
                vec3 prevSpatialHitScreenPos = coords_viewToScreen(prevSpatialHitViewPos, global_camProj);
                ivec2 prevSpatialHitTexelPos = ivec2(prevSpatialHitScreenPos.xy * uval_mainImageSize);
                vec3 prevSpatialHitRadiance = sampleIrradiance(prevSpatialHitTexelPos, -prevSpatialSampleDirView);
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

            const float RESET_START = 64.0;
            const float RESET_END = 256.0;
            float resetThreshold = linearStep(RESET_START, RESET_END, float(temporalReservoir.age));
            float resetRand = hash_uintToFloat(hash_44_q3(uvec4(baseRandKey, 123679546u)).w);
            if (resetRand < resetThreshold) {
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
                temporalReservoir.m = clamp(temporalReservoir.m, 0u, 32u);
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
        transient_ssgiOut_store(texelPos, ssgiOut);
        transient_restir_reservoirReprojected_store(texelPos, restir_reservoir_pack(temporalReservoir));
    }
}
#endif