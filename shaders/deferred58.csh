#version 460 compatibility

#include "/techniques/SSGI.glsl"
#include "/util/GBufferData.glsl"
#include "/util/Material.glsl"
#include "/util/Rand.glsl"
#include "/util/Hash.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform restrict image2D uimg_temp3;

#if USE_REFERENCE
void main() {

}
#else
void main() {

    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    sst_init();

    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        vec4 ssgiOut = vec4(0.0);
        if (RANDOM_FRAME < MAX_FRAMES){
            if (RANDOM_FRAME >= 0) {
                ReSTIRReservoir temporalReservoir = restir_initReservoir(texelPos);

                float viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).x;
                vec2 screenPos = coords_texelToUV(texelPos, uval_mainImageSizeRcp);
                vec3 viewPos = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);

                GBufferData gData = gbufferData_init();
                gbufferData1_unpack(texelFetch(usam_gbufferData1, texelPos, 0), gData);
                gbufferData2_unpack(texelFetch(usam_gbufferData2, texelPos, 0), gData);
                Material material = material_decode(gData);

                uvec3 baseRandKey = uvec3(texelPos, RANDOM_FRAME);

                temporalReservoir = restir_loadReservoir(texelPos, 0);
//                if (temporalReservoir.m < 20u)
//                    temporalReservoir = restir_loadReservoir(texelPos, 1);

                const uint MAX_AGE = 100u;

                float wSum = 0.0;
                vec4 prevSample = vec4(0.0);
                vec3 prevHitNormal = vec3(0.0);

                if (restir_isReservoirValid(temporalReservoir)) {
                    vec3 prevSampleDirView = temporalReservoir.Y.xyz;


                    //                    float prevSamplePdf = 1.0 / (2.0 * PI);
                    float prevSamplePdf = saturate(dot(gData.normal, prevSampleDirView)) / PI;

                    float prevHitDistance = temporalReservoir.Y.w;
                    vec3 prevHitViewPos = viewPos + prevSampleDirView * prevHitDistance;
                    vec3 prevHitScreenPos = coords_viewToScreen(prevHitViewPos, global_camProj);
                    ivec2 prevHitTexelPos = ivec2(prevHitScreenPos.xy * uval_mainImageSize);
                    vec3 prevHitRadiance = texelFetch(usam_temp2, prevHitTexelPos, 0).rgb;
                    float brdf = saturate(dot(gData.normal, prevSampleDirView)) / PI;
                    prevSample = vec4(prevHitRadiance, brdf);

                    GBufferData prevGData = gbufferData_init();
                    gbufferData1_unpack(texelFetch(usam_gbufferData1, prevHitTexelPos, 0), prevGData);
                    prevHitNormal = prevGData.normal;

//                    float prevHitDistance;
//                    prevSample = ssgiEvalF(viewPos, gData, prevSampleDirView, prevHitDistance);
//                    prevPHat = length(prevSample);
                } else {
                    temporalReservoir.m = 0u;
                }
                if (temporalReservoir.age > MAX_AGE) {
                    temporalReservoir.m = 0u;
                }

                float prevPHat = length(prevSample.xyz * prevSample.w);
                wSum = max(0.0, temporalReservoir.avgWY) * float(temporalReservoir.m) * prevPHat;

                {
//                    vec2 rand2 = hash_uintToFloat(hash_44_q3(uvec4(baseRandKey, 12312745u)).zw);
//                    vec4 sampleDirTangentAndPdf = rand_sampleInCosineWeightedHemisphere(rand2);
//                    float samplePdf = sampleDirTangentAndPdf.w;
//                    vec3 sampleDirView = normalize(material.tbn * sampleDirTangentAndPdf.xyz);
//                    vec4 ssgiData = uintBitsToFloat(imageLoad(uimg_csrgba32ui, csrgba32ui_temp4_texelToTexel(texelPos)));
//                    float hitDistance = ssgiData.w;
//                    vec3 initalSample = ssgiData.xyz;

                    InitialSampleData initialSample = initialSampleData_unpack(imageLoad(uimg_csrgba32ui, csrgba32ui_temp4_texelToTexel(texelPos)));
                    vec3 hitRadiance = initialSample.hitRadiance;
                    vec3 sampleDirView = initialSample.directionAndLength.xyz;
                    float hitDistance = initialSample.directionAndLength.w;


                    float brdf = saturate(dot(gData.normal, sampleDirView)) / PI;
                    vec3 f = brdf * hitRadiance;
                    vec3 initalSample = f;

//                    float samplePdf = saturate(dot(gData.normal, sampleDirView)) / PI;
                    float samplePdf = brdf;

                    float newPHat = length(initalSample);
                    float newWi = samplePdf <= 0.0 ? 0.0 : newPHat / samplePdf;

                    float reservoirRand1 = hash_uintToFloat(hash_44_q3(uvec4(baseRandKey, 547679546u)).w);

                    float reservoirPHat = prevPHat;
                    vec4 finalSample = prevSample;
                    vec3 hitNormal = prevHitNormal;
                    if (restir_updateReservoir(temporalReservoir, wSum, vec4(sampleDirView, hitDistance), newWi, 1u, reservoirRand1)) {
                        reservoirPHat = newPHat;
                        finalSample = vec4(hitRadiance, f);

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
                    ssgiOut = vec4(finalSample.xyz * finalSample.w * temporalReservoir.avgWY, 1.0);

                    SpatialSampleData spatialSample = spatialSampleData_init();
                    spatialSample.hitRadiance = finalSample.xyz;
                    spatialSample.geomNormal = gData.geomNormal;
                    spatialSample.normal = gData.normal;
                    spatialSample.hitNormal = hitNormal;
                    imageStore(uimg_csrgba32ui, csrgba32ui_temp3_texelToTexel(texelPos), spatialSampleData_pack(spatialSample));

                    temporalReservoir.age++;
                }

                restir_storeReservoir(texelPos, temporalReservoir, 0);
            }
        }
        imageStore(uimg_csrgba32ui, csrgba32ui_temp4_texelToTexel(texelPos), floatBitsToUint(ssgiOut));
    }
}
#endif