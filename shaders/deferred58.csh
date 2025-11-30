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
                float prevPHat = 0.0;
                vec3 prevSample = vec3(0.0);

                if (restir_isReservoirValid(temporalReservoir)) {
                    vec3 prevSampleDirView = temporalReservoir.Y.xyz;
                    float prevSamplePdf = saturate(dot(gData.normal, prevSampleDirView)) / PI;
//                    float prevSamplePdf = 1.0 / (2.0 * PI);
                    float prevHitDistance;
                    prevSample = ssgiEvalF(viewPos, gData, prevSampleDirView, prevHitDistance);
                    prevPHat = length(prevSample);
                } else {
                    temporalReservoir.m = 0u;
                }
                if (temporalReservoir.age > MAX_AGE) {
                    temporalReservoir.m = 0u;
                }

                wSum = max(0.0, temporalReservoir.avgWY) * float(temporalReservoir.m) * prevPHat;

                {
                                        vec2 rand2 = hash_uintToFloat(hash_44_q3(uvec4(baseRandKey, 12312745u)).zw);
//                    ivec2 stbnPos = texelPos + ivec2(rand_r2Seq2(RANDOM_FRAME / 64) * vec2(128.0));
//                    vec2 rand2 = rand_stbnVec2(stbnPos, RANDOM_FRAME);
//                    vec4 sampleDirTangentAndPdf = rand_sampleInCosineWeightedHemisphere(rand2);
                    vec4 sampleDirTangentAndPdf = rand_sampleInHemisphere(rand2);
                    vec3 sampleDirView = normalize(material.tbn * sampleDirTangentAndPdf.xyz);
                    float samplePdf = sampleDirTangentAndPdf.w;
                    float hitDistance;

                    vec3 initalSample = ssgiEvalF(viewPos, gData, sampleDirView, hitDistance);
                    float newPHat = length(initalSample);
                    float newWi = newPHat / samplePdf;

                    float reservoirRand1 = hash_uintToFloat(hash_44_q3(uvec4(baseRandKey, 547679546u)).w);

                    float reservoirPHat = prevPHat;
                    vec3 finalSample = prevSample;
                    if (restir_updateReservoir(temporalReservoir, wSum, vec4(sampleDirView, hitDistance), newWi, 1u, reservoirRand1)) {
                        reservoirPHat = newPHat;
                        finalSample = initalSample;
                    }
                    float avgWSum = wSum / float(temporalReservoir.m);
                    temporalReservoir.avgWY = reservoirPHat <= 0.0 ? 0.0 : (avgWSum / reservoirPHat);
                    temporalReservoir.m = clamp(temporalReservoir.m, 0u, 20u);
                    ssgiOut = vec4(finalSample * temporalReservoir.avgWY, 1.0);

                    temporalReservoir.age++;
                }

                restir_storeReservoir(texelPos, temporalReservoir, 0);
            } else {
                ReSTIRReservoir newReservoir = restir_initReservoir(texelPos);
                restir_storeReservoir(texelPos, newReservoir, 0);
                restir_storeReservoir(texelPos, newReservoir, 1);
            }
        }
        imageStore(uimg_csrgba32ui, csrgba32ui_temp4_texelToTexel(texelPos), floatBitsToUint(ssgiOut));
    }
}
#endif