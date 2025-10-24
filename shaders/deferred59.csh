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
                ReSTIRReservoir newReservoir = restir_initReservoir(texelPos);

                float viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).x;
                vec2 screenPos = coords_texelToUV(texelPos, uval_mainImageSizeRcp);
                vec3 viewPos = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);

                GBufferData gData = gbufferData_init();
                gbufferData1_unpack(texelFetch(usam_gbufferData1, texelPos, 0), gData);
                gbufferData2_unpack(texelFetch(usam_gbufferData2, texelPos, 0), gData);
                Material material = material_decode(gData);

                uvec3 baseRandKey = uvec3(texelPos, RANDOM_FRAME);

                vec2 rand2 = hash_uintToFloat(hash_44_q3(uvec4(baseRandKey, 0)).xy);
//                vec4 sampleDirTangentAndPdf = rand_sampleInHemisphere(rand2);
                vec4 sampleDirTangentAndPdf = rand_sampleInCosineWeightedHemisphere(rand2);
                vec3 sampleDirView = normalize(material.tbn * sampleDirTangentAndPdf.xyz);
                float samplePdf = sampleDirTangentAndPdf.w;
//                samplePdf = 1.0 / (2.0 * PI);
                ivec2 hitTexelPos;

                vec3 initalSample = ssgiEvalF(viewPos, gData, sampleDirView, hitTexelPos);
                float pHatXInitial = length(initalSample);

                float reservoirRand1 = hash_uintToFloat(hash_44_q3(uvec4(baseRandKey, 1)).x);
                {
                    float WXi = rcp(samplePdf); // WXi: unbiased contribution weight
                    //        float mi = float(m) / float(max(1u, reservoir.m));
//                            float mi = pX / (pX + reservoir.pY);
//                    float mi = 1.0;
                    float wi = /*mi **/ pHatXInitial * WXi; // Wi: Resampling weight
                    if (restir_updateReservoir(newReservoir, hitTexelPos, wi, 1u, reservoirRand1)) {
                        restir_updateReservoirWY(newReservoir, pHatXInitial);
                        ssgiOut = vec4(initalSample * newReservoir.wY, 1.0);
                    }
                }

                ReSTIRReservoir prevReservoir = restir_loadReservoir(texelPos, 0);

                uint newM = 20 * newReservoir.m;
                uint prevM = prevReservoir.m;
                if (prevReservoir.m > newM) {
                    prevReservoir.wSum *= float(newM) / float(prevReservoir.m);
                    prevReservoir.m = newM;
                }

                if (restir_isReservoirValid(prevReservoir)) {
                    ivec2 prevHitTexelPos = ivec2(prevReservoir.Y);
                    float prevHitViewZ = texelFetch(usam_gbufferViewZ, prevHitTexelPos, 0).x;
                    vec2 prevHitScreenPos = coords_texelToUV(prevHitTexelPos, uval_mainImageSizeRcp);
                    vec3 prevHitViewPos = coords_toViewCoord(prevHitScreenPos, prevHitViewZ, global_camProjInverse);
                    vec3 prevSampleDirView = normalize(prevHitViewPos - viewPos);
                    float prevSamplePdf = saturate(dot(gData.normal, prevSampleDirView)) / PI;
//                    float prevSamplePdf = 1.0 / (2.0 * PI);
                    ivec2 newHitTexelPos;
                    vec3 prevSample = ssgiEvalF(viewPos, gData, prevSampleDirView, newHitTexelPos);
                        float prevPHatY = length(prevSample);
                        restir_updateReservoirWY(prevReservoir, prevPHatY);
                        float prevWi = (prevPHatY / (prevPHatY + pHatXInitial))  * prevPHatY * prevReservoir.wY * float(prevReservoir.m);

                        float reservoirRand2 = hash_uintToFloat(hash_44_q3(uvec4(baseRandKey, 2)).x);
                        if (restir_updateReservoir(newReservoir, newHitTexelPos, prevWi, prevReservoir.m, reservoirRand2)) {
                            restir_updateReservoirWY(newReservoir, prevPHatY);
                            ssgiOut = vec4(prevSample * newReservoir.wY, 1.0);
                        } else {
                            restir_updateReservoirWY(newReservoir, pHatXInitial);
                            ssgiOut = vec4(initalSample * newReservoir.wY, 1.0);
                        }
                } else  {
                    newReservoir.m += prevReservoir.m;
                    restir_updateReservoirWY(newReservoir, pHatXInitial);
                }

                restir_storeReservoir(texelPos, newReservoir, 0);
            } else {
                ReSTIRReservoir newReservoir = restir_initReservoir(texelPos);
                restir_storeReservoir(texelPos, newReservoir, 0);
            }
        }
//        imageStore(uimg_temp3, texelPos, ssgiOut);
        imageStore(uimg_csrgba32ui, csrgba32ui_restir2_texelToTexel(texelPos), floatBitsToUint(ssgiOut));
    }
}
#endif