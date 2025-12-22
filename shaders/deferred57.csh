#version 460 compatibility

#include "/util/GBufferData.glsl"
#include "/util/Material.glsl"
#include "/util/Rand.glsl"
#include "/util/Hash.glsl"

layout(local_size_x = 16, local_size_y = 8) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba32ui) uniform restrict uimage2D uimg_rgba32ui;
#include "/techniques/SSGI.glsl"

shared vec3 sharedData[128];

#if USE_REFERENCE
void main() {

}
#else
void main() {

    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    sst_init();

    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
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
                vec2 rand2 = hash_uintToFloat(hash_44_q3(uvec4(baseRandKey, 12312745u)).zw);
//                vec2 rand2 = rand_stbnVec2(texelPos, RANDOM_FRAME);
                vec4 sampleDirTangentAndPdf = rand_sampleInCosineWeightedHemisphere(rand2);
//                vec4 sampleDirTangentAndPdf = rand_sampleInHemisphere(rand2);
                vec3 sampleDirView = normalize(material.tbn * sampleDirTangentAndPdf.xyz);

//                ivec2 stbnPos = texelPos + ivec2(rand_r2Seq2(RANDOM_FRAME / 64u) * vec2(128, 128));
//                vec3 sampleDirTangent = rand_stbnUnitVec3Cosine(stbnPos, RANDOM_FRAME);
//                vec3 sampleDirView = normalize(material.tbn * sampleDirTangent);

                vec4 ssgiOut = vec4(0.0);
//                sharedData[gl_LocalInvocationIndex] = sampleDirView;
                vec4 resultStuff = ssgiEvalF2(viewPos, sampleDirView);

                InitialSampleData initialSample = initialSampleData_init();
                initialSample.hitRadiance = resultStuff.xyz;
                initialSample.directionAndLength.xyz = sampleDirView;
                initialSample.directionAndLength.w = resultStuff.w;
                transient_restir_initialSample_store(texelPos, initialSampleData_pack(initialSample));
            } else {
                ReSTIRReservoir newReservoir = restir_initReservoir(texelPos);
                restir_storeReservoir(texelPos, newReservoir, 0);
                restir_storeReservoir(texelPos, newReservoir, 1);
            }
        }
    }
}
#endif