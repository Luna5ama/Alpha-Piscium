#version 460 compatibility

#include "/techniques/SSGI.glsl"
#include "/techniques/SST.glsl"
#include "/util/GBufferData.glsl"
#include "/util/Material.glsl"
#include "/util/Rand.glsl"
#include "/util/Hash.glsl"

#define SSP 32u

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform restrict image2D uimg_temp1;

vec3 ssgiRef(uint sampleIndex, ivec2 texelPos) {
    uint finalIndex = RANDOM_FRAME * SSP + sampleIndex;
    vec3 result = vec3(0.0);
    GBufferData gData = gbufferData_init();
    gbufferData1_unpack(texelFetch(usam_gbufferData1, texelPos, 0), gData);
    gbufferData2_unpack(texelFetch(usam_gbufferData2, texelPos, 0), gData);
    Material material = material_decode(gData);
    #define RAND_TYPE 2
    #if RAND_TYPE == 0
    vec2 rand2 = rand_r2Seq2(frameCounter * SSP + sampleIndex);
    #elif RAND_TYPE == 1
    vec2 rand2 = hash_uintToFloat(hash_33_q3(uvec3(texelPos, finalIndex)).xy);
    #else
    ivec2 stbnPos = texelPos + ivec2(rand_r2Seq2(finalIndex / 64) * vec2(128, 128));
    vec2 rand2 = rand_stbnVec2(stbnPos, finalIndex % 64u);
    #endif

    vec3 sampleDirTangent = rand_sampleInHemistexelPosphere(rand2);
    vec3 sampleDirView = normalize(material.tbn * sampleDirTangent);
    float viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).x;
    vec2 screenPos = coords_texelToUV(texelPos, uval_mainImageSizeRcp);
    vec3 viewPos = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);

    SSTResult sstResult = sst_trace(viewPos, sampleDirView, 0.01);
    float samplePdf = 1.0 / (2.0 * PI);

    if (sstResult.hit) {
        vec3 hitRadiance = texture(usam_temp2, sstResult.hitScreenPos.xy).rgb;
        float brdf = saturate(dot(gData.normal, sampleDirView)) / PI;
        vec3 f = brdf * hitRadiance;
        result = f / samplePdf;
    }

    return result;
}

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    sst_init();

    if (RANDOM_FRAME < MAX_FRAMES){
        vec4 ssgiOut = vec4(0.0);
        if (RANDOM_FRAME >= 0) {
            ssgiOut = imageLoad(uimg_temp1, texelPos);
            for (uint i = 0u; i < SSP; i++) {
                vec3 result;
                #if USE_REFERENCE
                result = ssgiRef(i, texelPos);
                #else
                //        result =
                #endif
                ssgiOut.a += 1.0;
                ssgiOut.rgb = mix(ssgiOut.rgb, result, 1.0 / ssgiOut.a);
            }
        }
        imageStore(uimg_temp1, texelPos, ssgiOut);
    } else {
        if (all(greaterThan(texelPos, uval_mainImageSizeI - 8))) {
            imageStore(uimg_temp1, texelPos, vec4(111.0));
        }
    }
}