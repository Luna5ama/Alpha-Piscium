#include "/util/Math.glsl"
#include "/util/GBufferData.glsl"
#include "/util/Coords.glsl"

layout(rgba32ui) uniform restrict readonly uimage2D uimg_voxyData;
layout(rgba32ui) uniform restrict uimage2D uimg_gbufferSolidData1;
layout(r32ui) uniform restrict uimage2D uimg_gbufferSolidData2;
layout(r32f) uniform restrict image2D uimg_gbufferSolidViewZ;

void voxy_merge() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(texelPos, uval_mainImageSizeI))) return;

    uvec4 voxyData = imageLoad(uimg_voxyData, texelPos);
    if (voxyData.w == 0u && voxyData.x == 0u) return;

    float voxyZ = uintBitsToFloat(voxyData.w);
    float solidZ = imageLoad(uimg_gbufferSolidViewZ, texelPos).r;

    if (voxyZ > solidZ) {
        imageStore(uimg_gbufferSolidViewZ, texelPos, vec4(voxyZ));
        // Unpack
        vec2 uv = unpackUnorm2x16(voxyData.x);
        vec4 colorFace = unpackUnorm4x8(voxyData.y);
        vec3 albedo = colorFace.rgb;
        uint face = uint(colorFace.a * 255.0 + 0.5);
        uint lmMat = voxyData.z;
        vec2 lmCoord = vec2(lmMat & 0xFFu, (lmMat >> 8) & 0xFFu) / 255.0;
        uint matID = (lmMat >> 16) & 0xFFFFu;

        // Simple Normal (Placeholder)
        vec3 N = vec3(0,1,0);
        // Ideally reconstruct using cross product of dFdx/dFdy of ViewPos derived from ViewZ neighbors

        GBufferData gData = gbufferData_init();
        gData.albedo = albedo;
        gData.normal = N;
        gData.geomNormal = N;
        gData.lmCoord = lmCoord;
        gData.materialID = matID;

        uvec4 d1; uvec4 d2; gbufferData1_pack(d1, gData); gbufferData2_pack(d2, gData);
        imageStore(uimg_gbufferSolidData1, texelPos, d1);
        imageStore(uimg_gbufferSolidData2, texelPos, uvec4(d2.r, 0, 0, 0));
    }
}
