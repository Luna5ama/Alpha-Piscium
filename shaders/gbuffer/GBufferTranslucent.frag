#include "../_Util.glsl"

uniform sampler2D gtexture;
uniform sampler2D normals;
uniform sampler2D specular;

uniform usampler2D usam_gbufferData;
uniform sampler2D usam_gbufferViewZ;

in vec3 frag_viewTangent;

in vec4 frag_colorMul; // 8 x 4 = 32 bits
in vec3 frag_viewNormal; // 11 + 11 + 10 = 32 bits
in vec2 frag_texCoord; // 16 x 2 = 32 bits
in vec2 frag_lmCoord; // 8 x 2 = 16 bits
flat in uint frag_materialID; // 16 x 1 = 16 bits

in float frag_viewZ; // 32 bits

ivec2 texelPos = ivec2(gl_FragCoord.xy);

layout(early_fragment_tests) in;

/* RENDERTARGETS:1 */
layout(location = 0) out vec4 rt_temp1;

vec4 processAlbedo() {
    vec4 albedo = frag_colorMul;
    albedo *= texture(gtexture, frag_texCoord);
    #ifdef SETTING_DEBUG_WHITE_WORLD
    return vec4(1.0);
    #else
    return albedo;
    #endif
}

GBufferData processOutput() {
    GBufferData gData;

    float noiseIGN = rand_IGN(gl_FragCoord.xy, frameCounter);

    vec4 normalSample = textureLod(normals, frag_texCoord, 0.0);
    vec4 specularSample = textureLod(specular, frag_texCoord, 0.0);

    gData.materialAO = normalSample.b;
    gData.pbrSpecular = specularSample;

    const float _1o255 = 1.0 / 255.0;
    float emissiveS = linearStep(1.0, _1o255, gData.pbrSpecular.a);
    emissiveS *= step(_1o255, gData.pbrSpecular.a);

    gData.pbrSpecular.a = emissiveS;

    #ifndef SETTING_NORMAL_MAPPING
    gData.normal = frag_viewNormal;
    #else
    vec3 bitangent = cross(frag_viewTangent.xyz, frag_viewNormal);
    mat3 tbn = mat3(frag_viewTangent.xyz, bitangent, frag_viewNormal);
    vec3 tagentNormal;
    tagentNormal.xy = normalSample.rg * 2.0 - 1.0;
    tagentNormal.z = sqrt(saturate(1.0 - dot(tagentNormal.xy, tagentNormal.xy)));
    vec3 mappedNormal = normalize(tbn * tagentNormal);
    gData.normal = normalize(mix(frag_viewNormal, mappedNormal, SETTING_NORMAL_MAPPING_STRENGTH));
    #endif

    gData.normal = dither(gData.normal, noiseIGN, 1023.0);

    gData.lmCoord = frag_lmCoord;
    gData.materialID = frag_materialID;

    gData.lmCoord = dither(gData.lmCoord, noiseIGN, 255.0);

    return gData;
}

void main() {
    vec4 albedo = processAlbedo();
    float viewZ = frag_viewZ;
    GBufferData gData = processOutput();

    rt_temp1 = albedo;
}