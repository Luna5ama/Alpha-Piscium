#include "../_Util.glsl"

uniform sampler2D gtexture;
uniform sampler2D normals;
uniform sampler2D specular;

uniform usampler2D usam_gbuffer;
uniform sampler2D usam_viewZ;

in vec3 frag_viewTangent;

in vec4 frag_colorMul; // 8 x 4 = 32 bits
in vec3 frag_viewNormal; // 11 + 11 + 10 = 32 bits
in vec2 frag_texCoord; // 16 x 2 = 32 bits
in vec2 frag_lmCoord; // 8 x 2 = 16 bits
flat in uint frag_materialID; // 16 x 1 = 16 bits

in float frag_viewZ; // 32 bits

/* RENDERTARGETS:8,9 */
layout(location = 0) out uvec4 rt_gbuffer;
layout(location = 1) out float rt_viewZ;

float hash( vec2 x ) {
    return fract( 1.0e4 * sin( 17.0*x.x + 0.1*x.y ) *( 0.1 + abs( sin( 13.0*x.y + x.x ))));
}
float hash3D( vec3 x ) {
    return hash( vec2( hash( x.xy ), x.z ) );
}


vec4 processAlbedo() {
    vec4 albedo = frag_colorMul;

    #ifdef GBUFFER_PASS_TEXTURED
    albedo *= texture(gtexture, frag_texCoord);
    #endif

    #ifdef GBUFFER_PASS_ENTITY_COLOR
    albedo.rgb = mix(albedo.rgb, entityColor.rgb, entityColor.a);
    #endif

    #ifdef GBUFFER_PASS_ALPHA_TEST
    if (albedo.a < 0.1) {
        discard;
    }
    #endif

    #ifdef GBUFFER_PASS_TRANLUCENT
    uint r2Index = rand_hash11(floatBitsToUint(frag_viewZ)) & 31u;
    vec2 randR2 = rand_r2Seq2(r2Index);
    float randAlpha = rand_IGN(gl_FragCoord.xy + randR2, frameCounter);

    if (albedo.a < randAlpha) {
        discard;
    }
    #endif

    #ifdef SETTING_DEBUG_WHITE_WORLD
    return vec4(1.0);
    #else
    return albedo;
    #endif
}

#ifdef GBUFFER_PASS_ARMOR_GLINT
void processOutput(out GBufferData gData, out float viewZ) {
    ivec2 pixelCoord = ivec2(gl_FragCoord.xy);
    float noiseIGN = rand_IGN(gl_FragCoord.xy, frameCounter);
    vec4 albedo = processAlbedo();

    GBufferData gDataPrev;
    gbuffer_unpack(texelFetch(usam_gbuffer, pixelCoord, 0), gDataPrev);

    gData.albedo = gDataPrev.albedo + albedo.rgb * albedo.rgb;
    gData.materialAO = gDataPrev.materialAO;
    gData.pbrSpecular = gDataPrev.pbrSpecular;

    gData.normal = gDataPrev.normal;
    gData.lmCoord = gDataPrev.lmCoord;
    gData.materialID = gDataPrev.materialID;

    float glintEmissive = colors_srgbLuma(albedo.rgb);
    glintEmissive *= 0.1;
    glintEmissive = dither(glintEmissive, noiseIGN, 64.0);
    gData.pbrSpecular.a -= glintEmissive;

    viewZ = texelFetch(usam_viewZ, pixelCoord, 0).r;
}
#else
void processOutput(out GBufferData gData, out float viewZ) {
    ivec2 pixelCoord = ivec2(gl_FragCoord.xy);
    float noiseIGN = rand_IGN(gl_FragCoord.xy, frameCounter);
    vec4 albedo = processAlbedo();

    gData.albedo = albedo.rgb;

    #if defined(GBUFFER_PASS_TEXTURED)
    vec4 normalSample = textureLod(normals, frag_texCoord, 0.0);
    vec4 specularSample = textureLod(specular, frag_texCoord, 0.0);

    gData.materialAO = normalSample.b;
    gData.pbrSpecular = specularSample;

    #ifndef SETTING_NORMAL_MAPPING
    gData.normal = frag_viewNormal;
    #else
    vec3 bitangent = cross(frag_viewTangent, frag_viewNormal);
    mat3 tbn = mat3(frag_viewTangent, bitangent, frag_viewNormal);
    vec3 tagentNormal;
    tagentNormal.xy = normalSample.rg * 2.0 - 1.0;
    tagentNormal.z = sqrt(saturate(1.0 - dot(tagentNormal.xy, tagentNormal.xy)));
    vec3 mappedNormal = normalize(tbn * tagentNormal);
    gData.normal = normalize(mix(frag_viewNormal, mappedNormal, SETTING_NORMAL_MAPPING_STRENGTH));
    #endif

    gData.normal = dither(gData.normal, noiseIGN, 1023.0);

    #else
    // TODO: hardcoded PBR
    gData.materialAO = 1.0;
    gData.pbrSpecular = vec4(0.0, 1.0, 0.0, 0.0);

    gData.normal = vec3(1.0);
    #endif

    gData.lmCoord = frag_lmCoord;
    gData.materialID = frag_materialID;

    gData.lmCoord = dither(gData.lmCoord, noiseIGN, 255.0);

    #ifdef GBUFFER_PASS_VIEWZ_OVERRIDE
    viewZ = GBUFFER_PASS_VIEWZ_OVERRIDE;
    #else
    viewZ = frag_viewZ;
    #endif

    #ifdef GBUFFER_PASS_TRANLUCENT
    gData.isTranslucent = true;
    #else
    gData.isTranslucent = false;
    #endif
}
#endif

void main() {
    GBufferData gData;
    processOutput(gData, rt_viewZ);
    gbuffer_pack(rt_gbuffer, gData);
}