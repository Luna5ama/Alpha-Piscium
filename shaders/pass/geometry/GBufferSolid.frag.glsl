#include "/util/Colors.glsl"
#include "/util/Colors2.glsl"
#include "/util/Dither.glsl"
#include "/util/Math.glsl"
#include "/util/Rand.glsl"
#include "/util/GBufferData.glsl"

uniform sampler2D gtexture;
uniform sampler2D normals;
uniform sampler2D specular;

in vec4 frag_viewTangent;

in vec4 frag_colorMul;// 8 x 4 = 32 bits
in vec3 frag_viewNormal;// 11 + 11 + 10 = 32 bits
in vec2 frag_texCoord;// 16 x 2 = 32 bits
in vec2 frag_lmCoord;// 8 x 2 = 16 bits
flat in uint frag_materialID;// 16 x 1 = 16 bits
flat in float frag_emissiveOverride;

in float frag_viewZ;// 32 bits

#ifndef GBUFFER_PASS_ALPHA_TEST
layout(early_fragment_tests) in;
#endif

#if defined(GBUFFER_PASS_NO_LIGHTING)
/* RENDERTARGETS:6,10 */
layout(location = 0) out vec4 rt_color;
layout(location = 1) out float rt_gbufferViewZ;
#elif defined(GBUFFER_PASS_ARMOR_GLINT)
/* RENDERTARGETS:4 */
layout(location = 0) out vec4 rt_glintColor;
#else
/* RENDERTARGETS:8,9,10 */
layout(location = 0) out uvec4 rt_gbufferData1;
layout(location = 1) out uvec4 rt_gbufferData2;
layout(location = 2) out float rt_gbufferViewZ;
#endif

#ifdef SETTING_SCREENSHOT_MODE
vec2 dUVdx = vec2(0.0);
vec2 dUVdy = vec2(0.0);
#else
vec2 dUVdx = dFdx(frag_texCoord);
vec2 dUVdy = dFdy(frag_texCoord);
#endif

ivec2 texelPos = ivec2(gl_FragCoord.xy);
float ditherNoise = rand_stbnVec1(texelPos, frameCounter);

vec4 albedo;
float viewZ;

GBufferData gData = gbufferData_init();

void processAlbedo() {
    albedo = frag_colorMul;

    #ifdef GBUFFER_PASS_TEXTURED
    vec4 sample1 = textureGrad(gtexture, frag_texCoord, dUVdx * 0.5, dUVdy * 0.5);
    vec4 sample2 = textureGrad(gtexture, frag_texCoord, dUVdx * 0.25, dUVdy * 0.25);
    albedo *= vec4(sample2.rgb, sample1.a);
    #endif

    #ifdef GBUFFER_PASS_ENTITY
    albedo.rgb = mix(albedo.rgb, entityColor.rgb, entityColor.a);
    #endif

    #ifdef GBUFFER_PASS_ALPHA_TEST
    if (albedo.a < 0.1) {
        discard;
    }
    #endif

    #ifdef SETTING_DEBUG_WHITE_WORLD
    albedo.rgb = vec3(1.0);
    #endif
}

void processViewZ() {
    #if defined(GBUFFER_PASS_VIEWZ_OVERRIDE)
    viewZ = GBUFFER_PASS_VIEWZ_OVERRIDE;
    #else
    viewZ = frag_viewZ;
    #endif
}

void processData2() {
    gData.albedo = albedo.rgb;
    #ifdef GBUFFER_PASS_HAND
    gData.isHand = true;
    #else
    gData.isHand = false;
    #endif
}

void processData1() {
    float bitangentSignF = frag_viewTangent.w < 0.0 ? -1.0 : 1.0;
    vec3 geomViewNormal = normalize(frag_viewNormal);
    vec3 geomViewTangent = normalize(frag_viewTangent.xyz);
    vec3 geomViewBitangent = normalize(cross(geomViewTangent, geomViewNormal) * bitangentSignF);

    gData.normal = geomViewNormal;
    gData.geomNormal = geomViewNormal;
    gData.geomTangent = geomViewTangent;
    gData.bitangentSign = int(bitangentSignF);

    gData.pbrSpecular = vec4(0.1, 0.01, 0.0, 0.0);
    #ifdef GBUFFER_PASS_DH
    gData.pbrSpecular.a = frag_emissiveOverride;
    #endif
    gData.lmCoord = frag_lmCoord;
    gData.materialID = 65534u;

    #if defined(GBUFFER_PASS_TEXTURED)
    vec4 normalSample = textureGrad(normals, frag_texCoord, dUVdx, dUVdy);
    vec4 specularSample = textureGrad(specular, frag_texCoord, dUVdx, dUVdy);

    gData.pbrSpecular = specularSample;
    gData.lmCoord.y *= normalSample.b;
    gData.materialID = frag_materialID;

    float emissiveS = specularSample.a;
    emissiveS *= float(specularSample.a < 1.0);

    gData.pbrSpecular.a = emissiveS;

    #ifdef SETTING_NORMAL_MAPPING
    mat3 tbn = mat3(geomViewTangent, geomViewBitangent, geomViewNormal);
    vec3 tangentNormal;
    tangentNormal.xy = normalSample.rg * 2.0 - 1.0;
    tangentNormal.z = sqrt(saturate(1.0 - dot(tangentNormal.xy, tangentNormal.xy)));
    tangentNormal.xy *= exp2(SETTING_NORMAL_MAPPING_STRENGTH);
    tangentNormal = normalize(tangentNormal);
    gData.normal = normalize(tbn * tangentNormal);
    #endif

    #endif

    #ifdef GBUFFER_PASS_DH
    gData.materialID = 65533;
    #endif

    #ifdef GBUFFER_PASS_ENTITY
    gData.pbrSpecular.a *= SETTING_ENTITY_EMISSIVE_STRENGTH;
    #endif

    #ifdef GBUFFER_PASS_PARTICLE
    gData.materialID = 65533u;
    if (SETTING_PARTICLE_EMISSIVE_STRENGTH > 0.0) {
        if (textureQueryLevels(gtexture) == 1) {
            float particleEmissive = pow2(colors2_colorspaces_luma(COLORS2_WORKING_COLORSPACE, colors2_material_toWorkSpace(albedo.rgb)));
            particleEmissive *= SETTING_PARTICLE_EMISSIVE_STRENGTH;
            gData.pbrSpecular.a = saturate(gData.pbrSpecular.a + particleEmissive);
        }
    }
    #endif

    gData.lmCoord = dither_u8(gData.lmCoord, ditherNoise);
}

void main() {
    if ((1.0 - gl_FragCoord.z) > MC_HAND_DEPTH) {
        discard;
        return;
    }
    #ifdef DISTANT_HORIZONS
    #ifndef GBUFFER_PASS_DH
    vec2 screenPos = gl_FragCoord.xy * uval_mainImageSizeRcp;
    vec3 viewPos = coords_toViewCoord(screenPos, frag_viewZ, global_camProjInverse);
    float edgeFactor = linearStep(min(far * 0.75, far - 24.0), far, length(viewPos));
    if (ditherNoise < edgeFactor) {
        discard;
        return;
    }
    #endif
    #endif

    processAlbedo();
    processViewZ();

    #if defined(GBUFFER_PASS_NO_LIGHTING)
    rt_color = albedo;
    rt_gbufferViewZ = viewZ;
    #elif defined(GBUFFER_PASS_ARMOR_GLINT)
    rt_glintColor = dither_u8(albedo, ditherNoise);
    #else
    processData1();
    processData2();

    gbufferData1_pack(rt_gbufferData1, gData);
    gbufferData2_pack(rt_gbufferData2, gData);
    rt_gbufferViewZ = viewZ;
    #endif
}