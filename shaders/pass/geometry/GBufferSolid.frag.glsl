#include "/util/Colors.glsl"
#include "/util/Colors2.glsl"
#include "/util/Dither.glsl"
#include "/util/Math.glsl"
#include "/util/Rand.glsl"
#include "/util/GBufferData.glsl"

#if defined(GBUFFER_PASS_STEEP_PARALLAX) && defined(SETTING_NORMAL_MAPPING) && defined(SETTING_STEEP_PARALLAX)
#include "/techniques/parallax/Trace.glsl"
#endif

uniform sampler2D gtexture;
uniform sampler2D normals;
uniform sampler2D specular;

#ifdef SETTING_TBN_PACKING
flat in uint frag_worldTN;
#else
in vec3 frag_worldTangent;
in vec3 frag_worldNormal;// 11 + 11 + 10 = 32 bits
#endif

in vec3 frag_colorMul;// 8 x 4 = 32 bits
in vec2 frag_texCoord;// 16 x 2 = 32 bits
in vec2 frag_lmCoord;// 8 x 2 = 16 bits
flat in uint frag_materialID;// 16 x 1 = 16 bits
flat in float frag_emissiveOverride;

#if defined(GBUFFER_PASS_STEEP_PARALLAX) && defined(SETTING_NORMAL_MAPPING) && defined(SETTING_STEEP_PARALLAX)
flat in vec4 frag_spriteBounds;
#endif

#ifndef GBUFFER_PASS_ALPHA_TEST
layout(early_fragment_tests) in;
#endif

#if defined(GBUFFER_PASS_NO_LIGHTING)
/* RENDERTARGETS:6,10 */
layout(location = 0) out vec4 rt_color;
layout(location = 1) out float rt_gbufferSolidViewZ;
#elif defined(GBUFFER_PASS_ARMOR_GLINT)
/* RENDERTARGETS:4 */
layout(location = 0) out vec4 rt_glintColor;
#else
/* RENDERTARGETS:8,9,10 */
layout(location = 0) out uvec4 rt_gbufferSolidData1;
layout(location = 1) out uvec4 rt_gbufferSolidData2;
layout(location = 2) out float rt_gbufferSolidViewZ;
#endif

#ifdef SETTING_SCREENSHOT_MODE
vec2 dUVdx = vec2(0.0);
vec2 dUVdy = vec2(0.0);
#else
vec2 dUVdx = dFdx(frag_texCoord);
vec2 dUVdy = dFdy(frag_texCoord);
#endif

ivec2 texelPos = ivec2(gl_FragCoord.xy);
float ditherNoise = rand_stbnVec1(rand_newStbnPos(texelPos, 4u), frameCounter);

float frag_viewZ = -rcp(gl_FragCoord.w);
vec4 albedo;
float viewZ;
vec2 materialTexCoord = frag_texCoord;

float bitangentSignF;
vec3 geomViewNormal;
vec3 geomViewTangent;
vec3 geomViewBitangent;

#if defined(GBUFFER_PASS_STEEP_PARALLAX) && defined(SETTING_NORMAL_MAPPING) && defined(SETTING_STEEP_PARALLAX)
float displacedViewZ;
vec3 parallaxSurfaceNormal = vec3(0.0, 0.0, 1.0);
#endif

GBufferData gData = gbufferData_init();

void processAlbedo() {
    albedo = vec4(frag_colorMul, 1.0);

    #ifdef GBUFFER_PASS_TEXTURED
    float alphaTestBias = 1.0 - global_taaResetFactor.y * 0.75;
    vec4 sample1 = textureGrad(gtexture, materialTexCoord, dUVdx * alphaTestBias, dUVdy * alphaTestBias);
    vec4 sample2 = textureGrad(gtexture, materialTexCoord, dUVdx * 0.25, dUVdy * 0.25);
    albedo *= vec4(sample2.rgb, sample1.a);
    #endif

    #ifdef GBUFFER_PASS_ENTITY
    albedo.rgb = mix(albedo.rgb, entityColor.rgb, entityColor.a);
    #endif

    #ifdef GBUFFER_PASS_ALPHA_TEST
    float alphaTestThreshold = 0.05;
    #ifndef SETTING_SCREENSHOT_MODE
    float alphaLod = textureQueryLod(gtexture, materialTexCoord).y;
    alphaTestThreshold += min(pow(rand_stbnVec1(texelPos, 0), alphaLod * 2.0 + 1.0), 0.9) * saturate(alphaLod);
    #endif
    if (albedo.a < alphaTestThreshold) {
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
    #elif defined(GBUFFER_PASS_STEEP_PARALLAX) && defined(SETTING_NORMAL_MAPPING) && defined(SETTING_STEEP_PARALLAX)
    #ifdef SETTING_STEEP_PARALLAX_WRITE_VIEWZ
    viewZ = displacedViewZ;
    #else
    viewZ = frag_viewZ;
    #endif
    #else
    viewZ = frag_viewZ;
    #endif
}

void processGeometryBasis() {
    bitangentSignF = float(bitfieldExtract(frag_materialID, 30, 1)) * 2.0 - 1.0;

    vec3 geomWorldNormal;
    vec3 geomWorldTangent;
    #ifdef SETTING_TBN_PACKING
    nzpacking_unpackNormalOct16(frag_worldTN, geomWorldNormal, geomWorldTangent);
    #else
    geomWorldNormal = frag_worldNormal;
    geomWorldTangent = frag_worldTangent;
    #endif

    geomViewNormal = coords_dir_worldToView(geomWorldNormal);
    geomViewTangent = coords_dir_worldToView(geomWorldTangent);
    geomViewBitangent = normalize(cross(geomViewTangent, geomViewNormal) * bitangentSignF);
}

#if defined(GBUFFER_PASS_STEEP_PARALLAX) && defined(SETTING_NORMAL_MAPPING) && defined(SETTING_STEEP_PARALLAX)
void processParallax() {
    processGeometryBasis();

    vec2 screenPos = gl_FragCoord.xy * uval_mainImageSizeRcp - uval_taaJitterUV;
    vec3 viewPos = coords_toViewCoord(screenPos, frag_viewZ, global_camProjInverse);
    vec3 viewRay = -viewPos;
    displacedViewZ = frag_viewZ;

    vec3 viewRayTS = transpose(mat3(geomViewTangent, geomViewBitangent, geomViewNormal)) * viewRay;
    if (viewRayTS.z <= 0.0 || viewRayTS.z * viewRayTS.z <= dot(viewRay, viewRay) * 1e-8) {
        return;
    }

    vec2 atlasSize = vec2(textureSize(usam_blocksNormal, 0));
    vec2 spriteExtentTexels = max((frag_spriteBounds.zw - frag_spriteBounds.xy) * atlasSize, vec2(1.0));
    float parallaxScale = SETTING_STEEP_PARALLAX_DEPTH / viewRayTS.z;
    vec2 rayDeltaTexels = -viewRayTS.xy * parallaxScale * spriteExtentTexels;
    vec2 hitTexCoord;
    float hitT;
    if (traceParallax(materialTexCoord, frag_spriteBounds, rayDeltaTexels, hitTexCoord, hitT, parallaxSurfaceNormal)) {
        materialTexCoord = hitTexCoord;
        displacedViewZ = frag_viewZ - viewRay.z * parallaxScale * hitT;
    }
}
#endif

void processData2() {
    gData.albedo = albedo.rgb;
    #ifdef GBUFFER_PASS_HAND
    gData.isHand = true;
    #else
    gData.isHand = false;
    #endif
}

void processData1() {
    #if !defined(GBUFFER_PASS_STEEP_PARALLAX) || !defined(SETTING_NORMAL_MAPPING) || !defined(SETTING_STEEP_PARALLAX)
    processGeometryBasis();
    #endif

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
    vec4 normalSample = textureGrad(normals, materialTexCoord, dUVdx, dUVdy);
    vec4 specularSample = textureGrad(specular, materialTexCoord, dUVdx, dUVdy);

    gData.pbrSpecular = specularSample;
    gData.lmCoord.y *= normalSample.b;
    gData.materialID = frag_materialID & 0xFFFFu;

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
    #if defined(GBUFFER_PASS_STEEP_PARALLAX) && defined(SETTING_STEEP_PARALLAX) && defined(SETTING_STEEP_PARALLAX_NORMAL)
    #if SETTING_PARALLAX_MODE == 0
    if (parallaxSurfaceNormal.x != 0.0) {
        tangentNormal = vec3(parallaxSurfaceNormal.x * tangentNormal.z, tangentNormal.y, -parallaxSurfaceNormal.x * tangentNormal.x);
    } else if (parallaxSurfaceNormal.y != 0.0) {
        tangentNormal = vec3(tangentNormal.y, parallaxSurfaceNormal.y * tangentNormal.z, parallaxSurfaceNormal.y * tangentNormal.x);
    }
    #else
    vec3 surfaceNormal = normalize(parallaxSurfaceNormal);
    vec3 surfaceTangent = normalize(vec3(1.0, 0.0, -parallaxSurfaceNormal.x));
    vec3 surfaceBitangent = cross(surfaceNormal, surfaceTangent);
    tangentNormal = mat3(surfaceTangent, surfaceBitangent, surfaceNormal) * tangentNormal;
    #endif
    #endif
    gData.normal = normalize(tbn * tangentNormal);
    #endif

    #endif

    #ifdef GBUFFER_PASS_DH
    gData.materialID = 0u;
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
    #ifdef DISTANT_HORIZONS
    #ifndef GBUFFER_PASS_DH
    vec2 screenPos = gl_FragCoord.xy * uval_mainImageSizeRcp;
    vec3 distantViewPos = coords_toViewCoord(screenPos, frag_viewZ, global_camProjInverse);
    float edgeFactor = linearStep(min(far * 0.75, far - 24.0), far, length(distantViewPos));
    if (ditherNoise < edgeFactor) {
        discard;
        return;
    }
    #endif
    #endif

    #if defined(GBUFFER_PASS_STEEP_PARALLAX) && defined(SETTING_NORMAL_MAPPING) && defined(SETTING_STEEP_PARALLAX)
    processParallax();
    #endif
    processAlbedo();
    processViewZ();

    #if defined(GBUFFER_PASS_NO_LIGHTING)
    rt_color = albedo;
    rt_gbufferSolidViewZ = viewZ;
    #elif defined(GBUFFER_PASS_ARMOR_GLINT)
    rt_glintColor = dither_u8(albedo, ditherNoise);
    #else
    processData1();
    processData2();

    gbufferData1_pack(rt_gbufferSolidData1, gData);
    gbufferData2_pack(rt_gbufferSolidData2, gData);
    rt_gbufferSolidViewZ = viewZ;
    #endif
}
