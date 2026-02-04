#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_ballot : enable

#include "/util/Coords.glsl"
#include "/util/Colors.glsl"
#include "/util/Rand.glsl"
#include "/util/Translucent.glsl"
#include "/techniques/rtwsm/RTWSM.glsl"

uniform sampler2D gtexture;
uniform sampler2D specular;

#if defined(SHADOW_PASS_ALPHA_TEST) || defined(SHADOW_PASS_TRANSLUCENT)
in vec2 frag_texcoord;
#if defined(SHADOW_PASS_TRANSLUCENT)
in vec4 frag_color;
#endif
#endif
in vec2 frag_screenPos;
flat in uint frag_worldNormalMaterialID;

#if defined(SHADOW_PASS_ALPHA_TEST) || defined(SHADOW_PASS_TRANSLUCENT)
flat in uvec2 frag_texcoordMinMax;
#endif

#ifdef SHADOW_PASS_TRANSLUCENT
/* RENDERTARGETS:0,2,3,4,5 */
layout(location = 0) out float rt_depthOffset;
layout(location = 1) out vec4 rt_translucentColor;
layout(location = 2) out vec2 rt_unwarpedUV;
layout(location = 3) out float rt_pixelArea;
layout(location = 4) out vec4 rt_waterMask;
#else
/* RENDERTARGETS:0,1 */
layout(location = 0) out float rt_depthOffset;
layout(location = 1) out vec4 rt_specular;
#endif

void main() {
    uint materialID = bitfieldExtract(frag_worldNormalMaterialID, 16, 16);
    vec2 worldNormalOct = unpackSnorm4x8(frag_worldNormalMaterialID).xy;
    vec3 worldNormal = coords_octDecode11(worldNormalOct);

    vec3 shadowScreenPos = vec3(frag_screenPos, gl_FragCoord.z);

    vec2 fragUV = gl_FragCoord.xy * SHADOW_MAP_SIZE.y;
    vec2 texelSize;
    vec2 warppedUV = rtwsm_warpTexCoordTexelSize(shadowScreenPos.xy, texelSize);
    vec2 pixelDiff = (fragUV - warppedUV) * SHADOW_MAP_SIZE.x;

    float dZdx = dFdxFine(gl_FragCoord.z);
    float dZdy = dFdyFine(gl_FragCoord.z);
    float depthFixOffset = abs(pixelDiff.x * dZdx) + abs(pixelDiff.y * dZdy);

    float lightDot = abs(dot(worldNormal, uval_shadowLightDirWorld));
    lightDot = max(lightDot, 0.05);

    float depthBiasSlopeFactor = sqrt(1.0 - pow2(lightDot)) / lightDot; // tan(acos(lightDot))
    depthBiasSlopeFactor = log2(depthBiasSlopeFactor + 1.0) + 0.01;
    float depthBiasTexelSizeFactor = SHADOW_MAP_SIZE.y * (1.0 / 256.0) / min2(texelSize);
    float depthBias = depthBiasSlopeFactor * depthBiasTexelSizeFactor;
    depthFixOffset += depthBias;

    if (materialID == 4u) {
        depthFixOffset = 0.0;
    }

    vec4 inputAlbedo = vec4(1.0);

    #if defined(SHADOW_PASS_ALPHA_TEST) || defined(SHADOW_PASS_TRANSLUCENT)
    vec2 offsetTexCoord = frag_texcoord + pixelDiff.x * dFdx(frag_texcoord) + pixelDiff.y * dFdy(frag_texcoord);
    vec2 texcoordMin = unpackSnorm2x16(frag_texcoordMinMax.x);
    vec2 texcoordMax = unpackSnorm2x16(frag_texcoordMinMax.y);
    offsetTexCoord = clamp(offsetTexCoord, texcoordMin, texcoordMax);
    inputAlbedo = texture(gtexture, offsetTexCoord);
    #if defined(SHADOW_PASS_TRANSLUCENT)
    inputAlbedo.rgb *= frag_color.rgb;
    #endif
    #endif

    #ifdef SHADOW_PASS_ALPHA_TEST
    if (inputAlbedo.a < alphaTestRef) {
        discard;
    }
    #endif

    #ifdef SHADOW_PASS_TRANSLUCENT
    bool isWater = materialID == 3u;

    vec4 matv1 = vec4(0.0);
    vec4 matv2 = vec4(0.0);
    vec4 matv3 = vec4(0.0);
    vec4 matv4 = vec4(0.0);
    if (subgroupElect()) {
        mat4 tempMat = global_shadowNDCToScene;
        matv1 = tempMat[0];
        matv2 = tempMat[1];
        matv3 = tempMat[2];
        matv4 = tempMat[3];
    }
    matv1 = subgroupBroadcastFirst(matv1);
    matv2 = subgroupBroadcastFirst(matv2);
    matv3 = subgroupBroadcastFirst(matv3);
    matv4 = subgroupBroadcastFirst(matv4);
    mat4 matv = mat4(matv1, matv2, matv3, matv4);

    vec4 shadowNDCPos = vec4(shadowScreenPos * 2.0 - 1.0, 1.0);
    vec4 scenePos = matv * shadowNDCPos;
    rt_pixelArea = length(dFdx(scenePos)) * length(dFdy(scenePos));
    rt_unwarpedUV = shadowScreenPos.xy;
    float waterMask = float(isWater);
    rt_waterMask = vec4(waterMask);

    float alpha = inputAlbedo.a;
    vec3 materialColor = colors2_material_toWorkSpace(inputAlbedo.rgb);
    rt_translucentColor = translucent_albedoToTransmittance(materialColor, alpha, materialID);

    depthFixOffset = -depthFixOffset;
    #else
    rt_specular.b = 0.0;
    #endif

    rt_depthOffset = depthFixOffset;
}