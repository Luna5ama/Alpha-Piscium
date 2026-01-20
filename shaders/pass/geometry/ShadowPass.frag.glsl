#include "/util/Colors.glsl"
#include "/util/Rand.glsl"
#include "/util/Translucent.glsl"
#include "/techniques/rtwsm/RTWSM.glsl"

uniform sampler2D gtexture;

in vec2 frag_unwarpedTexCoord;
in vec2 frag_texcoord;
in vec4 frag_color;
in vec3 frag_normal;
in vec3 frag_scenePos;
flat in uint frag_materialID;
flat in vec2 frag_texcoordMin;
flat in vec2 frag_texcoordMax;

#ifdef SHADOW_PASS_TRANSLUCENT
/* RENDERTARGETS:0,1,2,3,4,5 */
layout(location = 0) out float rt_depthOffset;
layout(location = 1) out vec3 rt_normal;
layout(location = 2) out vec4 rt_translucentColor;
layout(location = 3) out vec2 rt_unwarpedUV;
layout(location = 4) out float rt_pixelArea;
layout(location = 5) out vec4 rt_waterMask;
#else
/* RENDERTARGETS:0,1 */
layout(location = 0) out float rt_depthOffset;
layout(location = 1) out vec3 rt_normal;
#endif

void main() {
    vec2 fragUV = gl_FragCoord.xy * SHADOW_MAP_SIZE.y;
    vec2 texelSize;
    vec2 warppedUV = rtwsm_warpTexCoordTexelSize(frag_unwarpedTexCoord, texelSize);
    vec2 pixelDiff = (fragUV - warppedUV) * SHADOW_MAP_SIZE.x;

    vec2 offsetTexCoord = frag_texcoord + pixelDiff.x * dFdx(frag_texcoord) + pixelDiff.y * dFdy(frag_texcoord);
    offsetTexCoord = clamp(offsetTexCoord, frag_texcoordMin, frag_texcoordMax);
    vec4 inputAlbedo = textureLod(gtexture, offsetTexCoord, 0.0) * frag_color;
    float depthFixOffset = pixelDiff.x * dFdx(gl_FragCoord.z) + pixelDiff.y * dFdy(gl_FragCoord.z);
    depthFixOffset = max(depthFixOffset, 0.0);

    float lightDot = abs(dot(frag_normal, uval_shadowLightDirWorld));
    lightDot = max(lightDot, 0.01);

    float depthBiasFactor = sqrt(1.0 - pow2(lightDot)) / lightDot; // tan(acos(lightDot))
    depthBiasFactor = depthBiasFactor * 2.0 + 1.0;
    float depthBias = SHADOW_MAP_SIZE.y * depthBiasFactor / min2(texelSize);
    depthFixOffset += rtwsm_linearDepthOffsetInverse(depthBias);

    rt_normal = normalize(frag_normal);

    #ifdef SHADOW_PASS_ALPHA_TEST
    if (inputAlbedo.a < alphaTestRef) {
        discard;
    }
    #endif

    #ifdef SHADOW_PASS_TRANSLUCENT
    bool isWater = frag_materialID == 3u;

    rt_pixelArea = length(dFdx(frag_scenePos)) * length(dFdy(frag_scenePos));
    rt_unwarpedUV = frag_unwarpedTexCoord;
    float waterMask = float(isWater);
    rt_waterMask = vec4(waterMask);

    float alpha = inputAlbedo.a;
    vec3 materialColor = colors2_material_toWorkSpace(inputAlbedo.rgb);
    vec3 transmittance = translucent_albedoToTransmittance(materialColor, alpha, isWater);
    rt_translucentColor = isWater ? vec4(1.0) : vec4(transmittance, 0.0);

    depthFixOffset = -depthFixOffset;
    #endif

    rt_depthOffset = depthFixOffset;
}