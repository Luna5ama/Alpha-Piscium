#include "/util/Colors.glsl"
#include "/util/Rand.glsl"
#include "/techniques/rtwsm/RTWSM.glsl"


uniform sampler2D gtexture;

in vec2 frag_unwarpedTexCoord;
in vec2 frag_texcoord;
in vec4 frag_color;
in vec3 frag_normal;
flat in vec2 frag_texcoordMin;
flat in vec2 frag_texcoordMax;

#ifdef SHADOW_PASS_TRANSLUCENT
/* RENDERTARGETS:0,1,2 */
layout(location = 0) out float rt_depthOffset;
layout(location = 1) out vec3 rt_normal;
layout(location = 2) out vec4 rt_translucentColor;
#else
/* RENDERTARGETS:0,1 */
layout(location = 0) out float rt_depthOffset;
layout(location = 1) out vec3 rt_normal;
#endif

void main() {
    vec2 fragUV = gl_FragCoord.xy * SHADOW_MAP_SIZE.y;
    vec2 warppedUV = rtwsm_warpTexCoord(usam_rtwsm_imap, frag_unwarpedTexCoord);
    vec2 pixelDiff = (fragUV - warppedUV) * SHADOW_MAP_SIZE.x;

    vec2 offsetTexCoord = frag_texcoord + pixelDiff.x * dFdx(frag_texcoord) + pixelDiff.y * dFdy(frag_texcoord);
    offsetTexCoord = clamp(offsetTexCoord, frag_texcoordMin, frag_texcoordMax);
    vec4 color = textureLod(gtexture, offsetTexCoord, 0.0) * frag_color;
    float depthFixOffset = pixelDiff.x * dFdx(gl_FragCoord.z) + pixelDiff.y * dFdy(gl_FragCoord.z);
    rt_depthOffset = depthFixOffset;
    rt_normal = normalize(frag_normal);

    #ifdef SHADOW_PASS_ALPHA_TEST
    if (color.a < alphaTestRef) {
        discard;
    }
    #endif

    #ifdef SHADOW_PASS_TRANSLUCENT
    color.rgb = colors_sRGB_decodeGamma(color.rgb);
    vec2 randCoord = gl_FragCoord.xy;
    randCoord.y += abs(rtwsm_linearDepth(gl_FragCoord.z));
    float randAlpha = rand_IGN(uvec2(randCoord), frameCounter);
    if (color.a < randAlpha) {
        discard;
    }
    rt_translucentColor = mix(color, vec4(1.0, 1.0, 1.0, 0.0), float(color.a == 1.0));
    #endif
}