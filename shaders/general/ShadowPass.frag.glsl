#include "/util/Colors.glsl"
#include "/util/Rand.glsl"
#include "/rtwsm/RTWSM.glsl"

uniform sampler2D usam_rtwsm_imap;

uniform sampler2D gtexture;

in vec2 frag_unwarpedTexCoord;
in vec2 frag_texcoord;
in vec4 frag_color;
in float frag_viewZ;
flat in vec2 frag_texcoordMin;
flat in vec2 frag_texcoordMax;

#ifdef SHADOW_PASS_TRANSLUCENT
layout(location = 0) out vec4 rt_out;
#endif

void main() {
    vec2 fragUV = gl_FragCoord.xy * SHADOW_MAP_SIZE.y;
    vec2 warppedUV = rtwsm_warpTexCoord(usam_rtwsm_imap, frag_unwarpedTexCoord);
    vec2 pixelDiff = (fragUV - warppedUV) * SHADOW_MAP_SIZE.x;

    vec2 offsetTexCoord = frag_texcoord + pixelDiff.x * dFdx(frag_texcoord) + pixelDiff.y * dFdy(frag_texcoord);
    offsetTexCoord = clamp(offsetTexCoord, frag_texcoordMin, frag_texcoordMax);
    vec4 color = texture(gtexture, offsetTexCoord) * frag_color;

    #ifdef SHADOW_PASS_ALPHA_TEST
    if (color.a < alphaTestRef) {
        discard;
    }
    #endif

    #ifdef SHADOW_PASS_TRANSLUCENT
    color.rgb = colors_srgbToLinear(color.rgb);
    vec2 randCoord = gl_FragCoord.xy;
    randCoord.y += -frag_viewZ;
    float randAlpha = rand_IGN(uvec2(randCoord), frameCounter);
    if (color.a < randAlpha) {
        discard;
    }
    rt_out = mix(color, vec4(1.0, 1.0, 1.0, 0.0), float(color.a == 1.0));
    #endif
}