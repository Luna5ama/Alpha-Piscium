#version 460 compatibility

#include "_Util.glsl"

uniform sampler2D colortex0;

uniform sampler2D shadowtex0;
uniform sampler2D shadowtex1;
uniform sampler2D shadowcolor0;
uniform sampler2D shadowcolor1;

uniform sampler2D usam_rtwsm_imap2D;
uniform sampler2D usam_rtwsm_imap1D;
uniform sampler2D usam_rtwsm_warpingMap;
uniform sampler2D usam_transmittanceLUT;

varying vec2 texcoord;

const float IMPORTANCE_MUL = uintBitsToFloat(0x4F800000u);

bool inViewPort(vec4 originSize, out vec2 texCoord) {
    vec2 min = originSize.xy;
    vec2 max = originSize.xy + originSize.zw;
    texCoord = (gl_FragCoord.xy - min) / originSize.zw;
    if (all(greaterThanEqual(gl_FragCoord.xy, min)) && all(lessThan(gl_FragCoord.xy, max))) {
        return true;
    }
    return false;
}

layout(location = 0) out vec4 rt_out;

void main() {
    ivec2 intTexCoord = ivec2(gl_FragCoord.xy);

    vec4 color = texelFetch(colortex0, intTexCoord, 0);

    vec2 debugTexCoord;
    #ifdef SETTING_DEBUG_RTWSM
    if (inViewPort(vec4(0, 0, 512, 512), debugTexCoord)) {
        color.rgb = pow(texture(shadowtex0, debugTexCoord).r, 2.0).rrr;
    }
//    if (inViewPort(vec4(512, 0, 512, 512), debugTexCoord)) {
//        color.rgb = pow(texture(shadowtex1, debugTexCoord).r, 2.0).rrr;
//    }
//    if (inViewPort(vec4(1024, 0, 512, 512), debugTexCoord)) {
//        color.rgb = texture(shadowcolor0, debugTexCoord).rgb;
//    }

    if (inViewPort(vec4(0, 512, 512, 512), debugTexCoord)) {
        color.rgb = texture(usam_rtwsm_imap2D, debugTexCoord).rrr * 8.0;
    }

    if (inViewPort(vec4(0, 1024, 512, 16), debugTexCoord)) {
        color.rgb = texture(usam_rtwsm_imap1D, vec2(debugTexCoord.x, 0.25)).rrr;
    }

    if (inViewPort(vec4(512, 512, 16, 512), debugTexCoord)) {
        color.rgb = texture(usam_rtwsm_imap1D, vec2(debugTexCoord.y, 0.75)).rrr;
    }

    if (inViewPort(vec4(0, 1024 + 16, 512, 16), debugTexCoord)) {
        vec2 v = texture(usam_rtwsm_warpingMap, vec2(debugTexCoord.x, 0.5)).rg;
        color.rgb = vec3(max(v.x, 0.0), max(-v.x, 0.0), 0.0);
    }

    if (inViewPort(vec4(512 + 16, 512, 16, 512), debugTexCoord)) {
        vec2 v = texture(usam_rtwsm_warpingMap, vec2(debugTexCoord.y, 0.75)).rg;
        color.rgb = vec3(max(v.x, 0.0), max(-v.x, 0.0), 0.0);
    }
    #endif

    #ifdef SETTING_DEBUG_ATMOSPHERE
    if (inViewPort(vec4(0, 0, 256, 64), debugTexCoord)) {
        color.rgb = pow(texture(usam_transmittanceLUT, debugTexCoord).rgb, vec3(1.0 / SETTING_TONEMAP_OUTPUT_GAMMA));
    }
    #endif

    rt_out = color;
}