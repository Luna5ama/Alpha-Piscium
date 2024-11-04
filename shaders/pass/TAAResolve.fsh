#include "../_Util.glsl"

uniform sampler2D colortex0;
uniform sampler2D usam_taaLast;

in vec2 frag_texCoord;

/* RENDERTARGETS:0,15 */
layout(location = 0) out vec4 rt_out;
layout(location = 1) out vec4 rt_taaLast;

void main() {
    ivec2 intTexCoord = ivec2(gl_FragCoord.xy);

    vec2 unJitteredTexCoord = frag_texCoord + ssbo_globalData.taaJitter * viewResolution.zw;
    vec4 currColor = texture(colortex0, unJitteredTexCoord);
    vec4 near1Min = currColor;
    vec4 near1Max = currColor;

    {
        #define SAMPLE_NEAR1(offset) \
            nearColor = textureOffset(colortex0, unJitteredTexCoord, offset); \
            near1Min = min(near1Min, nearColor); \
            near1Max = max(near1Max, nearColor);

        vec4 nearColor;
        SAMPLE_NEAR1(ivec2(-1, -1));
        SAMPLE_NEAR1(ivec2(-1, 0));
        SAMPLE_NEAR1(ivec2(-1, 1));
        SAMPLE_NEAR1(ivec2(0, -1));
        SAMPLE_NEAR1(ivec2(0, 1));
        SAMPLE_NEAR1(ivec2(1, -1));
        SAMPLE_NEAR1(ivec2(1, 0));
        SAMPLE_NEAR1(ivec2(1, 1));
    }

    vec4 lastColor = texture(usam_taaLast, frag_texCoord, 0);
    lastColor = clamp(lastColor, near1Min, near1Max);

    vec4 finalColor = mix(currColor, lastColor, 0.9);

    rt_out = finalColor;
    rt_taaLast = finalColor;
}