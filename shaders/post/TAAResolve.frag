#include "../_Util.glsl"

uniform sampler2D usam_main;
uniform usampler2D usam_gbuffer;
uniform sampler2D usam_viewZ;
uniform sampler2D usam_taaLast;

in vec2 frag_texCoord;

/* RENDERTARGETS:0,15 */
layout(location = 0) out vec4 rt_out;
layout(location = 1) out vec4 rt_taaLast;

void main() {
    ivec2 intTexCoord = ivec2(gl_FragCoord.xy);

    GBufferData gData;
    gbuffer_unpack(texelFetch(usam_gbuffer, intTexCoord, 0), gData);

    float viewZ = texelFetch(usam_viewZ, intTexCoord, 0).r;
    vec3 viewCoord = coords_toViewCoord(frag_texCoord, viewZ, gbufferProjectionInverse);
    vec4 worldCoord = gbufferModelViewInverse * vec4(viewCoord, 1.0);
    vec3 cameraDelta = cameraPosition - previousCameraPosition;
    vec4 prevWorldCoord = worldCoord + vec4(cameraDelta, 0.0);
    vec4 prevViewCoord = gbufferPreviousModelView * prevWorldCoord;
    vec4 prevClipCoord = gbufferProjection * prevViewCoord;
    prevClipCoord /= prevClipCoord.w;
    vec2 prevTexCoord = prevClipCoord.xy * 0.5 + 0.5;
    prevTexCoord = mix(prevTexCoord, frag_texCoord, float(gData.materialID == MATERIAL_ID_HAND));

    vec4 currColor = texelFetch(usam_main, intTexCoord, 0);
    currColor = saturate(currColor);
    vec4 near1Min = currColor;
    vec4 near1Max = currColor;

    {
        #define SAMPLE_NEAR1(offset) \
            nearColor = texelFetchOffset(usam_main, intTexCoord, 0, offset); \
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

    vec4 lastColor = texture(usam_taaLast, prevTexCoord, 0);
    lastColor = saturate(lastColor);
    lastColor = clamp(lastColor, near1Min, near1Max);

    vec4 finalColor = mix(currColor, lastColor, 0.9);

    rt_out = finalColor;
    rt_taaLast = finalColor;
}