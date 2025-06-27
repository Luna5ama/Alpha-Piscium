
#include "/rtwsm/RTWSM.glsl"


layout(r32i) uniform iimage2D uimg_rtwsm_imap;

out uint vert_survived;
out vec2 vert_unwarpedTexCoord;
out vec2 vert_texcoord;
out vec4 vert_color;
out float vert_viewZ;

void main() {
    vert_texcoord = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
    vert_color = gl_Color;

    vec4 shadowClipPos = ftransform();
    vec4 shadowViewPos = global_shadowRotationMatrix * shadowProjectionInverse * shadowClipPos;
    shadowClipPos = global_shadowProjPrev * shadowViewPos;
    gl_Position = shadowClipPos;
    vert_viewZ = -gl_Position.w;
    gl_Position /= gl_Position.w;

    vec2 vPosTS = gl_Position.xy * 0.5 + 0.5;
    vert_unwarpedTexCoord = vPosTS;
    vPosTS = rtwsm_warpTexCoord(usam_rtwsm_imap, vPosTS);

    gl_Position.xy = vPosTS * 2.0 - 1.0;

    shadowViewPos /= shadowViewPos.w;
    vec4 scenePos = shadowModelViewInverse * shadowViewPos;
    vec4 camViewPos = gbufferModelView * scenePos;
    camViewPos /= camViewPos.w;

    uint survived = uint(all(lessThan(abs(shadowClipPos.xy), shadowClipPos.ww)));
    vert_survived = survived;

    survived &= uint((gl_VertexID & 3) == 0);

    #ifdef SETTING_RTWSM_F
    if (bool(survived)){
        vec2 shadowNdcPos = shadowClipPos.xy / shadowClipPos.w;
        vec2 shadowScreenPos = shadowNdcPos * 0.5 + 0.5;
        ivec2 importanceTexelPos = ivec2(shadowScreenPos * vec2(SETTING_RTWSM_IMAP_SIZE));

        float importance = SETTING_RTWSM_F_BASE;

        // Distance function
        #if SETTING_RTWSM_F_D > 0.0
        importance *= (SETTING_RTWSM_F_D) / (SETTING_RTWSM_F_D + dot(camViewPos, camViewPos));
        #endif

        importance = max(importance, uval_rtwsmMin.x);

        imageAtomicMax(uimg_rtwsm_imap, importanceTexelPos, floatBitsToInt(importance));
    }
    #endif
}