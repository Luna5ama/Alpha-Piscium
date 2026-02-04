#include "/techniques/rtwsm/RTWSM.glsl"
#include "/util/Fresnel.glsl"

layout(r32i) uniform iimage2D uimg_fr32f;

in vec2 mc_Entity;

#if defined(SHADOW_PASS_ALPHA_TEST) || defined(SHADOW_PASS_TRANSLUCENT)
out vec2 vert_texcoord;
#if defined(SHADOW_PASS_TRANSLUCENT)
out vec4 vert_color;
#endif
#endif

out vec2 vert_screenPos;
out uint vert_worldNormalMaterialID;

out uint vert_survived;

void main() {
    #if defined(SHADOW_PASS_ALPHA_TEST) || defined(SHADOW_PASS_TRANSLUCENT)
    vert_texcoord = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
    #if defined(SHADOW_PASS_TRANSLUCENT)
    vert_color = gl_Color;
    #endif
    #endif

    vec3 worldNormal = normalize(mat3(shadowModelViewInverse) * (gl_NormalMatrix * gl_Normal.xyz));
    vec2 worldNormalOct = coords_octEncode11(worldNormal);
    uint materialID = uint(int(mc_Entity.x));
    vert_worldNormalMaterialID = packSnorm4x8(vec4(worldNormalOct, 0.0, 0.0));
    vert_worldNormalMaterialID = bitfieldInsert(vert_worldNormalMaterialID, materialID, 16, 16);

    vec4 shadowClipPos = ftransform();
    vec4 shadowViewPos = shadowProjectionInverse * shadowClipPos;
    vec4 scenePos = shadowModelViewInverse * shadowViewPos;
    vec4 shadowNDCPos = global_sceneToShadowNDC * scenePos;
    gl_Position = shadowNDCPos;
    vec2 shadowScreenPosUnwarpped = shadowNDCPos.xy * 0.5 + 0.5;
    vert_screenPos = shadowScreenPosUnwarpped;
    vec2 texelSize;
    gl_Position.xy = rtwsm_warpTexCoordTexelSize(shadowScreenPosUnwarpped, texelSize) * 2.0 - 1.0;

    shadowViewPos /= shadowViewPos.w;
    vec4 camViewPos = gbufferModelView * scenePos;
    camViewPos /= camViewPos.w;

    uint survived = uint(all(lessThan(abs(shadowNDCPos.xyz), shadowNDCPos.www)));
    vert_survived = survived;

    survived &= uint((gl_VertexID & 3) == 0);

    #ifdef SETTING_RTWSM_F
    if (bool(survived)){
        ivec2 importanceTexelPos = ivec2(shadowScreenPosUnwarpped * vec2(RTWSM_IMAP_SIZE));

        float importance = SETTING_RTWSM_F_BASE * 16.0;

        float camDistanceSq = dot(camViewPos.xyz, camViewPos.xyz);

        if (isEyeInWater == 1 && materialID == 3u) {
            const float RIOR = AIR_IOR / WATER_IOR;
            vec3 refractDir = refract(-uval_shadowLightDirWorld, normalize(worldNormal), RIOR);

            // Calculate closest distance from refracted ray to origin (0,0,0)
            // Ray: P = camViewPos + t * refractDir
            // Closest point: t = -dot(camViewPos, refractDir) / dot(refractDir, refractDir)
            float t = -dot(camViewPos.xyz, refractDir) / dot(refractDir, refractDir);
            t = max(t, 0.0); // Clamp to ray origin side
            vec3 closestPointOnRay = camViewPos.xyz + t * refractDir;
            camDistanceSq = pow2(dot(closestPointOnRay, closestPointOnRay)) * 0.1;
        }

        camDistanceSq = max(4.0, camDistanceSq);

        // Distance function
        importance *= 1.0 / (1.0 + pow(camDistanceSq, SETTING_RTWSM_F_D));
        importance = max(importance, uval_rtwsmMin.x);
        persistent_rtwsm_importance2D_atomicMax(importanceTexelPos, floatBitsToInt(importance));
    }
    #endif
}