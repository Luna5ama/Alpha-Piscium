#include "/util/Colors.glsl"
#include "/util/Mat4.glsl"
#include "/util/Math.glsl"
#include "/util/Rand.glsl"
#include "/util/Morton.glsl"

layout(local_size_x = 1) in;
const ivec3 workGroups = ivec3(2, 1, 1);

layout(rgba16f) uniform writeonly readonly image2D uimg_main;

mat4 shadowDeRotateMatrix(mat4 shadowMatrix) {
    vec2 p1 = (shadowMatrix * vec4(0.0, -1000.0, 0.0, 1.0)).xy;
    vec2 p2 = (shadowMatrix * vec4(0.0, 1000.0, 0.0, 1.0)).xy;

    float angle1 = -atan(p1.y, p1.x);

    float cos1 = cos(angle1 - PI_HALF) * 0.7071;
    float sin1 = sin(angle1 - PI_HALF) * 0.7071;

    return mat4(
        cos1, sin1, 0.0, 0.0,
        -sin1, cos1, 0.0, 0.0,
        0.0, 0.0, 0.25, 0.0,
        0.0, 0.0, 0.0, 1.0
    );
}

vec2 taaJitter() {
    #ifdef SETTING_TAA_JITTER
    return rand_r2Seq2(frameCounter) - 0.5;
    #else
    return vec2(0.0);
    #endif
}

mat4 taaJitterMat(vec2 baseJitter) {
    vec2 jitter = baseJitter * 2.0 * (1.0 / imageSize(uimg_main));
    return mat4(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        jitter.x, jitter.y, 0.0, 1.0
    );
}

void main() {
    if (gl_WorkGroupID.x == 0) {
        global_shadowAABBMin = min(global_shadowAABBMin, ivec3(floor(mix(vec3(global_shadowAABBMin), vec3(global_shadowAABBMinPrev), 0.9))));
        global_shadowAABBMax = max(global_shadowAABBMax, ivec3(ceil(mix(vec3(global_shadowAABBMax), vec3(global_shadowAABBMaxPrev), 0.9))));
        vec2 jitter = taaJitter();
        global_shadowRotationMatrix = shadowDeRotateMatrix(shadowModelView);
        global_taaJitter = jitter;
        mat4 taaMat = taaJitterMat(jitter);
        global_taaJitterMat = taaMat;

        ivec2 mainImageSize = imageSize(uimg_main);
        global_mainImageSizeI = mainImageSize;
        global_mainImageSize = vec2(mainImageSize);
        global_mainImageSizeRcp = 1.0 / vec2(mainImageSize);

        global_frameMortonJitter = morton_8bDecode(uint(frameCounter) & 3u);

        global_prevCamProj = global_camProj;
        global_prevCamProjInverse = global_camProjInverse;
        global_camProj = mat4_infRevZFromRegular(gbufferProjection, near);
        global_camProjInverse = inverse(global_camProj);
    } else {
        ivec2 mainImageSize = imageSize(uimg_main);
        for (uint i = 0; i < 16; i++) {
            ivec2 mipSize = mainImageSize >> i;
            global_mipmapSizes[i] = vec2(mipSize);
            global_mipmapSizesRcp[i] = 1.0 / vec2(mipSize);
            global_mipmapSizesI[i] = mipSize;
            if (i == 0) {
                global_mipmapSizePrefixes[i] = mipSize;
            } else {
                global_mipmapSizePrefixes[i] = global_mipmapSizePrefixes[i - 1] + mipSize;
            }
        }

        global_dispatchSize1 = uvec4(0u, 1u, 1u, 0u);
        global_dispatchSize2 = uvec4(0u, 1u, 1u, 0u);
        global_dispatchSize3 = uvec4(0u, 1u, 1u, 0u);
        global_dispatchSize4 = uvec4(0u, 1u, 1u, 0u);
        int worldTimeDiff = min(
            abs(global_lastWorldTime + 24000 - worldTime) % 24000,
            abs(worldTime + 24000 - global_lastWorldTime) % 24000
        );
        global_historyResetFactor = exp2(-pow2(mix(0.05, 0.2, uval_dayNightTransition) * float(worldTimeDiff)));
        global_lastWorldTime = worldTime;

        #ifdef SETTING_DOF_MANUAL_FOCUS
        global_focusDistance = SETTING_DOF_FOCUS_DISTANCE_COARSE + SETTING_DOF_FOCUS_DISTANCE_FINE;
        #endif
    }
}