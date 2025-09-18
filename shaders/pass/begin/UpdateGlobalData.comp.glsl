#define GLOBAL_DATA_MODIFIER \

#include "/techniques/gtvbgi/Common.glsl"
#include "/util/Colors.glsl"
#include "/util/Mat4.glsl"
#include "/util/Math.glsl"
#include "/util/Rand.glsl"
#include "/util/Morton.glsl"
#include "/util/Time.glsl"

layout(local_size_x = 1) in;
const ivec3 workGroups = ivec3(2, 1, 1);

layout(rgba16f) uniform writeonly readonly image2D uimg_main;

vec3 rotateAxis(vec3 unitAxis) {
    vec4 p1 = global_shadowView * vec4(unitAxis * -65536.0, 0.0);
    vec4 p2 = global_shadowView * vec4(unitAxis * 65536.0, 0.0);
    vec2 delta = p2.xy - p1.xy;
    return vec3(delta, dot(delta, delta));
}

mat4 shadowDeRotateMatrix() {
    vec3 axisX = rotateAxis(vec3(1.0, 0.0, 0.0));
    vec3 axisY = rotateAxis(vec3(0.0, 1.0, 0.0));
    vec3 axisZ = rotateAxis(vec3(0.0, 0.0, 1.0));

    vec3 maxAxisDelta = axisX;
    maxAxisDelta = axisY.z > maxAxisDelta.z ? axisY : maxAxisDelta;
    maxAxisDelta = axisZ.z > maxAxisDelta.z ? axisZ : maxAxisDelta;

    float angle1 = atan(maxAxisDelta.x, maxAxisDelta.y);
    float cos1 = cos(angle1);
    float sin1 = sin(angle1);

    mat2 rotMat1 = mat2(cos1, sin1, -sin1, cos1);

    vec2 axisY1 = rotMat1 * axisY.xy;

    float angleToYUp = atan(axisY1.x, axisY1.y);
    angleToYUp = round(angleToYUp * RCP_PI_HALF) * PI_HALF;
    float cos2 = cos(angle1 + angleToYUp);
    float sin2 = sin(angle1 + angleToYUp);

    return mat4(
        cos2, sin2, 0.0, 0.0,
        -sin2, cos2, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
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
        global_shadowAABBMin = min(global_shadowAABBMinNew, ivec3(floor(mix(vec3(global_shadowAABBMin), vec3(global_shadowAABBMinNew), 0.1))));
        global_shadowAABBMax = max(global_shadowAABBMaxNew, ivec3(ceil(mix(vec3(global_shadowAABBMax), vec3(global_shadowAABBMaxNew), 0.1))));
        vec4 shadowAABBMin = global_shadowRotationMatrix * shadowModelView * vec4(0.0, 0.0, 0.0, 1.0);
        vec4 shadowAABBMax = global_shadowRotationMatrix * shadowModelView * vec4(0.0, 0.0, 0.0, 1.0);
        global_shadowAABBMinNew = ivec3(floor(shadowAABBMin.xyz));
        global_shadowAABBMaxNew = ivec3(ceil(shadowAABBMax.xyz));

        vec3 cameraDelta = cameraPosition - previousCameraPosition;

        vec2 jitter = taaJitter();
        global_shadowRotationMatrix = shadowDeRotateMatrix();
        global_shadowProjPrev = global_shadowProj;
        global_shadowProjInversePrev = global_shadowProjInverse;
        global_shadowProj = mat4_createOrthographicMatrix(
            global_shadowAABBMin.x - 16.0, global_shadowAABBMax.x + 16.0,
            global_shadowAABBMin.y - 16.0, global_shadowAABBMax.y + 16.0,
            -global_shadowAABBMax.z - 512.0, -global_shadowAABBMin.z + 16.0
        );
        global_shadowProjInverse = inverse(global_shadowProj);
        global_taaJitter = jitter;
        mat4 taaMat = taaJitterMat(jitter);
        global_taaJitterMat = taaMat;

        ivec2 mainImageSize = imageSize(uimg_main);
        global_mainImageSizeI = mainImageSize;
        global_mainImageSize = vec2(mainImageSize);
        global_mainImageSizeRcp = 1.0 / vec2(mainImageSize);

        global_frameMortonJitter = morton_8bDecode(vbgi_downSampleInputMortonIndex());

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
                ivec4 mipTile1 = ivec4(ivec2(0), mipSize);
                global_mipmapTiles[0][i] = mipTile1;
                global_mipmapTiles[1][i] = mipTile1;
            } else {
                global_mipmapSizePrefixes[i] = global_mipmapSizePrefixes[i - 1] + mipSize;
                ivec2 mipTileOffset = ivec2(global_mipmapSizePrefixes[i - 1].x - mainImageSize.x, mainImageSize.y);
                ivec4 mipTile1 = ivec4(mipTileOffset, mipSize);
                ivec4 mipTile2 = mipTile1;
                mipTile2.y += global_mipmapSizesI[1].y;
                global_mipmapTiles[0][i] = mipTile1;
                global_mipmapTiles[1][i] = mipTile2;
            }
        }

        global_dispatchSize1 = uvec4(0u, 1u, 1u, 0u);
        global_dispatchSize2 = uvec4(0u, 1u, 1u, 0u);
        global_dispatchSize3 = uvec4(0u, 1u, 1u, 0u);
        global_dispatchSize4 = uvec4(0u, 1u, 1u, 0u);
        for (uint i = 0u; i < 16u; i++) {
            global_atomicCounters[i] = 0u;
        }
        int worldTimeDiff = min(
            abs(global_lastWorldTime + 24000 - worldTime) % 24000,
            abs(worldTime + 24000 - global_lastWorldTime) % 24000
        );
        global_historyResetFactor = exp2(-pow2(mix(0.05, 0.2, uval_dayNightTransition) * float(worldTimeDiff)));
        global_lastWorldTime = worldTime;

        #ifdef SETTING_DOF_MANUAL_FOCUS
        global_focusDistance = SETTING_DOF_FOCUS_DISTANCE_COARSE + SETTING_DOF_FOCUS_DISTANCE_FINE;
        #endif

        #ifdef SETTING_ATM_MIE_TIME
        float turbidityExp = 0.0;
        turbidityExp += SETTING_ATM_MIE_TURBIDITY_EARLY_MORNING * time_interpolate(TIME_MIDNIGHT, TIME_EARLY_MORNING, TIME_SUNRISE);
        turbidityExp += SETTING_ATM_MIE_TURBIDITY_SUNRISE * time_interpolate(TIME_EARLY_MORNING, TIME_SUNRISE, TIME_MORNING);
        turbidityExp += SETTING_ATM_MIE_TURBIDITY_MORNING * time_interpolate(TIME_SUNRISE, TIME_MORNING, TIME_NOON);
        turbidityExp += SETTING_ATM_MIE_TURBIDITY_NOON * time_interpolate(TIME_MORNING, TIME_NOON, TIME_AFTERNOON);
        turbidityExp += SETTING_ATM_MIE_TURBIDITY_AFTERNOON * time_interpolate(TIME_NOON, TIME_AFTERNOON, TIME_SUNSET);
        turbidityExp += SETTING_ATM_MIE_TURBIDITY_SUNSET * time_interpolate(TIME_AFTERNOON, TIME_SUNSET, TIME_NIGHT);
        turbidityExp += SETTING_ATM_MIE_TURBIDITY_NIGHT * time_interpolate(TIME_SUNSET, TIME_NIGHT, TIME_MIDNIGHT);
        turbidityExp += SETTING_ATM_MIE_TURBIDITY_MIDNIGHT * time_interpolate(TIME_NIGHT, TIME_MIDNIGHT, TIME_EARLY_MORNING);
        global_turbidity = exp2(turbidityExp);
        #else
        global_turbidity = exp2(SETTING_ATM_MIE_TURBIDITY);
        #endif
    }
}