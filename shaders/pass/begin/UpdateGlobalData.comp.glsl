#define GLOBAL_DATA_MODIFIER buffer

#include "/util/Coords.glsl"
#include "/util/Colors.glsl"
#include "/util/Mat4.glsl"
#include "/util/Math.glsl"
#include "/util/Rand.glsl"
#include "/util/Morton.glsl"
#include "/util/Time.glsl"

layout(local_size_x = 1) in;
const ivec3 workGroups = ivec3(3, 1, 1);

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

#define SMOOTH 0

vec3 shadowAABBSmooth(vec3 historyF, ivec3 newI) {
    vec3 newF = vec3(newI);
    vec3 dist = sqrt(abs(historyF - newF));
    return mix(newF, historyF, max(saturate(rcp(1.0 + dist * 0.1)), 0.8));
}

void main() {
    if (gl_WorkGroupID.x == 0) {
        global_shadowAABBMinHistory = min(vec3(global_shadowAABBMinNew), shadowAABBSmooth(global_shadowAABBMinHistory, global_shadowAABBMinNew));
        global_shadowAABBMaxHistory = max(vec3(global_shadowAABBMaxNew), shadowAABBSmooth(global_shadowAABBMaxHistory, global_shadowAABBMaxNew));
        global_shadowAABBMinPrev = global_shadowAABBMin;
        global_shadowAABBMaxPrev = global_shadowAABBMax;

        global_shadowAABBMin = ivec3(floor(global_shadowAABBMinHistory / 16.0)) * 16;
        global_shadowAABBMax = ivec3(ceil(global_shadowAABBMaxHistory / 16.0)) * 16;

        vec4 shadowAABBMin = global_shadowRotationMatrix * shadowModelView * vec4(0.0, 0.0, 0.0, 1.0);
        vec4 shadowAABBMax = global_shadowRotationMatrix * shadowModelView * vec4(0.0, 0.0, 0.0, 1.0);
        global_shadowAABBMinNew = ivec3(floor(shadowAABBMin.xyz));
        global_shadowAABBMaxNew = ivec3(ceil(shadowAABBMax.xyz));

        vec3 cameraDelta = uval_cameraDelta;

        vec2 jitter = taaJitter();
        global_shadowRotationMatrix = shadowDeRotateMatrix();
        global_shadowRotationMatrixInverse = inverse(global_shadowRotationMatrix);
        global_shadowProjPrev = global_shadowProj;
        global_shadowProjInversePrev = global_shadowProjInverse;
        global_shadowProj = mat4_createOrthographicMatrix(
            global_shadowAABBMin.x - 16.0, global_shadowAABBMax.x + 16.0,
            global_shadowAABBMin.y - 16.0, global_shadowAABBMax.y + 16.0,
            -global_shadowAABBMax.z - 512.0, -global_shadowAABBMin.z + 16.0
        );
        global_shadowProjInverse = inverse(global_shadowProj);
        global_prevTaaJitter = global_taaJitter;
        global_taaJitter = jitter;
        mat4 taaMat = taaJitterMat(jitter);
        global_taaJitterMat = taaMat;

        global_prevCamProj = global_camProj;
        global_prevCamProjInverse = global_camProjInverse;
        global_camProj = mat4_infRevZFromRegular(gbufferProjection, near);
        global_camProjInverse = inverse(global_camProj);

        {
            const float EPS = 1e-32;
            mat4 rowProj = transpose(gbufferProjection);
            vec4 row0 = rowProj[0];
            vec4 row1 = rowProj[1];
            vec4 row2 = rowProj[2];
            vec4 row3 = rowProj[3];

            // Build candidate planes. We use the form:
            //   plane(p) = dot(plane.xyz, p.xyz) + plane.w  >= 0  means "inside".
            // For NDC x,y in [-w,w] and z in [0,w] (reverse-Z):
            // left:   q.w + q.x >= 0   ->  row3 + row0
            // right:  q.w - q.x >= 0   ->  row3 - row0
            // bottom: q.w + q.y >= 0   ->  row3 + row1
            // top:    q.w - q.y >= 0   ->  row3 - row1
            // near:   q.w - q.z >= 0   ->  row3 - row2  (reverse-Z: near maps to NDC z = 1)
            // far:    q.z     >= 0     ->  row2         (maps to NDC z = 0)  -- may be degenerate for infinite far
            vec4 cand[6];
            cand[0] = row3 + row0;// left
            cand[1] = row3 - row0;// right
            cand[2] = row3 + row1;// bottom
            cand[3] = row3 - row1;// top
            cand[4] = row3 + row2;// near (reverse-Z)
            cand[5] = row3 - row2;// far-ish (z >= 0). Skip if degenerate.

            // normalize and collect valid planes
            vec4 planes[6];
            uint pcount = 0;
            for (uint i = 0; i < 6; ++i) {
                float nlen = length(cand[i].xyz);
                if (nlen > EPS) {
                    global_cameraData.frustumPlanes[pcount++] = cand[i] / nlen; // normalize plane (xyz and w)
                }
            }
            global_cameraData.frustumPlaneCount = pcount;
        }
    } else if (gl_WorkGroupID.x == 1) {
        ivec2 mainImageSize = imageSize(uimg_main);
        for (uint i = 0; i < 16; i++) {
            ivec2 mipSize = mainImageSize >> i;
            global_mipmapSizes[i] = vec2(mipSize);
            global_mipmapSizesRcp[i] = 1.0 / vec2(mipSize);
            global_mipmapSizesI[i] = mipSize;
            if (i == 0) {
                global_mipmapSizePrefixesPadded[i] = mipSize + 1;
                global_mipmapSizePrefixes[i] = mipSize;
            } else {
                global_mipmapSizePrefixesPadded[i] = global_mipmapSizePrefixesPadded[i - 1] + mipSize + 1;
                global_mipmapSizePrefixes[i] = global_mipmapSizePrefixes[i - 1] + mipSize;
            }
        }

        global_dispatchSize1 = uvec4(0u, 1u, 1u, 0u);

        global_dispatchSize2 = uvec4(uvec2((uval_mainImageSizeI + 63) / 64), 3u, 0u);

        global_dispatchSize3 = uvec4(0u, 1u, 1u, 0u);
        global_dispatchSize4 = uvec4(0u, 1u, 1u, 0u);
        for (uint i = 0u; i < 16u; i++) {
            global_atomicCounters[i] = 0u;
        }
        int worldTimeDiff = min(
            abs(global_lastWorldTime + 24000 - worldTime) % 24000,
            abs(worldTime + 24000 - global_lastWorldTime) % 24000
        );
        float newResetFactor = exp2(-float(worldTimeDiff) * ldexp(1.0, SETTING_TIME_CHANGE_SENSITIVITY));
        newResetFactor = min(mix(global_historyResetFactor, newResetFactor, 0.1), newResetFactor);
        global_historyResetFactor = newResetFactor;
        global_lastWorldTime = worldTime;

        vec3 cameraDelta = uval_cameraDelta;
        float cameraSpeed = length(cameraDelta);
        float prevCameraSpeed = length(global_prevCameraDelta);
        float cameraSpeedDiff = cameraSpeed - prevCameraSpeed;
        float cameraSpeedDiffAbc = abs(cameraSpeed - prevCameraSpeed);
        vec3 prevFrontVec = coords_dir_viewToWorldPrev(vec3(0.0, 0.0, -1.0));
        vec3 currFrontVec = coords_dir_viewToWorld(vec3(0.0, 0.0, -1.0));
        float frontVecDiff = dot(prevFrontVec, currFrontVec);
        vec4 lastMotionFactor = global_motionFactor;
        float angleVecDiff = abs(lastMotionFactor.z - frontVecDiff);

        vec4 taaResetFactor = vec4(0.5, 1.0, 1.0, 1.0);
        const float SPEED_EPS = 1e-16;
        uint startOrEndMove = uint(cameraSpeedDiffAbc > SPEED_EPS);
        startOrEndMove &= uint(cameraSpeed < SPEED_EPS) | uint(prevCameraSpeed < SPEED_EPS);

        const float ANGLE_EPS = 1e-16;
        const float REV_ANGLE_EPS = 0.99999;
        uint startOrEndRotate = uint(angleVecDiff > ANGLE_EPS);
        startOrEndRotate &= uint(frontVecDiff >= REV_ANGLE_EPS) | uint(lastMotionFactor.z >= REV_ANGLE_EPS);

        float stationary = 1.0;
        stationary *= float(cameraSpeedDiffAbc < 0.00001);
        stationary *= float(cameraSpeed < 0.0001);
        stationary *= float(frontVecDiff > 0.99999);

        float startOrEndMoveF = float(startOrEndMove);
        float startOrEndRotateF = float(startOrEndRotate);
        float startOrEndMoveRotateF = float(startOrEndMove | startOrEndRotate);

        #ifdef SETTING_SCREENSHOT_MODE
        taaResetFactor.w *= 1.0 - stationary;
        taaResetFactor.yz *= 1.0 - startOrEndMoveRotateF;
        taaResetFactor.z *= float(frameCounter > SETTING_SCREENSHOT_MODE_SKIP_INITIAL);
        #endif
        taaResetFactor.y *= 1.0 - startOrEndMoveF * 0.5;
        taaResetFactor.z *= 1.0 - startOrEndMoveF * 0.25;
        const float DECAY = 0.1;
        float log2Speed = log2(cameraSpeed + 1.0);
        taaResetFactor.y *= DECAY * rcp(DECAY + log2Speed + pow3(max(-16.0 * cameraSpeedDiff, 0.0)));
        taaResetFactor.x += log2(log2Speed + abs(cameraSpeedDiff) * 2.0 + 1.0) * 2.0;
        taaResetFactor.z *= newResetFactor;

        vec4 finalTaaResetFactor = mix(global_taaResetFactor, taaResetFactor, 0.25);
        finalTaaResetFactor.yz = min(finalTaaResetFactor.yz, taaResetFactor.yz);
        finalTaaResetFactor.xw = max(finalTaaResetFactor.xw, taaResetFactor.xw);
        global_taaResetFactor = finalTaaResetFactor;
        global_motionFactor = vec4(cameraSpeed, cameraSpeedDiff, frontVecDiff, 0.0);

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
    } else {
        ivec4 mipTileCeil0 = ivec4(0, 0, uval_mainImageSizeI);
        ivec4 mipTileCeil1 = ivec4(0, 0, ivec2(ceil(uval_mainImageSize / 2.0)));
        global_mipmapTileCeil[0] = mipTileCeil0;
        global_mipmapTileCeilPadded[0] = mipTileCeil0;
        global_hizTiles[0] = mipTileCeil0;

        global_mipmapTileCeil[1] = mipTileCeil1;
        ivec4 hizTile1 = mipTileCeil1;
        hizTile1.y += uval_mainImageSizeI.y;
        global_hizTiles[1] = hizTile1;

        ivec4 mipTileCeilPadded1 = mipTileCeil1;
        mipTileCeilPadded1.xy += 1;
        global_mipmapTileCeilPadded[1] = mipTileCeilPadded1;

        for (uint i = 2u; i < 16u; i++) {
            ivec4 mipTile = ivec4(0);
            mipTile.zw = max(ivec2(ceil(ldexp(uval_mainImageSize, ivec2(-int(i))))), ivec2(1));
            ivec4 mipTilePadded = mipTile;
            ivec4 lastTile = global_mipmapTileCeil[i - 1u];
            ivec4 lastTilePadded = global_mipmapTileCeilPadded[i - 1u];
            if (bool(i & 1u)) {
                mipTile.x = lastTile.x + lastTile.z;
                mipTile.y = lastTile.y;
                mipTilePadded.x = lastTilePadded.x + lastTilePadded.z + 1;
                mipTilePadded.y = lastTilePadded.y;
            } else {
                mipTile.x = lastTile.x;
                mipTile.y = lastTile.y + lastTile.w;
                mipTilePadded.x = lastTilePadded.x;
                mipTilePadded.y = lastTilePadded.y + lastTilePadded.w + 1;
            }
            global_mipmapTileCeil[i] = mipTile;
            global_mipmapTileCeilPadded[i] = mipTilePadded;

            ivec4 mipTile1 = mipTile;
            mipTile1.y += uval_mainImageSizeI.y;
            global_hizTiles[i] = mipTile1;
        }
    }
}