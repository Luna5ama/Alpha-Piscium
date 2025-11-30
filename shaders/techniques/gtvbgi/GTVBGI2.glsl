/*
    References:
        [SAL24] Salm, Mirko. "GT-VBGI (diffuse) (unidir)". 2024.
            MIT License. Copyright (c) 2024 Mirko Salm.
            https://www.shadertoy.com/view/XcdBWf

        You can find full license texts in /licenses
*/

#define SSVBIL_SAMPLE_STEPS SSVBIL_SAMPLE_STEPS222
#define SSVBIL_SAMPLE_SLICES SETTING_VBGI_SLICES
#define RANDOM_FRAME (frameCounter - SKIP_FRAMES)

#include "/techniques/EnvProbe.glsl"
#include "/util/Coords.glsl"
#include "/util/Lighting.glsl"
#include "/util/NZPacking.glsl"
#include "/util/BSDF.glsl"
#include "/util/FastMathLib.glsl"
#include "/util/Math.glsl"
#include "/util/Hash.glsl"
#include "Common.glsl"



ivec2 vbgi_texelPos = ivec2(0);

// Inverse function approximation
// See https://www.desmos.com/calculator/gs3clmp5hj
float radiusToLodStep(float y) {
    #if SSVBIL_SAMPLE_STEPS == 8
    const float a0 = 0.163523912451;
    const float a1 = 0.266941582847;
    const float a2 = -1.22024640462;
    #elif SSVBIL_SAMPLE_STEPS == 12
    const float a0 = 0.106829556893;
    const float a1 = 0.171480940545;
    const float a2 = -1.13391617357;
    #elif SSVBIL_SAMPLE_STEPS == 16
    const float a0 = 0.0795356209197;
    const float a1 = 0.125400926129;
    const float a2 = -1.07532664373;
    #elif SSVBIL_SAMPLE_STEPS == 24
    const float a0 = 0.0527947221187;
    const float a1 = 0.0808744790024;
    const float a2 = -0.999305945929;
    #elif SSVBIL_SAMPLE_STEPS == 32
    const float a0 = 0.0396467304144;
    const float a1 = 0.0590825497733;
    const float a2 = -0.939977972788;
    #elif SSVBIL_SAMPLE_STEPS == 48
    const float a0 = 0.0264701344291;
    const float a1 = 0.0383770338223;
    const float a2 = -0.882202260084;
    #elif SSVBIL_SAMPLE_STEPS == 64
    const float a0 = 0.0199939489785;
    const float a1 = 0.0279244889461;
    const float a2 = -0.817485681694;
    #elif SSVBIL_SAMPLE_STEPS == 96
    const float a0 = 0.0134047383578;
    const float a1 = 0.0181493759954;
    const float a2 = -0.764070193656;
    #elif SSVBIL_SAMPLE_STEPS == 128
    const float a0 = 0.0101298394507;
    const float a1 = 0.0132862874602;
    const float a2 = -0.715597089117;
    #else
    #error "Invalid SSVBIL_SAMPLE_STEPS"
    #endif
    y = clamp(y, float(SSVBIL_SAMPLE_STEPS), 32768.0);
    return max(a0 * log2(a1 * y + a2), 0.0);
}

float lodTexelSize(float lod) {
    return exp2(lod);
}


vec3 view2screen(vec3 vpos) {
    vec4 ppos = global_camProj * vec4(vpos, 1.0);
    vec2 tc21 = ppos.xy / ppos.w;
    vec2 uv0 = (tc21 * 0.5 + 0.5) * uval_mainImageSize;
    return vec3(uv0, vpos.z);
}

#define NOISE_FRAME RANDOM_FRAME

//==================================================================================//
//////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////// partial slice sampling
//==================================================================================//

vec2 cmul(vec2 c0, vec2 c1) {
    return vec2(c0.x * c1.x - c0.y * c1.y,
        c0.y * c1.x + c0.x * c1.y);
}

float SamplePartialSlice(float x, float sin_thVN) {
    if (x == 0.0 || abs(x) >= 1.0) return x;

    bool sgn = x < 0.0;
    x = abs(x);

    float s = sin_thVN;

    float o = s - s * s;
    float slp0 = 1.0 / (1.0 + (PI  - 1.0) * (s - o * 0.30546));
    float slp1 = 1.0 / (1.0 - (1.0 - exp2(-20.0)) * (s + o * mix(0.5, 0.785, s)));
    float k = mix(0.1, 0.25, s);

    float a = 1.0 - (PI - 2.0) / (PI - 1.0);
    float b = 1.0 / (PI - 1.0);

    float d0 =   a - slp0 * b;
    float d1 = 1.0 - slp1;

    float f0 = d0 * (PI * x - asinFast4(clamp(x, -1.0, 1.0)));
    float f1 = d1 * (x - 1.0);

    float kk = k * k;

    float h0 = sqrt(f0*f0 + kk) - k;
    float h1 = sqrt(f1*f1 + kk) - k;

    float hh = (h0 * h1) / (h0 + h1);

    float y = x - sqrt(hh*(hh + 2.0*k));

    return sgn ? -y : y;
}

vec2 SamplePartialSliceDir(vec3 vvsN, vec2 dir0) {
    float l = length(vvsN.xy);
    if (l == 0.0) return dir0;

    vec2 n = vvsN.xy / l;
    // align n with x-axis
    dir0 = cmul(dir0, n * vec2(1.0, -1.0));

    // sample slice angle
    float ang;
    {
        float x = atan(dir0.x, dir0.y) / PI;
        float sinNV = l;
        ang = SamplePartialSlice(x, sinNV) * PI;
    }

    // ray space slice direction
    vec2 dir = vec2(cos(ang), sin(ang));
    // align x-axis with n
    dir = cmul(dir, n);

    return dir;
}

//==================================================================================//
//////////////////////////////////////////////////////////////////////////////////////

const float MAX_RADIUS_SQ = SETTING_VBGI_MAX_RADIUS * SETTING_VBGI_MAX_RADIUS;

////////////////////////////////////////////////////////////////////////////////////// quaternion utils
//==================================================================================//

vec4 GetQuaternion(vec3 from, vec3 to) {
    vec3 xyz = cross(from, to);
    float s  =   dot(from, to);

    float u = inversesqrt(max(0.0, s * 0.5 + 0.5));// rcp(cosine half-angle formula)

    s    = 1.0 / u;
    xyz *= u * 0.5;

    return vec4(xyz, s);
}

vec4 GetQuaternion(vec3 to) {
    //vec3 from = vec3(0.0, 0.0, 1.0);

    vec3 xyz = vec3(-to.y, to.x, 0.0);// cross(from, to);
    float s  =                   to.z;//   dot(from, to);

    float u = inversesqrt(max(0.0, s * 0.5 + 0.5));// rcp(cosine half-angle formula)

    s    = 1.0 / u;
    xyz *= u * 0.5;

    return vec4(xyz, s);
}

// transform v by unit quaternion q.xyzs
vec3 Transform(vec3 v, vec4 q) {
    vec3 k = cross(q.xyz, v);

    return v + 2.0 * vec3(dot(vec3(q.wy, -q.z), k.xzy),
        dot(vec3(q.wz, -q.x), k.yxz),
        dot(vec3(q.wx, -q.y), k.zyx));
}

// transform v by unit quaternion q.xy0s
vec3 Transform_Qz0(vec3 v, vec4 q) {
    float k = v.y * q.x - v.x * q.y;
    float g = 2.0 * (v.z * q.w + k);

    vec3 r;
    r.xy = v.xy + q.yx * vec2(g, -g);
    r.z  = v.z  + 2.0 * (q.w * k - v.z * dot(q.xy, q.xy));

    return r;
}

// transform v.xy0 by unit quaternion q.xy0s
vec3 Transform_Vz0Qz0(vec2 v, vec4 q) {
    float o = q.x * v.y;
    float c = q.y * v.x;

    vec3 b = vec3(o - c,
        -o + c,
        o - c);

    return vec3(v, 0.0) + 2.0 * (b * q.yxw);
}

float sliceRelCDF(float x, float angN, float cosN) {
    if (x <= 0.0 || x >= 1.0) return x;

    float phi = x * PI - PI_HALF;

    const float n0 = 3.0;
    const float n1 = -1.0;
    const float n2 = 4.0;

    float t0 = n0 * cosN + n1 * cos(angN - 2.0 * phi) + (n2 * angN + (n1 * 2.0) * phi + PI) * sin(angN);
    float t1 = 4.0 * (cosN + angN * sin(angN));

    return t0 / t1;
}

const vec2 RADIUS_SQ = vec2(SETTING_VBGI_RADIUS * SETTING_VBGI_RADIUS, SETTING_VBGI_MAX_RADIUS * SETTING_VBGI_MAX_RADIUS);

uint toBitMask(vec2 h01) {
    uint bitmask;// turn arc into bit mask
    {
        uvec2 horInt = uvec2(floor(h01 * 32.0));

        uint OxFFFFFFFFu = 0xFFFFFFFFu;// don't inline here! ANGLE bug: https://issues.angleproject.org/issues/353039526

        uint mX = horInt.x < 32u ? OxFFFFFFFFu <<        horInt.x  : 0u;
        uint mY = horInt.y != 0u ? OxFFFFFFFFu >> (32u - horInt.y) : 0u;

        bitmask = mX & mY;
    }
    return bitmask;
}

void uniGTVBGI(vec3 viewPos, vec3 viewNormal, inout vec3 result) {
    vec3 viewDir = -normalize(viewPos);
    vec2 rayStart = vec2(vbgi_texelPos) + 0.5;

    ////////////////////////////////////////////////// slice direction sampling
    vec3 smplDirVS;// view space sampling vector
    vec2 dir;// screen space sampling vector
    {
        // approximate partial slice dir importance sampling via 1 rnd number

        // set up View Vec Space <-> View Space mapping
        vec4   Q_toV = GetQuaternion(viewDir);
        vec4 Q_fromV = Q_toV * vec4(vec3(-1.0), 1.0);// conjugate
        vec3 normalVVS = viewNormal;
        normalVVS = Transform_Qz0(viewNormal, Q_fromV);

        vec2 dir0 = rand_stbnUnitVec211(vbgi_texelPos, NOISE_FRAME);
        dir = SamplePartialSliceDir(normalVVS, dir0);

        smplDirVS = vec3(dir.xy, 0.0);
        smplDirVS = Transform_Vz0Qz0(dir, Q_toV);

        vec3 rayStart = view2screen(viewPos);
        vec3 rayEnd = view2screen(viewPos + smplDirVS * (near * 0.5));
        vec3 rayDir = rayEnd - rayStart;

        rayDir /= length(rayDir.xy);
        dir = rayDir.xy;
    }
    //////////////////////////////////////////////////

    ////////////////////////////////////////////////// construct slice
    vec3 sliceN, projN, T;
    float cosN, angN, projNRcpLen;
    float sgn;
    {
        sliceN = cross(viewDir, smplDirVS);
        projN = viewNormal - sliceN * dot(viewNormal, sliceN);

        float projNSqrLen = dot(projN, projN);
        if (projNSqrLen == 0.0) return;

        projNRcpLen = inversesqrt(projNSqrLen);
        cosN = dot(projN, viewDir) * projNRcpLen;
        T = cross(sliceN, projN);

        sgn = dot(viewDir, T) < 0.0 ? -1.0 : 1.0;
        angN = sgn * acosFast4(clamp(cosN, -1.0, 1.0));
    }
    //////////////////////////////////////////////////

    float angOff = angN * RCP_PI + 0.5;

    // percentage of the slice we don't use ([0, angN]-integrated slice-relative pdf)
    float w0 = clamp((sin(angN) / (cos(angN) + angN * sin(angN))) * (PI/4.0) + 0.5, 0.0, 1.0);

    // partial slice re-mapping constants
    float w0_remap_mul = 1.0 / (1.0 - w0);
    float w0_remap_add = -w0 * w0_remap_mul;

    vec2 rayDir = dir.xy;

    vec2 texelCenterPos = vec2(vbgi_texelPos) + 0.5;
    float maxDistX = rayDir.x != 0.0 ? (rayDir.x >= 0.0 ? (uval_mainImageSize.x - texelCenterPos.x) / rayDir.x : -texelCenterPos.x / rayDir.x) : 1e6;
    float maxDistY = rayDir.y != 0.0 ? (rayDir.y < 0.0 ? (-texelCenterPos.y / rayDir.y) : (uval_mainImageSize.y - texelCenterPos.y) / rayDir.y) : 1e6;
    float maxDist = min(maxDistX, maxDistY) - 2.0;

    uvec2 hashKey = uvec2(vbgi_texelPos) & uvec2(255u);
    uint r2Index = (hash_21_q3(hashKey) & 1023u) + NOISE_FRAME;
    float jitter = rand_r2Seq1(r2Index);

    float lodStep = radiusToLodStep(maxDist);
    float sampleLod = 0.0;

    float sampleTexelDist = 1.5;

    uint occBits = 0u;

    float NDotV = dot(viewNormal, viewDir);
    GBufferData gData = gbufferData_init();
    gbufferData1_unpack(texelFetch(usam_gbufferData1, vbgi_texelPos, 0), gData);
    Material material = material_decode(gData);
    material.roughness = max(material.roughness, 0.01);

    for (uint stepIndex = 0; stepIndex < SSVBIL_SAMPLE_STEPS; ++stepIndex) {
        float sampleLodTexelSize = lodTexelSize(sampleLod);
        sampleTexelDist += sampleLodTexelSize * jitter;
        sampleTexelDist = min(sampleTexelDist, maxDist);

        ivec2 sampleTexelPos = clamp(ivec2(rayDir * sampleTexelDist + rayStart), ivec2(0), uval_mainImageSizeI - 1);
        vec2 sampleUV = saturate((sampleTexelPos + 0.5) * uval_mainImageSizeRcp);

//        float sampleViewZ;
//        vec3 sampleViewNormal;
//        nzpacking_unpack(texelFetch(usam_packedZN, sampleTexelPos, 0).xy, sampleViewNormal, sampleViewZ);

        float sampleViewZ = texelFetch(usam_gbufferViewZ, sampleTexelPos, 0).x;

        vec3 samplePosVS = coords_toViewCoord(sampleUV, sampleViewZ, global_camProjInverse);
        vec3 frontDiff = samplePosVS - viewPos;
        float frontDistSq = dot(frontDiff, frontDiff);

        if (frontDistSq < RADIUS_SQ.y) {
            float frontDiffRcpLen = fastRcpSqrtNR0(frontDistSq);
            float frontDist = frontDistSq * frontDiffRcpLen;
            float thickness = max(SETTING_VBGI_THICKNESS * frontDist, 0.25);
            vec3 backDiff = coords_toViewCoord(sampleUV, sampleViewZ - thickness, global_camProjInverse) - viewPos;

            float backDiffRcpLen = fastRcpSqrtNR0(dot(backDiff, backDiff));
            vec3 thisToSample = frontDiff * frontDiffRcpLen;

            // project samples onto unit circle and compute angles relative to viewDir
            vec2 horCos = vec2(dot(thisToSample, viewDir), dot(backDiff * backDiffRcpLen, viewDir));

            vec2 horAng = acosFast4(clamp(horCos, -1.0, 1.0));

            // shift relative angles from viewDir to N + map to [0,1]
            vec2 hor01 = saturate(horAng * RCP_PI + angOff);

            // map to slice relative distribution
            hor01.x = sliceRelCDF(hor01.x, angN, cosN);
            hor01.y = sliceRelCDF(hor01.y, angN, cosN);

            // partial slice re-mapping
            hor01 = hor01 * w0_remap_mul + w0_remap_add;

            // jitter sample locations + clamp01
            hor01 = saturate(hor01);
            uint occBits0 = toBitMask(hor01);

            // compute gi contribution
            {
                uint visBits0 = occBits0 & (~occBits);

                if (visBits0 != 0u) {
                    vec4 radiance = texelFetch(usam_temp2, sampleTexelPos, 0);
                    vec3 sampleRad = radiance.rgb;
                    float bitV = float(bitCount(visBits0)) * (1.0 / 32.0);

                    result += sampleRad * bitV;
                }
            }

            occBits = occBits | occBits0;
        }

        sampleLod = sampleLod + lodStep;
        sampleTexelDist += sampleLodTexelSize * (1.0 - jitter);
    }
}

vec3 gtvbgi(ivec2 texelPos1x1) {
    vbgi_texelPos = texelPos1x1;

    float viewZ = texelFetch(usam_gbufferViewZ, vbgi_texelPos, 0).x;
    vec2 screenPos = coords_texelToUV(vbgi_texelPos, uval_mainImageSizeRcp);
    vec3 viewPos = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);

    GBufferData gData = gbufferData_init();
    gbufferData1_unpack(texelFetch(usam_gbufferData1, vbgi_texelPos, 0), gData);
    gbufferData2_unpack(texelFetch(usam_gbufferData2, vbgi_texelPos, 0), gData);

    float centerViewZ = viewZ;
    vec3 centerViewNormal = gData.normal;

    vec3 result = vec3(0.0, 0.0, 0.0);
    if (centerViewZ != -65536.0) {
        uniGTVBGI(viewPos, centerViewNormal, result);
    }

    return result;
}