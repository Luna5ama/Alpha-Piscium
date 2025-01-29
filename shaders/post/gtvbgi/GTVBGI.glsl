// Adopted from: https://www.shadertoy.com/view/XcdBWf
// MIT License
// You can find full license texts in /licenses
#include "GTVBGICommon.glsl"

uniform sampler2D usam_viewZ;
uniform sampler2D usam_temp1;
uniform sampler2D usam_temp2;
uniform sampler2D usam_skyLUT;


in vec2 frag_texCoord;

/* RENDERTARGETS:14 */
layout(location = 0) out vec4 rt_out;

// Inverse function approximation
// See https://www.desmos.com/calculator/cdliscjjvi
float radiusToLodStep(float y) {
    #if SSVBIL_SAMPLE_STEPS == 8
    const float a0 = 0.16378974015;
    const float a1 = 0.265434760971;
    const float a2 = -1.20565634012;
    #elif SSVBIL_SAMPLE_STEPS == 12
    const float a0 = 0.101656225858;
    const float a1 = 0.209110996684;
    const float a2 = -1.6842356559;
    #elif SSVBIL_SAMPLE_STEPS == 16
    const float a0 = 0.0731878065138;
    const float a1 = 0.181085743518;
    const float a2 = -2.15482462553;
    #elif SSVBIL_SAMPLE_STEPS == 24
    const float a0 = 0.046627957233;
    const float a1 = 0.152328399412;
    const float a2 = -3.03009898524;
    #elif SSVBIL_SAMPLE_STEPS == 32
    const float a0 = 0.0341139143777;
    const float a1 = 0.137416190981;
    const float a2 = -3.83742275706;
    #elif SSVBIL_SAMPLE_STEPS == 64
    const float a0 = 0.0163745667855;
    const float a1 = 0.114241124966;
    const float a2 = -6.79009299987;
    #else
    #error "Invalid SSVBIL_SAMPLE_STEPS"
    #endif
    y = clamp(y, SSVBIL_SAMPLE_STEPS, 32768.0);
    return saturate(a0 * log2(a1 * y + a2));
}

float lodTexelSize(float lod) {
    return exp2(lod);
}

#define NOISE_FRAME uint(frameCounter)

#define ACOS_QUALITY_MODE 2

#define USE_HQ_APPROX_SLICE_IMPORTANCE_SAMPLING

//==================================================================================//
//////////////////////////////////////////////////////////////////////////////////////

//#define KeyBoard iChannel3
//#define KeyBoardChannel w
//
//float ReadKey(int keyCode) { return texelFetch(KeyBoard, ivec2(keyCode, 0), 0).KeyBoardChannel; }
//float ReadKeyToggle(int keyCode) { return texelFetch(KeyBoard, ivec2(keyCode, 2), 0).KeyBoardChannel; }
//
//#define VarTex iChannel3
//#define OutCol outCol
//#define OutChannel w
//
//#define WriteVar(v, cx, cy) { if(uv.x == uint(cx) && uv.y == uint(cy)) OutCol.OutChannel = v; }
//#define WriteVar4(v, cx, cy) { WriteVar(v.x, cx, cy) WriteVar(v.y, cx, cy + 1) WriteVar(v.z, cx, cy + 2) WriteVar(v.w, cx, cy + 3) }
//
//float ReadVar(int cx, int cy) { return texelFetch(VarTex, ivec2(cx, cy), 0).OutChannel; }
//vec4 ReadVar4(int cx, int cy) { return vec4(ReadVar(cx, cy), ReadVar(cx, cy + 1), ReadVar(cx, cy + 2), ReadVar(cx, cy + 3)); }


////////////////////////////////////////////////////////////////////////////////////// atan/acos/asin approx
//==================================================================================//

// https://www.shadertoy.com/view/lXBfWm
// dir: normalized vector | out: angle in radians [-Pi, Pi] (max abs error ~0.000000546448 rad)
float ArcTan(vec2 dir) {
    const float Pi = 3.14159265359;

    float x = abs(dir.x);
    float y =     dir.y;

    //float u = 0.63662 + x * (0.405285 + x * (-0.0602976 + x * (0.0289292 + x * (-0.0162831 + (0.0075353 - 0.00178826 * x) * x))));
    float u = 0.63662 + x * (0.405285 + x * (-0.0602976 + (0.0261141 - 0.00772104 * x) * x));// max abs err ~0.0000454545 rad

    float f = y / u;

    if (dir.x < 0.0) f = (dir.y < 0.0 ? -Pi : Pi) - f;

    return f;
}

float ArcTan11(vec2 dir)// == ArcTan(dir) / Pi
{
    float x = abs(dir.x);
    float y =     dir.y;

    //float u = 2.0 + x * (1.27324 + x * (-0.189431 + x * (0.0908837 + x * (-0.0511549 + (0.0236728 - 0.005618 * x) * x))));
    float u = 2.0 + x * (1.27324 + x * (-0.189431 + (0.08204 - 0.0242564 * x) * x));

    float f = y / u;

    if (dir.x < 0.0) f = (dir.y < 0.0 ? -1.0 : 1.0) - f;

    return f;
}

float ACosPoly(float x) {
    #if ACOS_QUALITY_MODE == 1
    // GTAOFastAcos
    return 1.5707963267948966 - 0.1565827644218014 * x;
    #else
    // higher quality version of GTAOFastAcos (for the cost of one additional mad)
    // minimizes max abs(ACos_Approx(cos(x)) - x)
    return 1.5707963267948966 + (-0.20491203466059038 + 0.04832927023878897 * x) * x;
    #endif
}

float ACos_Approx(float x) {
    float u = ACosPoly(abs(x)) * sqrt(1.0 - abs(x));

    return x >= 0.0 ? u : Pi - u;
}

float ACos01_Approx(float x)// x: [0,1]
{
    return ACosPoly(x) * sqrt(1.0 - x);
}

float ASin_Approx(float x) {
    float u = ACosPoly(abs(x)) * sqrt(1.0 - abs(x)) - 1.5707963267948966;

    return x >= 0.0 ? -u : u;
}

float ASin01_Approx(float x)// x: [0,1]
{
    return 1.5707963267948966 - ACosPoly(x) * sqrt(1.0 - x);
}

#if ACOS_QUALITY_MODE == 1 || ACOS_QUALITY_MODE == 2
float ACos(float x) {
    return ACos_Approx(clamp(x, -1.0, 1.0));
}
#else
float ACos(float x)// very slow; for debugging
{
    return acos(clamp(x, -1.0, 1.0));
}
#endif

vec2 ACos(vec2 v) {
    return vec2(ACos(v.x), ACos(v.y));
}

//==================================================================================//
//////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////// partial slice sampling
//==================================================================================//

vec2 cmul(vec2 c0, vec2 c1) {
    return vec2(c0.x * c1.x - c0.y * c1.y,
        c0.y * c1.x + c0.x * c1.y);
}

float SamplePartialSlice(float x, float sin_thVN) {
    const float Pi   = 3.1415926535897930;
    const float Pi05 = 1.5707963267948966;

    if (x == 0.0 || abs(x) >= 1.0) return x;

    bool sgn = x < 0.0;
    x = abs(x);

    float s = sin_thVN;

    #if 1
    float o = s - s * s;
    float slp0 = 1.0 / (1.0 + (Pi  - 1.0) * (s - o * 0.30546));
    float slp1 = 1.0 / (1.0 - (1.0 - exp2(-20.0)) * (s + o * mix(0.5, 0.785, s)));

    float k = mix(0.1, 0.25, s);
    #else
    // slightly better, but also slower
    float angVN = ASin01_Approx(s);
    float c = cos(angVN);

    float slp0 = 1.0 / (c + s * (angVN  + Pi05));
    float slp1 = 1.0 / (c + s * (abs(angVN) - Pi05));

    float sb = s + (s - s*s) * -0.1;
    float cb = c + (c - c*c) * -0.;

    float k = sb * cb * 0.494162;
    #endif

    float a = 1.0 - (Pi - 2.0) / (Pi - 1.0);
    float b = 1.0 / (Pi - 1.0);

    float d0 =   a - slp0 * b;
    float d1 = 1.0 - slp1;

    float f0 = d0 * (Pi * x - ASin01_Approx(x));
    float f1 = d1 * (x - 1.0);

    float kk = k * k;

    float h0 = sqrt(f0*f0 + kk) - k;
    float h1 = sqrt(f1*f1 + kk) - k;

    float hh = (h0 * h1) / (h0 + h1);

    float y = x - sqrt(hh*(hh + 2.0*k));

    return sgn ? -y : y;
}

// vvsN: view vec space normal | rnd01: [0, 1]
vec2 SamplePartialSliceDir(vec3 vvsN, float rnd01) {
    float ang0 = rnd01 * Pi2;

    vec2 dir0 = vec2(cos(ang0), sin(ang0));

    float l = length(vvsN.xy);

    if (l == 0.0) return dir0;

    vec2 n = vvsN.xy / l;

    // align n with x-axis
    dir0 = cmul(dir0, n * vec2(1.0, -1.0));

    // sample slice angle
    float ang;
    {
        float x = ArcTan11(dir0);
        float sinNV = l;

        ang = SamplePartialSlice(x, sinNV) * Pi;
    }

    // ray space slice direction
    vec2 dir = vec2(cos(ang), sin(ang));

    // align x-axis with n
    dir = cmul(dir, n);

    return dir;
}

//==================================================================================//
//////////////////////////////////////////////////////////////////////////////////////


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

// returns 4 uniformly distributed rnd numbers [0,1]
// rnd01.x/rnd01.xy -> used to sample a slice direction (exact importance sampling needs 2 rnd numbers)
// rnd01.z -> used to jitter sample positions along ray marching direction
// rnd01.w -> used to jitter sample positions radially around slice normal
vec4 Rnd01x4(vec2 uv, uint n) {
    uvec2 uvu = uvec2(uv);

    vec4 rnd01 = vec4(0.0);

    rnd01.x  = rand_IGN(uv, n);
    rnd01.z  = rand_IGN(uv, n + 2);
    rnd01.w  = rand_IGN(uv, n + 3);

    return rnd01;
}

// https://graphics.stanford.edu/%7Eseander/bithacks.html#CountBitsSetParallel | license: public domain
uint CountBits(uint v) {
    return bitCount(v);
}

float SliceRelCDF_Cos(float x, float angN, float cosN, bool isPhiLargerThanAngN) {
    if (x <= 0.0 || x >= 1.0) return x;

    float phi = x * Pi - Pi05;

    bool c = isPhiLargerThanAngN;

    float n0 = c ?  3.0 : 1.0;
    float n1 = c ? -1.0 : 1.0;
    float n2 = c ?  4.0 : 0.0;

    float t0 = n0 * cosN + n1 * cos(angN - 2.0 * phi) + (n2 * angN + (n1 * 2.0) * phi + Pi) * sin(angN);
    float t1 = 4.0 * (cosN + angN * sin(angN));

    return t0 / t1;
}

float SliceRelCDF_Cos(float x, float angN, float cosN) {
    if (x <= 0.0 || x >= 1.0) return x;

    float phi = x * Pi - Pi05;

    bool c = phi > angN;

    float n0 = c ?  3.0 : 1.0;
    float n1 = c ? -1.0 : 1.0;
    float n2 = c ?  4.0 : 0.0;

    float t0 = n0 * cosN + n1 * cos(angN - 2.0 * phi) + (n2 * angN + (n1 * 2.0) * phi + Pi) * sin(angN);
    float t1 = 4.0 * (cosN + angN * sin(angN));

    return t0 / t1;
}


void uniGTVBGI(vec3 wpos, vec3 normalWS) {
    vec3 positionVS = (gbufferModelView * vec4(wpos, 1.0)).xyz;
    vec3 normalVS   = mat3(gbufferModelView) * normalWS;

    vec3 V = -normalize(positionVS);

    vec2 rayStart = SPos_from_VPos(positionVS).xy;

    uvec2 hashKey = (uvec2(gl_FragCoord.xy) & uvec2(31u)) ^ (NOISE_FRAME & 0xFFFFFFF0u);
    uint r2Index = (rand_hash21(hashKey) & 65535u) + NOISE_FRAME;
    float baseSampleLod = rand_r2Seq1(r2Index);

    vec4 rnd01 = Rnd01x4(gl_FragCoord.xy, NOISE_FRAME);

    ////////////////////////////////////////////////// slice direction sampling
    vec3 smplDirVS;// view space sampling vector
    vec2 dir;// screen space sampling vector
    {
        // approximate partial slice dir importance sampling via 1 rnd number

        // set up View Vec Space <-> View Space mapping
        vec4   Q_toV = GetQuaternion(V);
        vec4 Q_fromV = Q_toV * vec4(vec3(-1.0), 1.0);// conjugate

        vec3 normalVVS = normalVS;

        normalVVS = Transform_Qz0(normalVS, Q_fromV);

        dir = SamplePartialSliceDir(normalVVS, rnd01.x);

        smplDirVS = vec3(dir.xy, 0.0);

        smplDirVS = Transform_Vz0Qz0(dir, Q_toV);

        vec3 rayStart = SPos_from_VPos(positionVS);
        vec3 rayEnd   = SPos_from_VPos(positionVS + smplDirVS*(near*0.5));

        vec3 rayDir   = rayEnd - rayStart;

        rayDir /= length(rayDir.xy);

        dir = rayDir.xy;
    }
    //////////////////////////////////////////////////

    ////////////////////////////////////////////////// construct slice
    vec3 sliceN, projN, T;
    float cosN, angN, projNRcpLen;
    {
        sliceN = cross(V, smplDirVS);

        projN = normalVS - sliceN * dot(normalVS, sliceN);

        float projNSqrLen = dot(projN, projN);
        if (projNSqrLen == 0.0) return;

        projNRcpLen = inversesqrt(projNSqrLen);

        cosN = dot(projN, V) * projNRcpLen;

        T = cross(sliceN, projN);

        float sgn = dot(V, T) < 0.0 ? -1.0 : 1.0;

        angN = sgn * ACos(cosN);
    }
    //////////////////////////////////////////////////

    float angOff = angN * RcpPi + 0.5;

    // percentage of the slice we don't use ([0, angN]-integrated slice-relative pdf)
    float w0 = clamp((sin(angN) / (cos(angN) + angN * sin(angN))) * (Pi/4.0) + 0.5, 0.0, 1.0);

    // partial slice re-mapping constants
    float w0_remap_mul = 1.0 / (1.0 - w0);
    float w0_remap_add = -w0 * w0_remap_mul;

    vec2 rayDir = dir.xy;

    float maxDistX = rayDir.x != 0.0 ? (rayDir.x >= 0.0 ? (global_mainImageSize.x - gl_FragCoord.x) / rayDir.x : -gl_FragCoord.x / rayDir.x) : 1e6;
    float maxDistY = rayDir.y != 0.0 ? (rayDir.y < 0.0 ? (-gl_FragCoord.y / rayDir.y) : (global_mainImageSize.y - gl_FragCoord.y) / rayDir.y) : 1e6;
    float maxDist = min(maxDistX, maxDistY);

    float lodStep = radiusToLodStep(maxDist);
    float sampleLod = lodStep * baseSampleLod;

    float sampleTexelDist = 0.5;

    uint occBits = 0u;

    for (uint stepIndex = 0; stepIndex < SSVBIL_SAMPLE_STEPS; ++stepIndex) {
        float sampleLodTexelSize = lodTexelSize(sampleLod) * 1.0;
        float stepTexelSize = sampleLodTexelSize * 0.5;
        sampleTexelDist += stepTexelSize;

        vec2 sampleTexelCoord = floor(rayDir * sampleTexelDist + rayStart) + 0.5;
        vec2 sampleUV = sampleTexelCoord / textureSize(usam_viewZ, 0).xy;

        float realSampleLod = round(sampleLod * 0.5);

        float sampleViewZ = textureLod(usam_viewZ, sampleUV, realSampleLod).r;

        vec3 samplePosVS = coords_toViewCoord(sampleUV, sampleViewZ, gbufferProjectionInverse);

        vec3 deltaPosFront = samplePosVS - positionVS;
        vec3 deltaPosBack  = deltaPosFront - V * SETTING_SSVBIL_THICKNESS;

        #if 1
        // required for correctness, but probably not worth to keep active in a practical application:
        #if 1
        deltaPosBack =  coords_toViewCoord(sampleUV, sampleViewZ - SETTING_SSVBIL_THICKNESS, gbufferProjectionInverse) - positionVS;
        #else
        //                                 also valid, but not consistent with reference ray marcher
        deltaPosBack = deltaPosFront + normalize(samplePosVS) * SETTING_SSVBIL_THICKNESS;
        #endif
        #endif

        // project samples onto unit circle and compute angles relative to V
        vec2 horCos = vec2(dot(normalize(deltaPosFront), V),
            dot(normalize(deltaPosBack), V));

        vec2 horAng = ACos(horCos);

        // shift relative angles from V to N + map to [0,1]
        vec2 hor01 = clamp(horAng * RcpPi + angOff, 0.0, 1.0);

        // map to slice relative distribution
        hor01.x = SliceRelCDF_Cos(hor01.x, angN, cosN, true);
        hor01.y = SliceRelCDF_Cos(hor01.y, angN, cosN, true);

        // partial slice re-mapping
        hor01 = hor01 * w0_remap_mul + w0_remap_add;

        // jitter sample locations + clamp01
        hor01 = clamp(hor01 + rnd01.w * (1.0 / 32.0), 0.0, 1.0);

        uint occBits0;// turn arc into bit mask
        {
            uvec2 horInt = uvec2(floor(hor01 * 32.0));

            uint OxFFFFFFFFu = 0xFFFFFFFFu;// don't inline here! ANGLE bug: https://issues.angleproject.org/issues/353039526

            uint mX = horInt.x < 32u ? OxFFFFFFFFu <<        horInt.x  : 0u;
            uint mY = horInt.y != 0u ? OxFFFFFFFFu >> (32u - horInt.y) : 0u;

            occBits0 = mX & mY;
        }

        // compute gi contribution
        {
            uint visBits0 = occBits0 & (~occBits);

            if (visBits0 != 0u) {
                vec4 sample1 = textureLod(usam_temp1, sampleUV, realSampleLod);
                float emissive = float(sample1.a > 0.0);
                vec3 N0 = mix(sample1.rgb, normalize(positionVS - samplePosVS), emissive);
                vec3 projN0 = N0 - sliceN * dot(N0, sliceN);

                float projN0SqrLen = dot(projN0, projN0);

                if (projN0SqrLen != 0.0){
                    float projN0RcpLen = inversesqrt(projN0SqrLen);

                    bool flipT = dot(T, N0) < 0.0;

                    float u = dot(projN, projN0);
                    u *= projNRcpLen;
                    u *= projN0RcpLen;

                    float hor01 = ACos(u) * RcpPi;
                    if (flipT) hor01 = 1.0 - hor01;

                    // map to slice relative distribution
                    hor01 = SliceRelCDF_Cos(hor01, angN, cosN);

                    // partial slice re-mapping
                    hor01 = hor01 * w0_remap_mul + w0_remap_add;

                    // jitter sample locations + clamp01
                    hor01 = clamp(hor01 + rnd01.w * (1.0 / 32.0), 0.0, 1.0);

                    uint visBitsN;// turn arc into bit mask
                    {
                        uint horInt = uint(floor(hor01 * 32.0));
                        visBitsN = horInt < 32u ? 0xFFFFFFFFu << horInt : 0u;
                        if (!flipT) visBitsN = ~visBitsN;
                    }

                    visBits0 = visBits0 & visBitsN;
                }

                if (visBits0 != 0u) {
                    vec4 sample2 = textureLod(usam_temp2, sampleUV, realSampleLod);
                    vec3 sampleRad = sample2.rgb;
                    float vis0 = float(CountBits(visBits0)) * (1.0 / 32.0);
                    rt_out.rgb += sampleRad * vis0;
                }
            }
        }

        occBits = occBits | occBits0;

        sampleLod = sampleLod + lodStep;
        sampleTexelDist += stepTexelSize;
    }

    mat3 viewToScene = mat3(gbufferModelViewInverse);

    uint aoSectionBits = ~occBits;
    vec3 realTangent = normalize(T);

    vec3 skyLighting = vec3(0.0);
    for (uint i = 0u; i < 4u; i++) {
        float fi = float(i);

        float ang0 = 0.25 * fi;

        // shift relative angles from V to N + map to [0,1]
        vec2 hor01 = vec2(ang0, ang0 + 0.25);

        // map to slice relative distribution
        hor01.x = SliceRelCDF_Cos(hor01.x, angN, cosN, true);
        hor01.y = SliceRelCDF_Cos(hor01.y, angN, cosN, true);

        // partial slice re-mapping
        hor01 = hor01 * w0_remap_mul + w0_remap_add;

        // jitter sample locations + clamp01
        hor01 = clamp(hor01 + rnd01.w * (1.0 / 32.0), 0.0, 1.0);

        uint sectorBitMask;// turn arc into bit mask
        {
            uvec2 horInt = uvec2(floor(hor01 * 32.0));

            uint OxFFFFFFFFu = 0xFFFFFFFFu;// don't inline here! ANGLE bug: https://issues.angleproject.org/issues/353039526

            uint mX = horInt.x < 32u ? OxFFFFFFFFu <<        horInt.x  : 0u;
            uint mY = horInt.y != 0u ? OxFFFFFFFFu >> (32u - horInt.y) : 0u;

            sectorBitMask = mX & mY;
        }

        uint sectorBits = (aoSectionBits & sectorBitMask);
        float bitCount = float(bitCount(sectorBits)) * (1.0 / 32.0);

        float angC = 0.125 + 0.25 * fi;
        float cosC = cos(angC);
        float sinC = sin(angC);

        vec3 skyNormal = viewToScene * normalize(normalVS * cosC + realTangent * sinC);
        vec2 skyLUTUV = coords_polarAzimuthEqualArea(skyNormal);
        vec3 skyRadiance = texture(usam_skyLUT, skyLUTUV).rgb;
        skyLighting += bitCount * skyRadiance;
    }

    // compute AO
    rt_out.a = float(CountBits(occBits)) * (1.0 / 32.0);
    rt_out.a = saturate(1.0 - rt_out.a);
    rt_out.a = pow(rt_out.a, SETTING_SSVBIL_AO_STRENGTH);

    rt_out.rgb *= PI;
    rt_out.rgb *= SETTING_SSVBIL_GI_STRENGTH;

    ivec2 intTexelPos = ivec2(gl_FragCoord.xy);

    float lmCoordSky = texelFetch(usam_temp2, intTexelPos, 0).a;
    float skyLightingIntensity = lmCoordSky * lmCoordSky;
    skyLightingIntensity *= rt_out.a * rt_out.a;
    skyLightingIntensity *= SETTING_SKYLIGHT_STRENGTH;

    rt_out.rgb += skyLighting * skyLightingIntensity;
}

void main() {
    Resolution = global_mainImageSize.xy;

    ivec2 intTexelPos = ivec2(gl_FragCoord.xy);
    float centerViewZ = texelFetch(usam_viewZ, intTexelPos, 0).r;

    rt_out = vec4(0.0);
    if (centerViewZ < 0.0) {
        vec3 N = texelFetch(usam_temp1, intTexelPos, 0).rgb;
        N = mat3(gbufferModelViewInverse) * N;

        vec3 wpos = coords_toViewCoord(frag_texCoord, centerViewZ, gbufferProjectionInverse);
        wpos = (gbufferModelViewInverse * vec4(wpos, 1.0)).xyz;
        wpos += N * (1.0 / 1024.0);

        uniGTVBGI(wpos, N);
    }
}