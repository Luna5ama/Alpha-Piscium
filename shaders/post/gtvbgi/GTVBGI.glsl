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


/*
    This work is licensed under a dual license, public domain and MIT, unless noted otherwise. Choose the one that best suits your needs:

    CC0 1.0 Universal https://creativecommons.org/publicdomain/zero/1.0/
    To the extent possible under law, the author has waived all copyrights and related or neighboring rights to this work.

    or

    The MIT License
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
    DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
    THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

////////////////////////////////////////////////////////////////////////////////////// config
//==================================================================================//

/*
    toggles:

    Tab  : toggle temporal accumulation on/off
    Ctrl : toggle ortho cam off/on
    Space: toggle between cosine and uniform hemisphere weighting (changes appearance of UI | default: cosine)

    U    : toggle UI off/on
*/

/*
    1: uniform sampling + slice weighting (works for gi almost as well as 3)
    2: exact  importance sampling via 2 random numbers
    3: approx importance sampling via 1 random number (default)
    -> only used if cosine weighting is active (otherwise uniform sampling is used)
*/
#define GTVBGI_SLICE_SAMPLING_MODE 3

/*
    1: GTAOFastAcos         : fastest; good enough
    2: improved GTAOFastAcos: slightly slower than 1, but quite a bit better
    3: acos                 : very slow, but very accurate (useful for testing/debugging)
*/
#define ACOS_QUALITY_MODE 2

#define USE_HQ_APPROX_SLICE_IMPORTANCE_SAMPLING

//==================================================================================//
//////////////////////////////////////////////////////////////////////////////////////


bool isLeft;// true on the left half of the screen; for debugging purposes

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
float ArcTan(vec2 dir)
{
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

float ACosPoly(float x)
{
    #if ACOS_QUALITY_MODE == 1
    // GTAOFastAcos
    return 1.5707963267948966 - 0.1565827644218014 * x;
    #else
    // higher quality version of GTAOFastAcos (for the cost of one additional mad)
    // minimizes max abs(ACos_Approx(cos(x)) - x)
    return 1.5707963267948966 + (-0.20491203466059038 + 0.04832927023878897 * x) * x;
    #endif
}

float ACos_Approx(float x)
{
    float u = ACosPoly(abs(x)) * sqrt(1.0 - abs(x));

    return x >= 0.0 ? u : Pi - u;
}

float ACos01_Approx(float x)// x: [0,1]
{
    return ACosPoly(x) * sqrt(1.0 - x);
}

float ASin_Approx(float x)
{
    float u = ACosPoly(abs(x)) * sqrt(1.0 - abs(x)) - 1.5707963267948966;

    return x >= 0.0 ? -u : u;
}

float ASin01_Approx(float x)// x: [0,1]
{
    return 1.5707963267948966 - ACosPoly(x) * sqrt(1.0 - x);
}

#if ACOS_QUALITY_MODE == 1 || ACOS_QUALITY_MODE == 2
float ACos(float x)
{
    return ACos_Approx(clamp(x, -1.0, 1.0));
}
#else
float ACos(float x)// very slow; for debugging
{
    return acos(clamp(x, -1.0, 1.0));
}
#endif

vec2 ACos(vec2 v)
{
    return vec2(ACos(v.x), ACos(v.y));
}

//==================================================================================//
//////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////// partial slice sampling
//==================================================================================//

vec2 cmul(vec2 c0, vec2 c1)
{
    return vec2(c0.x * c1.x - c0.y * c1.y,
        c0.y * c1.x + c0.x * c1.y);
}

float SamplePartialSlice(float x, float sin_thVN)
{
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
vec2 SamplePartialSliceDir(vec3 vvsN, float rnd01)
{
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

vec4 GetQuaternion(vec3 from, vec3 to)
{
    vec3 xyz = cross(from, to);
    float s  =   dot(from, to);

    float u = inversesqrt(max(0.0, s * 0.5 + 0.5));// rcp(cosine half-angle formula)

    s    = 1.0 / u;
    xyz *= u * 0.5;

    return vec4(xyz, s);
}

vec4 GetQuaternion(vec3 to)
{
    //vec3 from = vec3(0.0, 0.0, 1.0);

    vec3 xyz = vec3(-to.y, to.x, 0.0);// cross(from, to);
    float s  =                   to.z;//   dot(from, to);

    float u = inversesqrt(max(0.0, s * 0.5 + 0.5));// rcp(cosine half-angle formula)

    s    = 1.0 / u;
    xyz *= u * 0.5;

    return vec4(xyz, s);
}

// transform v by unit quaternion q.xyzs
vec3 Transform(vec3 v, vec4 q)
{
    vec3 k = cross(q.xyz, v);

    return v + 2.0 * vec3(dot(vec3(q.wy, -q.z), k.xzy),
        dot(vec3(q.wz, -q.x), k.yxz),
        dot(vec3(q.wx, -q.y), k.zyx));
}

// transform v by unit quaternion q.xy0s
vec3 Transform_Qz0(vec3 v, vec4 q)
{
    float k = v.y * q.x - v.x * q.y;
    float g = 2.0 * (v.z * q.w + k);

    vec3 r;
    r.xy = v.xy + q.yx * vec2(g, -g);
    r.z  = v.z  + 2.0 * (q.w * k - v.z * dot(q.xy, q.xy));

    return r;
}

// transform v.xy0 by unit quaternion q.xy0s
vec3 Transform_Vz0Qz0(vec2 v, vec4 q)
{
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
vec4 Rnd01x4(vec2 uv, uint n)
{
    uvec2 uvu = uvec2(uv);

    vec4 rnd01 = vec4(0.0);

    #if GTVBGI_SLICE_SAMPLING_MODE == 1 || GTVBGI_SLICE_SAMPLING_MODE == 3

    rnd01.x  = rand_IGN(uv, n);
    rnd01.z  = rand_IGN(uv, n + 1);
    rnd01.w  = rand_IGN(uv, n + 2);

    #elif GTVBGI_SLICE_SAMPLING_MODE == 2

    rnd01   = BlueNoise01x4(uvu, n);
    rnd01.x = BlueNoise01  (uvu, n);

    #endif

    return rnd01;
}

// https://graphics.stanford.edu/%7Eseander/bithacks.html#CountBitsSetParallel | license: public domain
uint CountBits(uint v)
{
    v = v - ((v >> 1u) & 0x55555555u);
    v = (v & 0x33333333u) + ((v >> 2u) & 0x33333333u);
    return ((v + (v >> 4u) & 0xF0F0F0Fu) * 0x1010101u) >> 24u;
}

float SliceRelCDF_Cos(float x, float angN, float cosN, bool isPhiLargerThanAngN)
{
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

float SliceRelCDF_Cos(float x, float angN, float cosN)
{
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


vec4 uniGTVBGI(vec2 uv0, vec3 wpos, vec3 normalWS, uint dirCount)
{
    vec3 positionVS = (gbufferModelView * vec4(wpos, 1.0)).xyz;
    vec3 normalVS   = mat3(gbufferModelView) * normalWS;

    vec3 V = isPerspectiveCam ? -normalize(positionVS) : vec3(0.0, 0.0, -1.0);

    vec2 rayStart = SPos_from_VPos(positionVS).xy;

//    uint frame = USE_TEMP_ACCU_COND ? uint(frameCounter) : 0u;
    uint frame = uint(frameCounter);

    vec3 gi = vec3(0.0);
    float ao = 0.0;
    for (uint i = 0u; i < dirCount; ++i)
    {
        uint n = frame * dirCount + i;
        vec4 rnd01 = Rnd01x4(uv0, n);

        ////////////////////////////////////////////////// slice direction sampling
        vec3 smplDirVS;// view space sampling vector
        vec2 dir;// screen space sampling vector
        {
            #if GTVBGI_SLICE_SAMPLING_MODE == 1
            // sample partial slice dir uniformly and later compute slice_weight accordingly

            float rndAng = rnd01.x * Pi2;

            dir = vec2(cos(rndAng), sin(rndAng));

            smplDirVS = vec3(dir.xy, 0.0);

            if (isPerspectiveCam)
            {
                // set up View Vec Space -> View Space mapping
                vec4 Q_toV = GetQuaternion(V);

                smplDirVS = Transform_Vz0Qz0(dir, Q_toV);

                vec3 rayStart = SPos_from_VPos(positionVS);
                vec3 rayEnd   = SPos_from_VPos(positionVS + smplDirVS*(near * 0.5));

                vec3 rayDir   = rayEnd - rayStart;

                rayDir /= length(rayDir.xy);

                dir = rayDir.xy;
            }

            #elif GTVBGI_SLICE_SAMPLING_MODE == 2
            // exact partial slice dir importance sampling via 2 rnd numbers
            // sample cos lobe in world space and project dir to screen space to be used as partial slice dir

            vec2 s = rnd01.xy * 2.0 - 1.0;

            vec3 cosDir;
            {
                #if 1
                // align up-axis of local sampling frame with view vector
                // -> results in better noise profile when looking straight at a surface
                vec3 sph;// = Sample_Sphere(s); inlined here to ensure z is up
                {
                    float ang = Pi * s.x;
                    float s1p = sqrt(1.0 - s.y*s.y);

                    sph = vec3(vec2(cos(ang), sin(ang)) * s1p, s.y);
                }

                if (isPerspectiveCam)
                {
                    sph = Transform_Qz0(sph, GetQuaternion(V));
                }

                vec3 cosDirVS = normalize(sph + VVec_from_WVec(normalWS));
                cosDir   = WVec_from_VVec(cosDirVS);

                smplDirVS = cosDirVS;

                #else

                cosDir = normalize(Sample_Sphere(s) + normalWS);

                smplDirVS = VVec_from_WVec(cosDir);

                #endif
            }

            vec3 rayDir = smplDirVS;

            if (isPerspectiveCam)
            {
                rayDir = SPos_from_WPos(wpos + cosDir * (near * 0.5)) - SPos_from_WPos(wpos);
            }

            // 1 px step size
            rayDir /= length(rayDir.xy);

            dir = rayDir.xy;

            // make orthogonal to V (alternatively use sliceN = normalize(sliceN);)
            smplDirVS = normalize(smplDirVS - V * dot(V, smplDirVS));

            #elif GTVBGI_SLICE_SAMPLING_MODE == 3
            // approximate partial slice dir importance sampling via 1 rnd number

            // set up View Vec Space <-> View Space mapping
            vec4   Q_toV = GetQuaternion(V);
            vec4 Q_fromV = Q_toV * vec4(vec3(-1.0), 1.0);// conjugate

            vec3 normalVVS = normalVS;

            if (isPerspectiveCam) normalVVS = Transform_Qz0(normalVS, Q_fromV);

            dir = SamplePartialSliceDir(normalVVS, rnd01.x);

            smplDirVS = vec3(dir.xy, 0.0);

            if (isPerspectiveCam)
            {
                smplDirVS = Transform_Vz0Qz0(dir, Q_toV);

                vec3 rayStart = SPos_from_VPos(positionVS);
                vec3 rayEnd   = SPos_from_VPos(positionVS + smplDirVS*(near*0.5));

                vec3 rayDir   = rayEnd - rayStart;

                rayDir /= length(rayDir.xy);

                dir = rayDir.xy;
            }
            #endif
        }
        //////////////////////////////////////////////////

        ////////////////////////////////////////////////// construct slice
        vec3 sliceN, projN, T;
        float cosN, angN, projNRcpLen;
        {
            sliceN = cross(V, smplDirVS);

            projN = normalVS - sliceN * dot(normalVS, sliceN);

            float projNSqrLen = dot(projN, projN);
            if (projNSqrLen == 0.0) return vec4(0.0, 0.0, 0.0, 1.0);

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

        vec3 gi0 = vec3(0.0);
        float ao0 = 0.0;
        {
            vec2 rayDir = dir.xy;

            const float s = pow(Raymarching_Width, 1.0 / SSVBIL_SAMPLE_STEPS);

            float t = pow(s, rnd01.z);// init t: [1, s]

            uint occBits = 0u;
            for (uint i = 0; i < SSVBIL_SAMPLE_STEPS; ++i) {
                vec2 samplePos = rayStart + rayDir * t;

                t *= s;

                // handle oob
                if (samplePos.x < 0.0 || samplePos.x >= global_mainImageSize.x || samplePos.y < 0.0 || samplePos.y >= global_mainImageSize.y) break;

                vec2 sampleUV = samplePos * global_mainImageSizeRcp;
                float realSampleLod = 0.0;
                //                vec4 buff = textureLod(iChannel2, samplePos / Resolution.xy, 0.0);
                //if(frameCounter != 0) buff.rgb = textureLod(iChannel0, samplePos / Resolution.xy, 0.0).rgb;// recursive bounces hack

                float sampleViewZ = textureLod(usam_viewZ, sampleUV, realSampleLod).r;

                vec3 samplePosVS = coords_toViewCoord(sampleUV, sampleViewZ, gbufferProjectionInverse);

                vec3 deltaPosFront = samplePosVS - positionVS;
                vec3 deltaPosBack  = deltaPosFront - V * SETTING_SSVBIL_THICKNESS;

                //                #if 1
                //                // required for correctness, but probably not worth to keep active in a practical application:
                //                if(isPerspectiveCam)
                //                {
                //                    #if 1
                //                   deltaPosBack =  coords_toViewCoord(sampleUV, sampleViewZ + Thickness, gbufferProjectionInverse) - positionVS;
                //                    #else
                //                    // also valid, but not consistent with reference ray marcher
                //                    deltaPosBack = deltaPosFront + normalize(samplePosVS) * Thickness;
                //                    #endif
                //                }
                //                #endif

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
                hor01 = clamp(hor01 + rnd01.w * (1.0/32.0), 0.0, 1.0);

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

                    if (visBits0 != 0u)
                    {
                        vec4 sample2 = textureLod(usam_temp2, sampleUV, realSampleLod);
                        vec3 sampleRad = sample2.rgb;

//                        #ifdef USE_BACKFACE_REJECTION
                                                #if 1
//                        #ifndef GTVBGI_USE_SIMPLE_HEURISTIC_FOR_BACKFACE_REJECTION
                        #if 1
                        {

                            //                            vec3 N0 = textureLod(iChannel3, samplePos / Resolution.xy, 0.0).xyz;
                            //                            N0 = VVec_from_WVec(N0);
                            vec4 sample1 = textureLod(usam_temp1, sampleUV, realSampleLod);
                        float emissive = float(sample1.a > 0.0);
                        vec3 N0 = mix(sample1.rgb, normalize(positionVS - samplePosVS), emissive);
//                            vec3 N0 = sample1.rgb;

                            vec3 projN0 = N0 - sliceN * dot(N0, sliceN);

                            float projN0SqrLen = dot(projN0, projN0);

                            if (projN0SqrLen != 0.0)
                            {
                                float projN0RcpLen = inversesqrt(projN0SqrLen);

                                bool flipT = dot(T, N0) < 0.0;

                                float u = dot(projN, projN0);
                                u *= projNRcpLen;
                                u *= projN0RcpLen;

                                #if 1

                                float hor01 = ACos(u) * RcpPi;

                                if (flipT) hor01 = 1.0 - hor01;

                                #else

                                // same as above but allows to skip sign handling in ACos (prob not worth it)
                                float hor01 = ACos(abs(u)) * RcpPi;

                                if (flipT != (u < 0.0)) hor01 = 1.0 - hor01;

                                #endif

                                // map to slice relative distribution
                                hor01 = SliceRelCDF_Cos(hor01, angN, cosN);

                                // partial slice re-mapping
                                hor01 = hor01 * w0_remap_mul + w0_remap_add;

                                // jitter sample locations + clamp01
                                hor01 = clamp(hor01 + rnd01.w * (1.0/32.0), 0.0, 1.0);

                                uint visBitsN;// turn arc into bit mask
                                {
                                    uint horInt = uint(floor(hor01 * 32.0));

                                    visBitsN = horInt < 32u ? 0xFFFFFFFFu << horInt : 0u;

                                    if (!flipT) visBitsN = ~visBitsN;
                                }

                                visBits0 = visBits0 & visBitsN;
                            }
                        }
                        #endif
                        #endif

                        float vis0 = float(CountBits(visBits0)) * (1.0/32.0);

//                        #ifdef USE_BACKFACE_REJECTION
//                        #ifdef GTVBGI_USE_SIMPLE_HEURISTIC_FOR_BACKFACE_REJECTION
//                        {
//                            vec4 sample1 = textureLod(usam_temp1, sampleUV, realSampleLod);
//                            vec3 N0 = sample1.rgb;
//
//                            vec3 projN0 = N0 - sliceN * dot(N0, sliceN);
//
//                            float projN0SqrLen = dot(projN0, projN0);
//
//                            if (projN0SqrLen != 0.0)
//                            {
//                                float projN0RcpLen = inversesqrt(projN0SqrLen);
//
//                                bool flipT = dot(T, N0) < 0.0;
//
//                                float u = dot(projN, projN0);
//                                u *= projNRcpLen;
//                                u *= projN0RcpLen;
//
//                                float v = u * -0.5 + 0.5;
//
//                                vis0 *= clamp(v * 4.0 + 0., 0.0, 1.0);// tune mapping for use case
//                            }
//                        }
//                        #endif
//                        #endif

                        gi0 += sampleRad * vis0;
                    }
                }

                occBits = occBits | occBits0;
            }

            // compute GI/AO contribution
            {
                float occ0 = float(CountBits(occBits)) * (1.0/32.0);
                float w;

                #if GTVBGI_SLICE_SAMPLING_MODE == 1
                w = 2.0 - 2.0 * w0;
                #else
                w = 1.0;
                #endif

                ao0 = w - w * occ0;
                gi0 =     w *  gi0;
            }
        }

        // accumulate AO contribution
        {
            float slice_weight = 1.0;

            #if GTVBGI_SLICE_SAMPLING_MODE == 1
            // if we sample the partial slice dir from a uniform distribution we need to account for that here
            slice_weight = max(0.0, 1.0/projNRcpLen * (cosN + angN * sin(angN)));
            #endif

            ao += ao0 * slice_weight;
            gi += gi0 * slice_weight;
        }
    }

    float norm = 1.0 / float(dirCount);
    //return vec4(ao) * norm;
    return vec4(gi, ao) * norm;
}

//==================================================================================//
//////////////////////////////////////////////////////////////////////////////////////

//vec3 Read_Albedo(vec2 uv0)
//{
//    // in a practical setting we would do a buffer read here, but because of shadertoy's
//    // limitations it is easier and cleaner to just re-render the scene here once to get the albedo
//    vec4 mouseAccu = ReadVar4(1, VAR_ROW);
//
//    uvec2 uv = uvec2(uv0.xy - 0.5);
//    vec2 tex = uv0.xy / global_mainImageSize.xy;
//    vec2 tex21 = tex * 2.0 - vec2(1.0);
//
//    PrepareCam(mouseAccu, USE_PERSPECTIVE_CAM_COND);
//
//    vec3 rp, rd;
//    GetRay(uv0, /*out*/ rp, rd);
//
//    float t; vec3 N, a;
//    Intersect_Scene(rp, rd, /*out:*/ t, N, a);
//
//    return a;
//    return a;
//}

void mainImage(out vec4 outCol, in vec2 uv0)
{
    Resolution = global_mainImageSize.xy;
    isLeft = uv0.x < global_mainImageSize.x * 0.5;

//    vec4 mouseAccu  = ReadVar4(1, VAR_ROW);
//    float frameAccu = ReadVar (3, VAR_ROW);

    ivec2 intTexelPos = ivec2(gl_FragCoord.xy);

    vec3 N = texelFetch(usam_temp1, intTexelPos, 0).rgb;
    N = mat3(gbufferModelViewInverse) * N;
    float centerViewZ = texelFetch(usam_viewZ, intTexelPos, 0).r;

    if (centerViewZ >= 0.0) {
        outCol = vec4(0.0);
        return;
    }

    vec3 wpos = coords_toViewCoord(frag_texCoord, centerViewZ, gbufferProjectionInverse);
    wpos = (gbufferModelViewInverse * vec4(wpos, 1.0)).xyz;
    wpos += N * (1.0/1024.0);

    vec3 col = vec3(0.0);

    vec3 gi;
    {
        uint count = 1u;

        gi = uniGTVBGI(uv0, wpos, N, count).rgb;
    }

    //    col = rad + a * gi;
    col = gi;

    //    // accumulate frames
    //    if(USE_TEMP_ACCU_COND)
    //    {
    //        vec2 tc = uv0.xy / global_mainImageSize.xy;
    //
    //        vec4 colLast = textureLod(iChannel0, tc, 0.0);
    //
    //        col = mix(colLast.rgb, col, 1.0 / (frameAccu));
    //
    //        frameAccu += 1.0;
    //
    //        outCol = vec4(col.rgb, frameAccu);
    //
    //        return;
    //    }

    outCol = vec4(col, 1.0);
}

void main() {
    mainImage(rt_out, gl_FragCoord.xy);
}