// Adopted from: https://www.shadertoy.com/view/XcdBWf
// MIT License
// You can find full license texts in /licenses
#include "../../_Util.glsl"

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

// affects both GT-VBGI and the reference GI (if USE_ANALYTICAL_RAYCASTING is not active in Buffer D)
//#define USE_BACKFACE_REJECTION

const float Raymarching_Width = 2000.0;

//==================================================================================//
//////////////////////////////////////////////////////////////////////////////////////

vec2 Resolution;

#define USE_TEMP_ACCU_COND                true
#define USE_UNIFORM_HEMISHPHERE_WEIGHTING false
#define USE_PERSPECTIVE_CAM_COND          true
#define SHOW_UI_COND                      false

////////////////////////////////////////////////////////////////////////////////////////////////////////////// misc
//==========================================================================================================//
#define VAR_ROW 4

/* http://keycode.info/ */
#define KEY_LEFT  37
#define KEY_UP    38
#define KEY_RIGHT 39
#define KEY_DOWN  40

#define KEY_TAB 9
#define KEY_CTRL 17
#define KEY_ALT 18
#define KEY_SHIFT 16
#define KEY_SPACE 32

#define KEY_Q 81
#define KEY_W 87
#define KEY_E 69
#define KEY_R 82
#define KEY_T 84

#define KEY_A 65
#define KEY_S 83
#define KEY_D 68
#define KEY_F 70
#define KEY_G 71

#define KEY_N1 49
#define KEY_N2 50
#define KEY_N3 51
#define KEY_N4 52
#define KEY_N5 53
#define KEY_N6 54
#define KEY_N7 55
#define KEY_N8 56

#define KEY_M 77
#define KEY_N 78
#define KEY_O 79
#define KEY_U 85

#define clamp01(x) clamp(x, 0.0, 1.0)
#define If(cond, resT, resF) mix(resF, resT, cond)

const float Pi = 3.1415926535897930;
const float Pi05 = 1.5707963267948966;
const float Pi2 = Pi * 2.0;
const float RcpPi = 1.0 / Pi;
const float RcpPi05 = 1.0 / Pi05;

float Pow2(float x) {return x*x;}
float Pow3(float x) {return x*x*x;}
float Pow4(float x) {return Pow2(Pow2(x));}

//==========================================================================================================//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////////////// camera logic + transforms
//==========================================================================================================//
bool isPerspectiveCam = true;// ortho/proj cam toggle


// ====== View <-> World ====== //
vec3 VPos_from_WPos(vec3 wpos) {
    return (gbufferModelView * vec4(wpos, 1.0)).xyz;
}

vec3 VVec_from_WVec(vec3 wvec) {
    return mat3(gbufferModelView) * wvec;
}

////

vec3 WPos_from_VPos(vec3 vpos) {
    return (gbufferModelViewInverse * vec4(vpos, 1.0)).xyz;
}

vec3 WVec_from_VVec(vec3 vvec) {
    return mat3(gbufferModelViewInverse) * vvec;
}
// =========================== //

// ====== Proj <-> View ====== //
vec4 PPos_from_VPos(vec3 vpos) {
    return gbufferProjection * vec4(vpos, 1.0);
}

vec3 VPos_from_PPos(vec4 ppos) {
    vec4 vpos = gbufferProjectionInverse * ppos;

    return vpos.xyz / vpos.w;
}

vec4 PVec_from_VVec(vec3 vvec) {
    return gbufferProjection * vec4(vvec, 0.0);
}
// =========================== //


// ====== Screen <-> View ====== //
vec3 SPos_from_VPos(vec3 vpos) {
    vec4 ppos = PPos_from_VPos(vpos);

    vec2 tc21 = ppos.xy / ppos.w;

    vec2 uv0 = (tc21 * 0.5 + 0.5) * Resolution;

    return vec3(uv0, vpos.z);
}
////

vec3 VPos_from_SPos(vec3 spos) {
//    vec2 uv0 = spos.xy;
//    float depth = spos.z;
//    depth = NonLinDepth_from_LinDepth(depth);
//
//    vec2 tc21 = uv0 / Resolution * 2.0 - 1.0;
//
//    vec3 ppos = vec3(tc21, depth);
//
//    vec4 vpos = ipmat * vec4(ppos, 1.0);
//
//    vpos /= vpos.w;
//
//    return vpos.xyz;
    return coords_toViewCoord(spos.xy * global_mainImageSizeRcp, spos.z, gbufferProjectionInverse);
}
// ============================= //

// ====== Proj <-> World ====== //
vec4 PPos_from_WPos(vec3 wpos) {
    vec3 vpos = VPos_from_WPos(wpos);

    return PPos_from_VPos(vpos);
}

vec3 WPos_from_PPos(vec4 ppos) {
    vec3 vpos = VPos_from_PPos(ppos);

    return WPos_from_VPos(vpos);
}

vec4 PVec_from_WVec(vec3 wvec) {
    vec3 vvec = VVec_from_WVec(wvec);

    return PVec_from_VVec(vvec);
}
// =========================== //

// ====== Screen <-> World ====== //
vec3 SPos_from_WPos(vec3 wpos) {
    vec3 vpos = VPos_from_WPos(wpos);

    return SPos_from_VPos(vpos);
}

////

vec3 WPos_from_SPos(vec3 spos) {
    vec3 vpos = VPos_from_SPos(spos);

    return WPos_from_VPos(vpos);
}
// ============================== //

//void GetRay(vec2 uv0, out vec3 rp, out vec3 rd)
//{
//    vec3 spos = vec3(uv0, near);
//
//    vec3 vpos = VPos_from_SPos(spos);
//
//    if(isPerspectiveCam)
//    {
//        rp = cpos;
//        rd = WVec_from_VVec(normalize(vpos));
//    }
//    else
//    {
//        rp = WPos_from_VPos(vpos);
//        rd = cmat[2];
//    }
//}
//==========================================================================================================//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////////////// intersection routines
//==========================================================================================================//
/*
IN:
	rp		: ray start position
	rd		: ray direction (normalized)

	cp		: cube position
	cth		: cube thickness (cth = 0.5 -> unit cube)

OUT:
	t		: distances to intersection points (negative if in backwards direction)

EXAMPLE:
	vec2 t;
	float hit = Intersect_Ray_Cube(pos, dir, vec3(0.0), vec3(0.5), OUT t);
*/
float Intersect_Ray_Cube(
vec3 rp, vec3 rd,
vec3 cp, vec3 cth,
out vec2 t) {
    rp -= cp;

    vec3 m = 1.0 / -rd;
    vec3 o = If(lessThan(rd, vec3(0.0)), -cth, cth);

    vec3 uf = (rp + o) * m;
    vec3 ub = (rp - o) * m;

    t.x = max(uf.x, max(uf.y, uf.z));
    t.y = min(ub.x, min(ub.y, ub.z));

    // if(ray start == inside cube)
    if(t.x < 0.0 && t.y > 0.0) {t.xy = t.yx;  return 1.0;}

    return t.y < t.x ? 0.0 : (t.x > 0.0 ? 1.0 : -1.0);
}

/*
[...]

OUT:
	n0 : normal for t.x
	n1 : normal for t.y

EXAMPLE:
	vec2 t; vec3 n0, n1;
	float hit = Intersect_Ray_Cube(pos, dir, vec3(0.0), vec3(0.5), OUT t, n0, n1);
*/
float Intersect_Ray_Cube(
vec3 rp, vec3 rd,
vec3 cp, vec3 cth,
out vec2 t, out vec3 n0, out vec3 n1) {
    rp -= cp;

    vec3 m = 1.0 / -rd;
    vec3 os = If(lessThan(rd, vec3(0.0)), vec3(1.0), vec3(-1.0));
    //vec3 os = sign(-rd);
    vec3 o = -cth * os;


    vec3 uf = (rp + o) * m;
    vec3 ub = (rp - o) * m;

    //t.x = max(uf.x, max(uf.y, uf.z));
    //t.y = min(ub.x, min(ub.y, ub.z));

    if(uf.x > uf.y) {t.x = uf.x; n0 = vec3(os.x, 0.0, 0.0);} else
    {t.x = uf.y; n0 = vec3(0.0, os.y, 0.0);}
    if(uf.z > t.x ) {t.x = uf.z; n0 = vec3(0.0, 0.0, os.z);}

    if(ub.x < ub.y) {t.y = ub.x; n1 = vec3(os.x, 0.0, 0.0);} else
    {t.y = ub.y; n1 = vec3(0.0, os.y, 0.0);}
    if(ub.z < t.y ) {t.y = ub.z; n1 = vec3(0.0, 0.0, os.z);}


    // if(ray start == inside cube)
    if(t.x < 0.0 && t.y > 0.0)
    {
        t.xy = t.yx;

        vec3 n00 = n0;
        n0 = n1;
        n1 = n00;

        return 1.0;
    }

    return t.y < t.x ? 0.0 : (t.x > 0.0 ? 1.0 : -1.0);
}

/*
IN:
	rp		: ray start position
	rd		: ray direction (normalized)

	sp2		: sphere position
	sr2		: sphere radius squared

OUT:
	t		: distances to intersection points (negative if in backwards direction)

EXAMPLE:
	vec2 t;
	float hit = Intersect_Ray_Sphere(pos, dir, vec3(0.0), 1.0, OUT t);
*/
float Intersect_Ray_Sphere(
vec3 rp, vec3 rd,
vec3 sp, float sr2,
out vec2 t) {
    rp -= sp;

    float a = dot(rd, rd);
    float b = 2.0 * dot(rp, rd);
    float c = dot(rp, rp) - sr2;

    float D = b*b - 4.0*a*c;

    if(D < 0.0) return 0.0;

    float sqrtD = sqrt(D);
    // t = (-b + (c < 0.0 ? sqrtD : -sqrtD)) / a * 0.5;
    t = (-b + vec2(-sqrtD, sqrtD)) / a * 0.5;

    // if(start == inside) ...
    if(c < 0.0) t.xy = t.yx;

    // t.x > 0.0 || start == inside ? infront : behind
    return t.x > 0.0 || c < 0.0 ? 1.0 : -1.0;
}

bvec2 minmask(vec2 v) {
    bool x = v.x < v.y || isnan(v.y);

    return bvec2(x, !x);
}


bvec2 maxmask(vec2 v) {
    bool x = v.x >= v.y || isnan(v.y);

    return bvec2(x, !x);
}


bvec3 minmask(vec3 v) {
    return bvec3(v.x <= v.y && v.x <= v.z,
        v.y <  v.z && v.y <  v.x,
        v.z <  v.x && v.z <= v.y);
}

bvec3 maxmask(vec3 v) {
    return bvec3(v.x >= v.y && v.x >= v.z,
        v.y >  v.z && v.y >  v.x,
        v.z >  v.x && v.z >= v.y);
}

bvec3 minmask2(vec3 v) {
    bool x = !(v.x >  v.y || v.x >  v.z) && !isnan(v.x);
    bool y = !(v.y >= v.z || v.y >= v.x) && !isnan(v.y);

    return bvec3(x, y, !(x || y));
}

bvec3 maxmask2(vec3 v) {
    bool x = !(v.x <  v.y || v.x <  v.z) && !isnan(v.x);
    bool y = !(v.y <= v.z || v.y <= v.x) && !isnan(v.y);

    return bvec3(x, y, !(x || y));
}

void Intersect_Ray_CubeBackside(
vec3 rp, vec3 rd,
vec3 cp, vec3 cth,
out float t, out vec3 N) {
    rp -= cp;

    vec3 m = 1.0 / -rd;
    vec3 os = If(lessThan(rd, vec3(0.0)), vec3(1.0), vec3(-1.0));
    vec3 o = -cth * os;

    vec3 ub = (rp - o) * m;

    bvec3 mb = minmask2(ub);

    N = os * vec3(mb);

    t = mb.x ? ub.x : mb.y ? ub.y : ub.z;
}

bool IsInsideCube(vec3 p, vec3 cp, vec3 cd) {
    vec3 b = abs(p - cp);

    return b.x < cd.x && b.y < cd.y && b.z < cd.z;
}
//==========================================================================================================//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////////////// ACEScg
//==========================================================================================================//

// ACES fit by Stephen Hill (@self_shadow) | MIT license
// https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl

// sRGB => XYZ => D65_2_D60 => AP1
const mat3 sRGBtoAP1 = mat3
(
    0.613097, 0.339523, 0.047379,
    0.070194, 0.916354, 0.013452,
    0.020616, 0.109570, 0.869815
);

const mat3 AP1toSRGB = mat3
(
    1.704859, -0.621715, -0.083299,
    -0.130078,  1.140734, -0.010560,
    -0.023964, -0.128975,  1.153013
);

// AP1 => RRT_SAT
const mat3 RRT_SAT = mat3
(
    0.970889, 0.026963, 0.002148,
    0.010889, 0.986963, 0.002148,
    0.010889, 0.026963, 0.962148
);


// sRGB => XYZ => D65_2_D60 => AP1 => RRT_SAT
const mat3 ACESInputMat = mat3
(
    0.59719, 0.35458, 0.04823,
    0.07600, 0.90834, 0.01566,
    0.02840, 0.13383, 0.83777
);

// ODT_SAT => XYZ => D60_2_D65 => sRGB
const mat3 ACESOutputMat = mat3
(
    1.60475, -0.53108, -0.07367,
    -0.10208,  1.10813, -0.00605,
    -0.00327, -0.07276,  1.07602
);

vec3 ToneTF(vec3 x) {
    vec3 a = (x            + 0.0822192) * x;
    vec3 b = (x * 0.983521 + 0.5001330) * x + 0.274064;

    return a / b;
}

vec3 Tonemap(vec3 acescg) {
    vec3 color = acescg * RRT_SAT;

    color = ToneTF(color);

    color = color * ACESOutputMat;

    return color;
}

//==========================================================================================================//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////////////// scene intersection
//==========================================================================================================//
bool Intersect_Ray_Cube(vec3 rp, vec3 rd, vec3 c0p, vec3 c0d,
inout float hit, inout float t, inout vec3 n) {
    vec2 tt; vec3 n0, n1;
    float th = Intersect_Ray_Cube(rp, rd, c0p, c0d, /*out:*/ tt, n0, n1);

    if(th > 0.0)
    if(hit <= 0.0 || tt.x < t)
    {
        t = tt.x;
        n = n0;

        hit = 1.0;

        return true;
    }

    return false;
}

bool Intersect_Ray_Plate(vec3 rp, vec3 rd, vec3 c0p, vec3 c0d, float c1p_y, float c1dxz, float c1dy, float c1ps,
inout float hit, inout float t, inout vec3 n, inout vec3 a) {
    vec3 c1p = vec3(0.0, c1p_y, 0.0);
    vec3 c1d = vec3(c1dxz, c1dy, c1dxz);

    //c1p.y = 0.0;// remove tile pattern

    c1p.xz = (floor(rp.xz*c1ps+0.5))/c1ps;

    bool isInsideC0 = IsInsideCube(rp, c0p, c0d);
    bool isInsideC1 = IsInsideCube(rp, c1p, c1d);

    if(isInsideC0 && isInsideC1)
    {
        float t0; vec3 n0;
        Intersect_Ray_CubeBackside(rp, rd, c1p, c1d, t0, n0);

        if(IsInsideCube(rp+rd*t0, c0p, c0d))
        {
            if(hit <= 0.0 || t0 < t)
            {
                hit = 1.0;
                t = t0;
                n = n0;
                a = vec3(1.0, 1.0, 1.0);

                return true;
            }
        }
    }
    else
    {
        vec2 tt; vec3 n0, n1;
        float th = Intersect_Ray_Cube(rp, rd, c0p, c0d, /*out:*/ tt, n0, n1);

        bool hit0 = th > 0.0;
        if ( hit0 )
        {
            c1p.xz = (floor((rp.xz+rd.xz*tt.x)*c1ps+0.5))/c1ps;

            if(IsInsideCube(rp+rd*tt.x, c1p, c1d))
            {
                float t0; vec3 n0;
                Intersect_Ray_CubeBackside(rp, rd, c1p, c1d, t0, n0);

                if(IsInsideCube(rp+rd*t0, c0p, c0d))
                {
                    if(hit <= 0.0 || t0 < t)
                    {
                        hit = 1.0;
                        t = t0;
                        n = n0;
                        a = vec3(1.0, 1.0, 1.0);

                        if(n0.y != 0.0) a = vec3(.0, 1.0, 1.0);

                        return true;
                    }
                }
            }
            else
            {
                if(hit <= 0.0 || tt.x < t)
                {
                    hit = 1.0;
                    t = tt.x;
                    n = n0;
                    a = vec3(1.0, 1.0, 1.0);

                    return true;
                }
            }

        }
    }

    return false;
}

vec3 TransA(vec3 u, bool foo) {
    if(foo)
    {
        u.xy = u.yx;
        u.x *= -1.0;
    }
    else
    {
        u.x *= -1.0;
        u.xy = u.yx;
    }

    return u;
}

vec3 TransB(vec3 u, bool foo) {
    if(foo)
    {
        u.zy = u.yz;
        u.z *= -1.0;
    }
    else
    {
        u.z *= -1.0;
        u.zy = u.yz;
    }

    return u;
}


bool Intersect_Ray_Plates(vec3 rp, vec3 rd, vec3 c0p, vec3 c0d, float c1p_y, float c1dxz, float c1dy, float c1ps,
inout float hit, inout float t, inout vec3 n, inout vec3 a) {
    bool hit0 = false;

    rp.x += exp2(-18.0);
    //if(false)
    if(Intersect_Ray_Plate(TransA(rp, false), TransA(rd, false), c0p, c0d, c1p_y, c1dxz, c1dy, c1ps, /*inout*/ hit, t, n, a))
    {
        n = TransA(n, true);
        //a = vec3(1.0, 0.0, 0.0);
        a = a.x == 0.0 ? vec3(1., 0.125, 0.) : vec3(1.0, 0., 0.);

        hit0 = true;
    }
    //if(false)
    if(Intersect_Ray_Plate(TransB(rp, false), TransB(rd, false), c0p, c0d, c1p_y, c1dxz, c1dy, c1ps, /*inout*/ hit, t, n, a))
    {
        n = TransB(n, true);
        //a = vec3(0.0, 1.0, 0.0);
        a = a.x == 0.0 ? vec3(0., 1., 0.25) : vec3(.0, 1., 0.);
        hit0 = true;
    }

    return hit0;
}

float Intersect_Scene(
vec3 rp, vec3 rd,
out float t, out vec3 n, out vec3 a) {
    float hit = 0.0;

    float c1dxz = 0.4;
    float c1ps = 0.85;

    a = vec3(1.0);

    // ground plate thingy:
    //if(false)
    {
        if(Intersect_Ray_Plate(rp, rd, vec3(0.0, -0.8, 0.0), vec3(2.0, 0.125*1.2, 2.0), -0.64, c1dxz,  0.125, c1ps, /*inout*/ hit, t, n, a))
        {
            //a = vec3(1.0, 0.0, 0.0);
            //a = vec3(1.0);
            //a = vec3(0.0, 0.5, 1.);
            a = a.x == 0.0 ? vec3(.25, 1.0, .25) : vec3(0.0, 0.5, 1.);

        }

        if(Intersect_Ray_Plates(rp, rd, vec3(0.0, -2., 0.0), vec3(2.0, 0.15, 2.0), -1.9, 0.18,  0.125, 2., /*inout*/ hit, t, n, a))
        {
            //a = vec3(0.0, 1.0, 0.);
            //a = vec3(1.0);
        }
    }

    // slim pillars:
    //if(false)
    {
        float t0 = t;

        float r = 0.125*2.;
        float u = 1.0/c1ps;
        float l = 0.4;
        float h = 1.0;

        vec3 c0p = vec3(c1dxz, l, c1dxz);
        c0p = vec3(u - c1dxz, l, u - c1dxz);
        c0p = vec3(u - c1dxz, l, c1dxz);
        vec3 c0d = vec3(r, h, r);

        vec3 pa = vec3(u - c1dxz, l,     c1dxz);
        vec3 pb = vec3(u - c1dxz, l, u - c1dxz);
        vec3 pc = vec3(    c1dxz, l, u - c1dxz);

        //Intersect_Ray_Cube(rp, rd, pa, c0d, /*inout*/ hit, t, n);
        //Intersect_Ray_Cube(rp, rd, pb, c0d, /*inout*/ hit, t, n);
        //Intersect_Ray_Cube(rp, rd, pc, c0d, /*inout*/ hit, t, n);

        pa = vec3(-pa.z, pa.y, pa.x);
        pb = vec3(-pb.z, pb.y, pb.x);
        pc = vec3(-pc.z, pc.y, pc.x);

        //Intersect_Ray_Cube(rp, rd, pa, c0d, /*inout*/ hit, t, n);
        Intersect_Ray_Cube(rp, rd, pb, c0d, /*inout*/ hit, t, n);
        //Intersect_Ray_Cube(rp, rd, pc, c0d, /*inout*/ hit, t, n);

        pa = vec3(-pa.z, pa.y, pa.x);
        pb = vec3(-pb.z, pb.y, pb.x);
        pc = vec3(-pc.z, pc.y, pc.x);

        #if 0
        Intersect_Ray_Cube(rp, rd, pa, c0d, /*inout*/ hit, t, n);
        Intersect_Ray_Cube(rp, rd, pb, c0d, /*inout*/ hit, t, n);
        Intersect_Ray_Cube(rp, rd, pc, c0d, /*inout*/ hit, t, n);
        #endif

        pa = vec3(-pa.z, pa.y, pa.x);
        pb = vec3(-pb.z, pb.y, pb.x);
        pc = vec3(-pc.z, pc.y, pc.x);

        //Intersect_Ray_Cube(rp, rd, pa, c0d, /*inout*/ hit, t, n);
        Intersect_Ray_Cube(rp, rd, pb, c0d, /*inout*/ hit, t, n);
        //Intersect_Ray_Cube(rp, rd, pc, c0d, /*inout*/ hit, t, n);

        if(t < t0)
        {
            a = vec3(0., 0.5, 1.0);
            a = vec3(1.0);
        }
    }

    // top sphere:
    //if(false)
    {
        vec2 tt;
        float th = Intersect_Ray_Sphere(rp, rd, vec3(0.0), 0.5, /*out:*/ tt);

        //float hit0 = 0.0; th = Intersect_Ray_Cube(rp, rd, vec3(0.0), vec3(0.5, 1.0, 0.5), /*inout*/ hit0, tt.x, n) ? 1.0 : 0.0;

        if(th > 0.0)
        if(hit <= 0.0 || tt.x < t)
        {
            t = tt.x;
            n = normalize(rp + rd * tt.x);
            a = vec3(1.0, 1., 1.0);

            hit = 1.0;
        }
    }

    // bottom sphere:
    //if(false)
    {
        vec2 tt;
        float th = Intersect_Ray_Sphere(rp, rd, vec3(0.0, -2.25, 0.0), 2.0, /*out:*/ tt);

        if(th > 0.0)
        if(hit <= 0.0 || tt.x < t)
        {
            t = tt.x;
            n = normalize(rp + rd * tt.x - vec3(0.0, -2.25, 0.0));
            a = vec3(1.0);

            hit = 1.0;
        }
    }

    a = a * sRGBtoAP1;

    if(hit <= 0.0)
    {
        t = exp2(20.0);
        n = vec3(0.0);
        a = vec3(0.0);
    }

    return hit;
}
//==========================================================================================================//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////////////// scene lighting
//==========================================================================================================//

float Eval_SpotLight(vec3 wpos, vec3 N, vec3 Lpos, vec3 Ldir, float Lap) {
    vec3 Lvec = Lpos - wpos;

    vec3 L = normalize(Lvec);

    float NoL = clamp(dot(N, L), 0.0, 1.0);

    float LoD = -dot(L, Ldir);

    float sh0 = LoD > Lap ? 1.0 : 0.0;
    sh0 = smoothstep(Lap*0.9, Lap, LoD);
    sh0 *= mix(min((cos(LoD * 10.0 + cos(LoD*36.0)) * 0.5 + 0.5) * 2.0, 1.0), 1.0, LoD) * 1.0;

    float rad = NoL * RcpPi * sh0;
    rad /= dot(Lvec, Lvec);

    float sh;
    {
        vec3 p = wpos + N * 0.001;

        float t0; vec3 N0; vec3 a0;
        sh = 1.0 - Intersect_Scene(p, L, /*out:*/ t0, N0, a0);
    }

    rad *= sh;

    return rad;
}

vec3 Eval_Lighting(vec3 wpos, vec3 N, vec3 a) {
    vec3 rad;

    #if 1

    rad = a * Eval_SpotLight(wpos, N, vec3(-0.25, 2.65, -0.25), normalize(vec3( 0., -1., 0.0)), 0.4) * 16.0;

    #elif 0
    #else
    #endif

    return rad;
}

//==========================================================================================================//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

// C. Schlick. "Fast alternatives to Perlin’s bias and gain function"
// inverse(Bias(x,s)) =   Bias(x  ,1-s)
// inverse(Bias(x,s)) = 1-Bias(1-x,  s)
float Bias(float x, float s) {
    if(s == 0.0) return x != 1.0 ? 0.0 : 1.0;
    if(s == 1.0) return x != 0.0 ? 1.0 : 0.0;

    return x / ((1.0/s - 2.0) * (1.0 - x) + 1.0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////// sampling routines
//==========================================================================================================//

// s0 [-1..1], s1 [-1..1]
// samples spherical cap for s1 [cosAng05..1]
// samples hemisphere if s1 [0..1]
vec3 Sample_Sphere(float s0, float s1) {
    float ang = Pi * s0;
    float s1p = sqrt(1.0 - s1*s1);

    return vec3(cos(ang) * s1p,
        s1 ,
        sin(ang) * s1p);
}

vec3 Sample_Sphere(vec2 s) { return Sample_Sphere(s.x, s.y); }

//==========================================================================================================//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

// interleaved gradient noise | license: unclear
// Jorge Jimenez http://www.iryoku.com/next-generation-post-processing-in-call-of-duty-advanced-warfare
float IGN(vec2 uv) { return fract(52.9829189 * fract(dot(uv, vec2(0.06711056, 0.00583715)))); }

float IGN(vec2 uv, uint frame) {
    frame = frame % 64u;

    uv += 5.588238 * float(frame);

    return IGN(uv);
}

// linearizes uv using a Hilbert curve; tile dimension = 2^N
uint EvalHilbertCurve(uvec2 uv, uint N) {
    uint C = 0xB4361E9Cu;// cost lookup
    uint P = 0xEC7A9107u;// pattern lookup

    uint c = 0u;// accumulated cost
    uint p = 0u;// current pattern

    for(uint i = N; --i < N;)
    {
        uvec2 m = (uv >> i) & 1u;// local uv

        uint n = m.x ^ (m.y << 1u);// linearized local uv

        uint o = (p << 3u) ^ (n << 1u);// offset into lookup tables

        c += ((C >> o) & 3u) << (i << 1u);// accu cost (scaled by level)

        p = (P >> o) & 3u;// update pattern
    }

    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////// low-discrepancy sobol noise
//==========================================================================================================//
// "Shuffled Scrambled Sobol (2D)" - https://www.shadertoy.com/view/3lcczS | license: unclear
//  code taken from "Practical Hash-based Owen Scrambling" - http://www.jcgt.org/published/0009/04/01/
uint reverse_bits(uint x) {
    x = (((x & 0xaaaaaaaau) >> 1) | ((x & 0x55555555u) << 1));
    x = (((x & 0xccccccccu) >> 2) | ((x & 0x33333333u) << 2));
    x = (((x & 0xf0f0f0f0u) >> 4) | ((x & 0x0f0f0f0fu) << 4));
    x = (((x & 0xff00ff00u) >> 8) | ((x & 0x00ff00ffu) << 8));

    return ((x >> 16) | (x << 16));
}

// license: unclear
uint laine_karras_permutation(uint x, uint seed) {
    x += seed;
    x ^= x*0x6c50b47cu;
    x ^= x*0xb82f1e52u;
    x ^= x*0xc7afe638u;
    x ^= x*0x8d22f6e6u;

    return x;
}

// license: unclear
uint nested_uniform_scramble(uint x, uint seed) {
    x = reverse_bits(x);
    x = laine_karras_permutation(x, seed);
    x = reverse_bits(x);

    return x;
}

// from https://www.shadertoy.com/view/3ldXzM | license: unclear
uvec2 sobol_2d(uint index) {
    uvec2 p = uvec2(0u);
    uvec2 d = uvec2(0x80000000u);

    for(; index != 0u; index >>= 1u)
    {
        if((index & 1u) != 0u)
        {
            p ^= d;
        }

        d.x >>= 1u;  // 1st dimension Sobol matrix, is same as base 2 Van der Corput
        d.y ^= d.y >> 1u; // 2nd dimension Sobol matrix
    }

    return p;
}

// license: unclear
uvec2 shuffled_scrambled_sobol_2d(uint index, uint seed) {
    index = nested_uniform_scramble(index, seed);

    uvec2 p = sobol_2d(index);

    seed = seed * 2891336453u + 1u;
    p.x = nested_uniform_scramble(p.x, seed );
    seed = seed * 2891336453u + 1u;
    p.y = nested_uniform_scramble(p.y, seed);

    return p;
}

uint shuffled_scrambled_sobol_angle01(uint x, uint seed) {
    x = reverse_bits(x);

    x = laine_karras_permutation(x, seed);

    return x;
}
//==========================================================================================================//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// RNG
//==============================================================================================================================================//
uint  asuint2(float x) { return x == 0.0 ? 0u : floatBitsToUint(x); }
uvec2 asuint2(vec2 x) { return uvec2(asuint2(x.x ), asuint2(x.y)); }
uvec3 asuint2(vec3 x) { return uvec3(asuint2(x.xy), asuint2(x.z)); }
uvec4 asuint2(vec4 x) { return uvec4(asuint2(x.xy), asuint2(x.zw)); }

float Float01(uint x) { return float(    x ) * (1.0 / 4294967296.0); }
float Float11(uint x) { return float(int(x)) * (1.0 / 2147483648.0); }

vec2 Float01(uvec2 x) { return vec2(      x ) * (1.0 / 4294967296.0); }
vec2 Float11(uvec2 x) { return vec2(ivec2(x)) * (1.0 / 2147483648.0); }

vec3 Float01(uvec3 x) { return vec3(      x ) * (1.0 / 4294967296.0); }
vec3 Float11(uvec3 x) { return vec3(ivec3(x)) * (1.0 / 2147483648.0); }

vec4 Float01(uvec4 x) { return vec4(      x ) * (1.0 / 4294967296.0); }
vec4 Float11(uvec4 x) { return vec4(ivec4(x)) * (1.0 / 2147483648.0); }

// http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
// https://probablydance.com/2018/06/16/fibonacci-hashing-the-optimization-that-the-world-forgot-or-a-better-alternative-to-integer-modulo/
const float rPhif1 =      0.6180340;
const vec2  rPhif2 = vec2(0.7548777, 0.5698403);
const vec3  rPhif3 = vec3(0.8191725, 0.6710436, 0.5497005);
const vec4  rPhif4 = vec4(0.8566749, 0.7338919, 0.6287067, 0.5385973);

const uint  rPhi1 =       2654435769u;
const uvec2 rPhi2 = uvec2(3242174889u, 2447445413u);
const uvec3 rPhi3 = uvec3(3518319153u, 2882110345u, 2360945575u);
const uvec4 rPhi4 = uvec4(3679390609u, 3152041523u, 2700274805u, 2313257605u);

// low bias version | https://nullprogram.com/blog/2018/07/31/ | license: public domain (http://unlicense.org/)
uint WellonsHash(uint x) {
    x ^= x >> 16u;
    x *= 0x7feb352dU;
    x ^= x >> 15u;
    x *= 0x846ca68bU;
    x ^= x >> 16u;

    return x;
}

// minimal bias version | https://nullprogram.com/blog/2018/07/31/ | license: public domain (http://unlicense.org/)
uint WellonsHash2(uint x) {
    x ^= x >> 17u;
    x *= 0xed5ad4bbU;
    x ^= x >> 11u;
    x *= 0xac4c1b51U;
    x ^= x >> 15u;
    x *= 0x31848babU;
    x ^= x >> 14u;

    return x;
}

// http://marc-b-reynolds.github.io/math/2016/03/29/weyl_hash.html | license: public domain (http://unlicense.org/)
uint WeylHash(uvec2 c) {
    return ((c.x * 0x3504f333u) ^ (c.y * 0xf1bbcdcbu)) * 741103597u;
}

// Pierre L'Ecuyer - "TABLES OF LINEAR CONGRUENTIAL GENERATORS OF DIFFERENT SIZES AND GOOD LATTICE STRUCTURE"
// https://www.ams.org/journals/mcom/1999-68-225/S0025-5718-99-00996-5/S0025-5718-99-00996-5.pdf
const uint lcgM = 2891336453u;// ideal for 32 bits with odd c

uint lcg(uint h) {
    return h * lcgM + 0x5C995C6Du;
}

#define SEED uvec4(0x5C995C6Du, 0x6A3C6A57u, 0xC65536CBu, 0x3563995Fu)

// Melissa E. O’Neill - "PCG: A Family of Simple Fast Space-Efficient Statistically Good Algorithms for Random Number Generation"
// https://www.cs.hmc.edu/tr/hmc-cs-2014-0905.pdf

// Mark Jarzynski & Marc Olano - "Hash Functions for GPU Rendering"
// http://jcgt.org/published/0009/03/02/ | https://www.shadertoy.com/view/XlGcRh
uvec3 pcg3Mix(uvec3 h) {
    h.x += h.y * h.z;
    h.y += h.z * h.x;
    h.z += h.x * h.y;

    return h;
}

uvec3 pcg3Permute(uvec3 h) {
    h = pcg3Mix(h);

    h ^= h >> 16u;

    return pcg3Mix(h);
}

uvec3 pcg3(inout uint state) {
    state = lcg(state);

    return pcg3Permute(uvec3(2447445413u, state, 3242174889u));
}

uvec3 pcg3(uvec3 h, uint seed) {
    uvec3 c = (seed << 1u) ^ SEED.xyz;

    return pcg3Permute(h * lcgM + c);
}

uvec4 pcg4Mix(uvec4 h) {
    h.x += h.y * h.w;
    h.y += h.z * h.x;
    h.z += h.x * h.y;
    h.w += h.y * h.z;

    return h;
}

uvec4 pcg4Permute(uvec4 h) {
    h = pcg4Mix(h);

    h ^= h >> 16u;

    return pcg4Mix(h);
}

uvec4 pcg4(inout uint state) {
    state = lcg(state);

    return pcg4Permute(uvec4(2882110345u, state, 3518319153u, 2360945575u));
}

uvec4 pcg4(uvec4 h, uint seed) {
    uvec4 c = (seed << 1u) ^ SEED;

    return pcg4Permute(h * lcgM + c);
}

uint pcg(inout uint state) {
    state = lcg(state);

    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;

    return (word >> 22u) ^ word;
}

uint pcg(uint h, uint seed) {
    uint c = (seed << 1u) ^ SEED.x;

    h = h * lcgM + c;

    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;

    return (h >> 22u) ^ h;
}

#undef SEED

//---------------------------------------------------------------------------------------------//

uint  Hash(uint  h, uint seed) { return pcg(h, seed); }
uvec2 Hash(uvec2 h, uint seed) { return pcg3(uvec3(h, 0u), seed).xy; }
uvec3 Hash(uvec3 h, uint seed) { return pcg3(h, seed); }
uvec4 Hash(uvec4 h, uint seed) { return pcg4(h, seed); }

vec4 Hash01x4(vec4  v, uint seed) { return Float01(pcg4(asuint2(v), seed)); }
vec4 Hash01x4(vec3  v, uint seed) { return Hash01x4(vec4(v, 0.0          ), seed); }
vec4 Hash01x4(vec2  v, uint seed) { return Hash01x4(vec4(v, 0.0, 0.0     ), seed); }
vec4 Hash01x4(float v, uint seed) { return Hash01x4(vec4(v, 0.0, 0.0, 0.0), seed); }

vec3 Hash01x3(vec4  v, uint seed) { return Float01(pcg4(asuint2(v), seed).xyz); }
vec3 Hash01x3(vec3  v, uint seed) { return Float01(pcg3(asuint2(v), seed)); }
vec3 Hash01x3(vec2  v, uint seed) { return Hash01x3(vec3(v, 0.0     ), seed); }
vec3 Hash01x3(float v, uint seed) { return Hash01x3(vec3(v, 0.0, 0.0), seed); }

vec2 Hash01x2(vec4  v, uint seed) { return Float01(pcg4(asuint2(v), seed).xy); }
vec2 Hash01x2(vec3  v, uint seed) { return Float01(pcg3(asuint2(v), seed).xy); }
vec2 Hash01x2(vec2  v, uint seed) { return Hash01x3(vec3(v, 0.0     ), seed).xy; }
vec2 Hash01x2(float v, uint seed) { return Hash01x3(vec3(v, 0.0, 0.0), seed).xy; }

float Hash01(vec4  v, uint seed) { return Float01(pcg4(asuint2(v), seed).x); }
float Hash01(vec3  v, uint seed) { return Float01(pcg3(asuint2(v), seed).x); }
float Hash01(vec2  v, uint seed) { return Float01(pcg3(asuint2(vec3(v, 0.0)), seed).x); }
float Hash01(float v, uint seed) { return Float01(pcg(asuint2(v), seed)); }


vec4 Hash11x4(vec4  v, uint seed) { return Float11(pcg4(asuint2(v), seed)); }
vec4 Hash11x4(vec3  v, uint seed) { return Hash11x4(vec4(v, 0.0          ), seed); }
vec4 Hash11x4(vec2  v, uint seed) { return Hash11x4(vec4(v, 0.0, 0.0     ), seed); }
vec4 Hash11x4(float v, uint seed) { return Hash11x4(vec4(v, 0.0, 0.0, 0.0), seed); }

vec3 Hash11x3(vec4  v, uint seed) { return Float11(pcg4(asuint2(v), seed).xyz); }
vec3 Hash11x3(vec3  v, uint seed) { return Float11(pcg3(asuint2(v), seed)); }
vec3 Hash11x3(vec2  v, uint seed) { return Hash11x3(vec3(v, 0.0     ), seed); }
vec3 Hash11x3(float v, uint seed) { return Hash11x3(vec3(v, 0.0, 0.0), seed); }

vec2 Hash11x2(vec4  v, uint seed) { return Float11(pcg4(asuint2(v), seed).xy); }
vec2 Hash11x2(vec3  v, uint seed) { return Float11(pcg3(asuint2(v), seed).xy); }
vec2 Hash11x2(vec2  v, uint seed) { return Hash11x3(vec3(v, 0.0     ), seed).xy; }
vec2 Hash11x2(float v, uint seed) { return Hash11x3(vec3(v, 0.0, 0.0), seed).xy; }

float Hash11(vec4  v, uint seed) { return Float11(pcg4(asuint2(v), seed).x); }
float Hash11(vec3  v, uint seed) { return Float11(pcg3(asuint2(v), seed).x); }
float Hash11(vec2  v, uint seed) { return Float11(pcg3(asuint2(vec3(v, 0.0)), seed).x); }
float Hash11(float v, uint seed) { return Float11(pcg(asuint2(v), seed)); }


vec4 Hash01x4(uvec4 v, uint seed) { return Float01(pcg4(v, seed)); }
vec4 Hash01x4(uvec3 v, uint seed) { return Hash01x4(uvec4(v, 0u        ), seed); }
vec4 Hash01x4(uvec2 v, uint seed) { return Hash01x4(uvec4(v, 0u, 0u    ), seed); }
vec4 Hash01x4(uint  v, uint seed) { return Hash01x4(uvec4(v, 0u, 0u, 0u), seed); }

vec3 Hash01x3(uvec4 v, uint seed) { return Float01(pcg4(v, seed).xyz); }
vec3 Hash01x3(uvec3 v, uint seed) { return Float01(pcg3(v, seed)); }
vec3 Hash01x3(uvec2 v, uint seed) { return Hash01x3(uvec3(v, 0u    ), seed); }
vec3 Hash01x3(uint  v, uint seed) { return Hash01x3(uvec3(v, 0u, 0u), seed); }

vec2 Hash01x2(uvec4 v, uint seed) { return Float01(pcg4(v, seed).xy); }
vec2 Hash01x2(uvec3 v, uint seed) { return Float01(pcg3(v, seed).xy); }
vec2 Hash01x2(uvec2 v, uint seed) { return Hash01x3(uvec3(v, 0u    ), seed).xy; }
vec2 Hash01x2(uint  v, uint seed) { return Hash01x3(uvec3(v, 0u, 0u), seed).xy; }

float Hash01(uvec4 v, uint seed) { return Float01(pcg4(v, seed).x); }
float Hash01(uvec3 v, uint seed) { return Float01(pcg3(v, seed).x); }
float Hash01(uvec2 v, uint seed) { return Float01(pcg3(uvec3(v, 0u), seed).x); }
float Hash01(uint  v, uint seed) { return Float01(pcg(v, seed)); }


vec4 Hash11x4(uvec4 v, uint seed) { return Float11(pcg4(v, seed)); }
vec4 Hash11x4(uvec3 v, uint seed) { return Hash11x4(uvec4(v, 0u        ), seed); }
vec4 Hash11x4(uvec2 v, uint seed) { return Hash11x4(uvec4(v, 0u, 0u    ), seed); }
vec4 Hash11x4(uint  v, uint seed) { return Hash11x4(uvec4(v, 0u, 0u, 0u), seed); }

vec3 Hash11x3(uvec4 v, uint seed) { return Float11(pcg4(v, seed).xyz); }
vec3 Hash11x3(uvec3 v, uint seed) { return Float11(pcg3(v, seed)); }
vec3 Hash11x3(uvec2 v, uint seed) { return Hash11x3(uvec3(v, 0u    ), seed); }
vec3 Hash11x3(uint  v, uint seed) { return Hash11x3(uvec3(v, 0u, 0u), seed); }

vec2 Hash11x2(uvec4 v, uint seed) { return Float11(pcg4(v, seed).xy); }
vec2 Hash11x2(uvec3 v, uint seed) { return Float11(pcg3(v, seed).xy); }
vec2 Hash11x2(uvec2 v, uint seed) { return Hash11x3(uvec3(v, 0u    ), seed).xy; }
vec2 Hash11x2(uint  v, uint seed) { return Hash11x3(uvec3(v, 0u, 0u), seed).xy; }

float Hash11(uvec4 v, uint seed) { return Float11(pcg4(v, seed).x); }
float Hash11(uvec3 v, uint seed) { return Float11(pcg3(v, seed).x); }
float Hash11(uvec2 v, uint seed) { return Float11(pcg3(uvec3(v, 0u), seed).x); }
float Hash11(uint  v, uint seed) { return Float11(pcg(v, seed)); }

//==============================================================================================================================================//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
