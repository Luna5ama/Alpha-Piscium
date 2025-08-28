/*
    References:
        [WIK25a] Wikipedia. "Rec. 601". 2025.
            https://en.wikipedia.org/wiki/Rec._601
        [WIK25b] Wikipedia. "Rec. 709". 2025.
            https://en.wikipedia.org/wiki/Rec._709
        [WIK25c] Wikipedia. "sRGB". 2025.
            https://en.wikipedia.org/wiki/SRGB
        [ITU25] ITU, "Recommendation BT.2100", 2025.
            https://www.itu.int/rec/R-REC-BT.2100
            
    Note:
        EOTF: Electro-Optical Transfer Function (decoding gamma, non-linear to linear)
        OETF: Opto-Electronic Transfer Function (encoding gamma, linear to non-linear)
*/
#include "/util/Math.glsl"

// -------------------------------------------------------- API --------------------------------------------------------
#define COLORS2_TF_IDENTITY 0
#define COLORS2_TF_REC601 1
#define COLORS2_TF_REC709 2
#define COLORS2_TF_SRGB   3
#define COLORS2_TF_EXP22  4
#define COLORS2_TF_EXP24  5
#define COLORS2_TF_PQ     6
#define COLORS2_TF_HLG    7

#define _colors2_oetf(a, x) colors2_oetf_## a ##(x)
#define colors2_oetf(a, x) _colors2_oetf(a, x)
#define _colors2_eotf(a, x) colors2_eotf_## a ##(x)
#define colors2_eotf(a, x) _colors2_eotf(a, x)


// ------------------------------------------------ Transfer Functions ------------------------------------------------
float colors2_oetf_identity(float c) {
    return c;
}

vec3 colors2_oetf_identity(vec3 c) {
    return c;
}

float colors2_eotf_identity(float c) {
    return c;
}

vec3 colors2_eotf_identity(vec3 c) {
    return c;
}

// ----------------------------------------------------- Rec. 601 -----------------------------------------------------
// [WIK25a]
float colors2_oetf_Rec601(float c) {
    float lower = 4.5 * c;
    float higher = pow(c, 0.45) * 1.099 - 0.099;
    return mix(lower, higher, c >= 0.018);
}

vec3 colors2_oetf_Rec601(vec3 c) {
    vec3 lower = 4.5 * c;
    vec3 higher = pow(c, vec3(0.45)) * 1.099 - 0.099;
    return mix(lower, higher, greaterThanEqual(c, vec3(0.018)));
}

float colors2_eotf_Rec601(float c) {
    float lower = c / 4.5;
    float higher = pow((c + 0.099) / 1.099, 1.0 / 0.45);
    return mix(lower, higher, c >= 0.081);
}

vec3 colors2_eotf_Rec601(vec3 c) {
    vec3 lower = c / 4.5;
    vec3 higher = pow((c + 0.099) / 1.099, vec3(1.0 / 0.45));
    return mix(lower, higher, greaterThanEqual(c, vec3(0.081)));
}

// ----------------------------------------------------- Rec. 709 -----------------------------------------------------
// [WIK25b]
float colors2_oetf_Rec709(float c) {
    return colors2_oetf_Rec601(c);
}

vec3 colors2_oetf_Rec709(vec3 c) {
    return colors2_oetf_Rec601(c);
}

float colors2_eotf_Rec709(float c) {
    return colors2_eotf_Rec601(c);
}

vec3 colors2_eotf_Rec709(vec3 c) {
    return colors2_eotf_Rec601(c);
}

// ------------------------------------------------------ sRGB --------------------------------------------------------
// [WIK25c]
float colors2_oetf_sRGB(float c) {
    float lower = 12.92 * c;
    float higher = pow(c, 1.0 / 2.4) * 1.055 - 0.055;
    return mix(lower, higher, c > 0.0031308);
}

vec3 colors2_oetf_sRGB(vec3 c) {
    vec3 lower = 12.92 * c;
    vec3 higher = pow(c, vec3(1.0 / 2.4)) * 1.055 - 0.055;
    return mix(lower, higher, greaterThan(c, vec3(0.0031308)));
}

float colors2_eotf_sRGB(float c) {
    float lower = c / 12.92;
    float higher = pow((c + 0.055) / 1.055, 2.4);
    return mix(lower, higher, c > 0.04045);
}

vec3 colors2_eotf_sRGB(vec3 c) {
    vec3 lower = c / 12.92;
    vec3 higher = pow((c + 0.055) / 1.055, vec3(2.4));
    return mix(lower, higher, greaterThan(c, vec3(0.04045)));
}

// ------------------------------------------------- Exponential 2.2 -------------------------------------------------

float colors2_oetf_Exp22(float c) {
    return pow(c, 1.0 / 2.2);
}

vec3 colors2_oetf_Exp22(vec3 c) {
    return pow(c, vec3(1.0 / 2.2));
}

float colors2_eotf_Exp22(float c) {
    return pow(c, 2.2);
}

vec3 colors2_eotf_Exp22(vec3 c) {
    return pow(c, vec3(2.2));
}

// ------------------------------------------------- Exponential 2.4 -------------------------------------------------
float colors2_oetf_Exp24(float c) {
    return pow(c, 1.0 / 2.4);
}

vec3 colors2_oetf_Exp24(vec3 c) {
    return pow(c, vec3(1.0 / 2.4));
}

float colors2_eotf_Exp24(float c) {
    return pow(c, 2.4);
}

vec3 colors2_eotf_Exp24(vec3 c) {
    return pow(c, vec3(2.4));
}

// --------------------------------------------------- PQ (ST 2084) ---------------------------------------------------
// [ITU25]
const float _PQ_M1 = 2610.0 / 16384.0;
const float _PQ_M2 = 2523.0 / 4096.0 * 128.0;
const float _PQ_C1 = 3424.0 / 4096.0;
const float _PQ_C2 = 2413.0 / 4096.0 * 32.0;
const float _PQ_C3 = 2392.0 / 4096.0 * 32.0;
const float _PQ_PEAK = 10000.0;

float colors2_oetf_PQ(float c) {
    float y = c / _PQ_PEAK;
    float yPow = pow(y, _PQ_M1);
    float numerator = _PQ_C1 + _PQ_C2 * yPow;
    float denominator = 1.0 + _PQ_C3 * yPow;
    return pow(numerator / denominator, _PQ_M2);
}

vec3 colors2_oetf_PQ(vec3 c) {
    vec3 y = c / _PQ_PEAK;
    vec3 yPow = pow(y, vec3(_PQ_M1));
    vec3 numerator = vec3(_PQ_C1) + vec3(_PQ_C2) * yPow;
    vec3 denominator = vec3(1.0) + vec3(_PQ_C3) * yPow;
    return pow(numerator / denominator, vec3(_PQ_M2));
}

float colors2_eotf_PQ(float c) {
    float ePow = pow(c, 1.0 / _PQ_M2);
    float numerator = max(ePow - _PQ_C1, 0.0);
    float denominator = _PQ_C2 - _PQ_C3 * ePow;
    float y = pow(numerator / denominator, 1.0 / _PQ_M1);
    return y * _PQ_PEAK;
}

vec3 colors2_eotf_PQ(vec3 c) {
    vec3 ePow = pow(c, vec3(1.0 / _PQ_M2));
    vec3 numerator = max(ePow - vec3(_PQ_C1), 0.0);
    vec3 denominator = _PQ_C2 - _PQ_C3 * ePow;
    vec3 y = pow(numerator / denominator, vec3(1.0 / _PQ_M1));
    return y * _PQ_PEAK;
}

// -------------------------------------------------------- HLG --------------------------------------------------------
// [ITU25]
const float _HLG_A = 0.17883277;
const float _HLG_B = 1.0 - 4.0 * _HLG_A;
const float _HLG_C = 0.5 - _HLG_A * log(4.0 * _HLG_A);

float colors2_oetf_HLG(float c) {
    float lower = sqrt(3.0 * c);
    float higher = _HLG_A * log(12.0 * c - _HLG_B) + _HLG_C;
    return mix(lower, higher, c > 1.0 / 12.0);
}

vec3 colors2_oetf_HLG(vec3 c) {
    vec3 lower = sqrt(3.0 * c);
    vec3 higher = _HLG_A * log(12.0 * c - _HLG_B) + _HLG_C;
    return mix(lower, higher, greaterThan(c, vec3(1.0 / 12.0)));
}

float colors2_eotf_HLG(float c) {
    float lower = c * c / 3.0;
    float higher = (exp((c - _HLG_C) / _HLG_A) + _HLG_B) / 12.0;
    return mix(lower, higher, c > 0.5);
}

vec3 colors2_eotf_HLG(vec3 c) {
    vec3 lower = c * c / 3.0;
    vec3 higher = (exp((c - vec3(_HLG_C)) / vec3(_HLG_A)) + vec3(_HLG_B)) / 12.0;
    return mix(lower, higher, greaterThan(c, vec3(0.5)));
}


// ------------------------------------------------- Adapter Functions -------------------------------------------------
float colors2_oetf_0(float c) { return colors2_eotf_identity(c); }
vec3 colors2_oetf_0(vec3 c) { return colors2_eotf_identity(c); }
float colors2_eotf_0(float c) { return colors2_oetf_identity(c); }
vec3 colors2_eotf_0(vec3 c) { return colors2_oetf_identity(c); }

float colors2_oetf_1(float c) { return colors2_oetf_Rec601(c); }
vec3 colors2_oetf_1(vec3 c) { return colors2_oetf_Rec601(c); }
float colors2_eotf_1(float c) { return colors2_eotf_Rec601(c); }
vec3 colors2_eotf_1(vec3 c) { return colors2_eotf_Rec601(c); }

float colors2_oetf_2(float c) { return colors2_oetf_Rec709(c); }
vec3 colors2_oetf_2(vec3 c) { return colors2_oetf_Rec709(c); }
float colors2_eotf_2(float c) { return colors2_eotf_Rec709(c); }
vec3 colors2_eotf_2(vec3 c) { return colors2_eotf_Rec709(c); }

float colors2_oetf_3(float c) { return colors2_oetf_sRGB(c); }
vec3 colors2_oetf_3(vec3 c) { return colors2_oetf_sRGB(c); }
float colors2_eotf_3(float c) { return colors2_eotf_sRGB(c); }
vec3 colors2_eotf_3(vec3 c) { return colors2_eotf_sRGB(c); }

float colors2_oetf_4(float c) { return colors2_oetf_Exp22(c); }
vec3 colors2_oetf_4(vec3 c) { return colors2_oetf_Exp22(c); }
float colors2_eotf_4(float c) { return colors2_eotf_Exp22(c); }
vec3 colors2_eotf_4(vec3 c) { return colors2_eotf_Exp22(c); }

float colors2_oetf_5(float c) { return colors2_oetf_Exp24(c); }
vec3 colors2_oetf_5(vec3 c) { return colors2_oetf_Exp24(c); }
float colors2_eotf_5(float c) { return colors2_eotf_Exp24(c); }
vec3 colors2_eotf_5(vec3 c) { return colors2_eotf_Exp24(c); }

float colors2_oetf_6(float c) { return colors2_oetf_PQ(c); }
vec3 colors2_oetf_6(vec3 c) { return colors2_oetf_PQ(c); }
float colors2_eotf_6(float c) { return colors2_eotf_PQ(c); }
vec3 colors2_eotf_6(vec3 c) { return colors2_eotf_PQ(c); }

float colors2_oetf_7(float c) { return colors2_oetf_HLG(c); }
vec3 colors2_oetf_7(vec3 c) { return colors2_oetf_HLG(c); }
float colors2_eotf_7(float c) { return colors2_eotf_HLG(c); }
vec3 colors2_eotf_7(vec3 c) { return colors2_eotf_HLG(c); }