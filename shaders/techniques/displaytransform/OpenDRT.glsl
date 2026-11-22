#ifndef INCLUDE_techniques_drt_OpenDRT_glsl
#define INCLUDE_techniques_drt_OpenDRT_glsl a



// -----------------------------------------------------------------------------
// OpenDRT v1.1.0 - GLSL Port
// Original: Jed Smith (https://github.com/jedypod/open-display-transform)
// License: GPLv3
// Look Preset: Standard
// -----------------------------------------------------------------------------

// P3-D65 <-> CIE XYZ D65 (column-major, transposed from DCTL row-major)
const mat3 _ODRT_XYZ_TO_P3D65 = mat3(
    2.49349691, -0.82948897, 0.03584583,
    -0.93138362, 1.76266406, -0.07617239,
    -0.40271078, 0.02362469, 0.95688452);
const mat3 _ODRT_P3D65_TO_XYZ = mat3(
    0.48657095, 0.22897456, 0.0,
    0.26566769, 0.69173852, 0.04511338,
    0.19821729, 0.07928691, 1.04394437);

#define _ODRT_SQRT3 1.73205080757

float _odrt_spow(float a, float b) {
    return a <= 0.0 ? a : pow(a, b);
}

float _odrt_compress_hyp_pow(float x, float s, float p) {
    return _odrt_spow(x / (x + s), p);
}

float _odrt_toe_quad(float x, float toe, bool inv) {
    if (toe == 0.0) return x;
    if (!inv) return x * x / (x + toe);
    return (x + sqrt(x * (4.0 * toe + x))) * 0.5;
}

float _odrt_softplus(float x, float s) {
    if (x > 10.0 * s || s < 1e-4) return x;
    return s * log(max(0.0, 1.0 + exp(x / s)));
}

float _odrt_gauss(float x, float w) {
    return exp(-x * x / w);
}

float _odrt_hue_off(float h, float o) {
    return mod(h - o + PI, PI_2) - PI;
}

vec2 _odrt_opponent(vec3 rgb) {
    return vec2(rgb.r - rgb.b, rgb.g - (rgb.r + rgb.b) * 0.5);
}

vec3 drt_openDRT(vec3 color) {
    vec3 rgb = max(color, 0.0);

    // CIE XYZ -> P3D65
    rgb = _ODRT_XYZ_TO_P3D65 * rgb;

    // --- Standard Look Preset Parameters ---
    // Tonescale
    const float tn_con = 1.66, tn_sh = 0.5, tn_toe = 0.003, tn_off = 0.005;
    #ifdef AP_HDR_OUTPUT
    float tn_Lp = 100.0 * hdrPeakRatio();
    #else
    const float tn_Lp = 100.0;
    #endif
    const float tn_gb = 0.13, tn_Lg = 10.0;
    const float pt_hdr = 0.5;
    const int tn_su = 1; // Dim surround
    // Render space
    const float rs_sa = 0.35, rs_rw = 0.25, rs_bw = 0.55;
    // Purity compress
    const float pt_lml = 0.25, pt_lml_r = 0.5, pt_lml_g = 0.0, pt_lml_b = 0.1;
    const float pt_lmh = 0.25, pt_lmh_r = 0.5, pt_lmh_b = 0.0;
    const float ptl_c = 0.06, ptl_m = 0.08, ptl_y = 0.06;
    // Mid purity
    const float ptm_low = 0.4, ptm_low_rng = 0.25, ptm_low_st = 0.5;
    const float ptm_high = -0.8, ptm_high_rng = 0.35, ptm_high_st = 0.4;
    // Brilliance
    const float brl = 0.0, brl_r = -2.5, brl_g = -1.5, brl_b = -1.5;
    const float brl_rng = 0.5, brl_st = 0.35;
    // Post brilliance
    const float brlp = -0.5, brlp_r = -1.25, brlp_g = -1.25, brlp_b = -0.25;
    // Hue contrast
    const float hc_r = 1.0, hc_r_rng = 0.3;
    // Hue shift RGB
    const float hs_r = 0.6, hs_r_rng = 0.6, hs_g = 0.35, hs_g_rng = 1.0, hs_b = 0.66, hs_b_rng = 1.0;
    // Hue shift CMY
    const float hs_c = 0.25, hs_c_rng = 1.0, hs_m = 0.0, hs_m_rng = 1.0, hs_y = 0.0, hs_y_rng = 1.0;

    // --- Tonescale Constraint Calculations ---
    float ts_x1 = pow(2.0, 6.0 * tn_sh + 4.0);
    float ts_y1 = tn_Lp / 100.0;
    float ts_x0 = 0.18 + tn_off;
    float ts_y0 = tn_Lg / 100.0 * (1.0 + tn_gb * log2(max(ts_y1, 1e-10)));
    float ts_s0 = _odrt_toe_quad(ts_y0, tn_toe, true);
    float ts_p = tn_con / (1.0 + float(tn_su) * 0.05);
    float ts_s10 = ts_x0 * (pow(ts_s0, -1.0 / tn_con) - 1.0);
    float ts_m1 = ts_y1 / pow(ts_x1 / (ts_x1 + ts_s10), tn_con);
    float ts_m2 = _odrt_toe_quad(ts_m1, tn_toe, true);
    float ts_s = ts_x0 * (pow(ts_s0 / ts_m2, -1.0 / tn_con) - 1.0);
    float ts_dsc = 100.0 / tn_Lp;
    float pt_cmp_Lf = pt_hdr * min(1.0, (tn_Lp - 100.0) / 900.0);
    float s_Lp100 = ts_x0 * (pow(tn_Lg / 100.0, -1.0 / tn_con) - 1.0);
    float ts_s1 = ts_s * pt_cmp_Lf + s_Lp100 * (1.0 - pt_cmp_Lf);

    // --- Render Space ---
    vec3 rs_w = vec3(rs_rw, 1.0 - rs_rw - rs_bw, rs_bw);
    float sat_L = dot(rgb, rs_w);
    rgb = mix(rgb, vec3(sat_L), rs_sa);

    // Offset
    rgb += tn_off;

    // Tonescale Norm
    float tsn = length(rgb) / _ODRT_SQRT3;

    // RGB Ratios
    rgb = tsn == 0.0 ? vec3(0.0) : rgb / tsn;

    // Opponent and achromatic distance
    vec2 opp = _odrt_opponent(rgb);
    float ach_d = length(opp) * 0.5;
    ach_d = 1.25 * _odrt_toe_quad(ach_d, 0.25, false);

    // Hue angle (rotated so red = 0)
    float hue = mod(atan(opp.x, opp.y) + PI + 1.10714931, PI_2);

    // Hue angle windows
    vec3 ha_rgb = vec3(
        _odrt_gauss(_odrt_hue_off(hue, 0.1), 0.66),
        _odrt_gauss(_odrt_hue_off(hue, 4.3), 0.66),
        _odrt_gauss(_odrt_hue_off(hue, 2.3), 0.66));
    vec3 ha_rgb_hs = vec3(
        _odrt_gauss(_odrt_hue_off(hue, -0.4), 0.66),
        ha_rgb.y,
        _odrt_gauss(_odrt_hue_off(hue, 2.5), 0.66));
    vec3 ha_cmy = vec3(
        _odrt_gauss(_odrt_hue_off(hue, 3.3), 0.5),
        _odrt_gauss(_odrt_hue_off(hue, 1.3), 0.5),
        _odrt_gauss(_odrt_hue_off(hue, -1.15), 0.5));

    // --- Brilliance ---
    {
        float brl_tsf = pow(tsn / (tsn + 1.0), 1.0 - brl_rng);
        float brl_exf = (brl + brl_r * ha_rgb.x + brl_g * ha_rgb.y + brl_b * ha_rgb.z) * _odrt_spow(ach_d, 1.0 / brl_st);
        float brl_ex = exp2(brl_exf * (brl_exf < 0.0 ? brl_tsf : 1.0 - brl_tsf));
        tsn *= brl_ex;
    }

    // --- Hyperbolic Compression ---
    // For SDR (100 nits): ts_s1 == s_Lp100, so tsn_pt == tsn_const
    float tsn_pt = _odrt_compress_hyp_pow(tsn, ts_s1, ts_p);
    float tsn_const = tsn_pt;
    tsn = _odrt_compress_hyp_pow(tsn, ts_s, ts_p);

    // --- Hue Contrast R ---
    {
        float hc_ts = 1.0 - tsn_const;
        float hc_c = hc_ts * (1.0 - ach_d) + ach_d * (1.0 - hc_ts);
        hc_c *= ach_d * ha_rgb.x;
        hc_ts = _odrt_spow(hc_ts, 1.0 / hc_r_rng);
        float hc_f = hc_r * (hc_c - 2.0 * hc_c * hc_ts) + 1.0;
        rgb.gb *= hc_f;
    }

    // --- Hue Shift RGB ---
    {
        vec3 hs_rgb_v = vec3(
            ha_rgb_hs.x * ach_d * _odrt_spow(tsn_pt, 1.0 / hs_r_rng),
            ha_rgb_hs.y * ach_d * _odrt_spow(tsn_pt, 1.0 / hs_g_rng),
            ha_rgb_hs.z * ach_d * _odrt_spow(tsn_pt, 1.0 / hs_b_rng));
        vec3 hsf = vec3(hs_rgb_v.x * hs_r, hs_rgb_v.y * -hs_g, hs_rgb_v.z * -hs_b);
        rgb += vec3(hsf.z - hsf.y, hsf.x - hsf.z, hsf.y - hsf.x);
    }

    // --- Hue Shift CMY ---
    {
        float tsn_pt_c = 1.0 - tsn_pt;
        vec3 hs_cmy_v = vec3(
            ha_cmy.x * ach_d * _odrt_spow(tsn_pt_c, 1.0 / hs_c_rng),
            ha_cmy.y * ach_d * _odrt_spow(tsn_pt_c, 1.0 / hs_m_rng),
            ha_cmy.z * ach_d * _odrt_spow(tsn_pt_c, 1.0 / hs_y_rng));
        vec3 hsf = vec3(hs_cmy_v.x * -hs_c, hs_cmy_v.y * hs_m, hs_cmy_v.z * hs_y);
        rgb += vec3(hsf.z - hsf.y, hsf.x - hsf.z, hsf.y - hsf.x);
    }

    // --- Purity Compression ---
    float pt_lml_p = 1.0 + 4.0 * (1.0 - tsn_pt) * (pt_lml + pt_lml_r * ha_rgb_hs.x + pt_lml_g * ha_rgb_hs.y + pt_lml_b * ha_rgb_hs.z);
    float ptf = 1.0 - _odrt_spow(tsn_pt, pt_lml_p);
    float pt_lmh_p = (1.0 - ach_d * (pt_lmh_r * ha_rgb_hs.x + pt_lmh_b * ha_rgb_hs.z)) * (1.0 - pt_lmh * ach_d);
    ptf = _odrt_spow(ptf, pt_lmh_p);

    // --- Mid-Range Purity ---
    {
        float ptm_low_f = 1.0 + ptm_low * exp(-2.0 * ach_d * ach_d / ptm_low_st) * _odrt_spow(1.0 - tsn_const, 1.0 / ptm_low_rng);
        float ptm_high_f = 1.0 + ptm_high * exp(-2.0 * ach_d * ach_d / ptm_high_st) * _odrt_spow(tsn_pt, 1.0 / (4.0 * ptm_high_rng));
        ptf *= ptm_low_f * ptm_high_f;
    }

    // Lerp ratios to peak achromatic
    rgb = rgb * ptf + 1.0 - ptf;

    // Inverse Rendering Space
    sat_L = dot(rgb, rs_w);
    rgb = (sat_L * rs_sa - rgb) / (rs_sa - 1.0);

    // --- Display Gamut Conversion ---
    // P3D65 -> CIE XYZ (D65 whitepoint, no creative white adaptation)
    rgb = _ODRT_P3D65_TO_XYZ * rgb;
    // CIE XYZ -> Output Colorspace
    if (targetIsRec709()) rgb = colors_colorspaces_convert(COLORS_COLORSPACES_CIE_XYZ, COLORS_COLORSPACES_SRGB, rgb);
    else rgb = inverse(targetGamutToXYZ()) * rgb;

    // --- Post Brilliance ---
    {
        vec2 brlp_opp = _odrt_opponent(rgb);
        float brlp_ach_d = length(brlp_opp) * 0.25;
        brlp_ach_d = 1.1 * (brlp_ach_d * brlp_ach_d / (brlp_ach_d + 0.1));
        vec3 brlp_ha_rgb = ach_d * ha_rgb;
        float brlp_m = brlp + brlp_r * brlp_ha_rgb.x + brlp_g * brlp_ha_rgb.y + brlp_b * brlp_ha_rgb.z;
        rgb *= exp2(brlp_m * brlp_ach_d * tsn);
    }

    // --- Purity Compress Low (softplus) ---
    rgb = vec3(_odrt_softplus(rgb.x, ptl_c), _odrt_softplus(rgb.y, ptl_m), _odrt_softplus(rgb.z, ptl_y));

    // --- Final Tonescale ---
    tsn *= ts_m2;
    tsn = _odrt_toe_quad(tsn, tn_toe, false);
    tsn *= ts_dsc;

    // Return from RGB ratios
    rgb *= tsn;

    rgb=saturate(rgb);
    if (!targetIsRec709()) rgb=inverse(rec709ToXYZ()) * targetGamutToXYZ() * rgb;
    #ifdef AP_HDR_OUTPUT
    rgb *= hdrPeakRatio();
    #endif
    return rgb;
}
#endif
