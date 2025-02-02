#ifndef INCLUDE_util_Colors.glsl
#define INCLUDE_util_Colors.glsl

// This function is adopted from: https://github.com/zubetto/BlackBodyRadiation
// MIT License
// Copyright (c) 2021 Alexander
// You can find full license text in /license
vec4 colors_blackBodyRadiation(float T, float radianceMul) {
    if (T <= 0.0) return vec4(0.0);

    vec4 chromaRadiance;

//    chromaRadiance.a = (230141698.067 * radianceMul) / (exp(25724.2 / T) - 1.0);
    // 25724.2 * log2(e) = 37112.1757708
    chromaRadiance.a = (230141698.067 * radianceMul) / (exp2(37112.1757708 / T) - 1.0);

    // luminance Lv = Km*ChromaRadiance.a in cd/m2, where Km = 683.002 lm/W

    // --- Chromaticity in linear sRGB ---
    // (i.e. color luminance Y = dot({r,g,b}, {0.2126, 0.7152, 0.0722}) = 1)
    // --- R ---
    float u = 0.000536332 * T;
    chromaRadiance.r = 0.638749 + (u + 1.57533) / (u*u + 0.28664);

    // --- G ---
    u = 0.0019639 * T;
    chromaRadiance.g = 0.971029 + (u - 10.8015) / (u*u + 6.59002);

    // --- B ---
    float p = 0.00668406 * T + 23.3962;
    u = 0.000941064 * T;
    float q = u * u + 0.00100641 * T + 10.9068;
    chromaRadiance.b = 2.25398 - p / q;

    return chromaRadiance;
}

float colors_srgbLuma(vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

vec3 colors_srgbToLinear(vec3 color) {
    const float a0 = 0.000570846;
    const float a1 = -0.0403863;
    const float a2 = 0.862127;
    const float a3 = 0.178572;
    vec3 x = max(color, 0.0232545);
    vec3 x2 = x * x;
    vec3 x3 = x2 * x;
    return a0 + a1 * x + a2 * x2 + a3 * x3;
}

vec4 colors_karisAverage(vec4 color) {
    float luma = colors_srgbLuma(color.rgb);
    return color / (1.0 + luma);
}


vec3 colors_karisAverage(vec3 color) {
    float luma = colors_srgbLuma(color.rgb);
    return color / (1.0 + luma);
}

#endif