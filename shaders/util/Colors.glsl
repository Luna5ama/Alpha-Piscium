#ifndef INCLUDE_Colors.glsl
#define INCLUDE_Colors.glsl

// (695700 / 149600000)^2
const float OMEGA_SUN = 2.1626230107380823014670136406531e-5;

// This function is adopted from: https://github.com/zubetto/BlackBodyRadiation
// MIT License
// Copyright (c) 2021 Alexander
// You can find the full license text in /licenses/MIT.txt
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

float colors_linearSRGBToLuminance(vec3 color) {
    return dot(color, vec3(0.212639005872, 0.715168678768, 0.0721923153607));
}

#endif