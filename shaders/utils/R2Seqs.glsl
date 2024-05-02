// R2 Sequence by Martine Roberts
// https://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
float r2Seq1(uint idx) {
    const float g = 1.32471795724474602596;
    const float a = 1.0 / g;
    return fract(0.5 + a * idx);
}

vec2 r2Seq2(uint idx) {
    const float g = 1.32471795724474602596;
    const vec2 a = vec2(1.0 / g, 1.0 / (g * g));
    return fract(0.5 + a * idx);
}
