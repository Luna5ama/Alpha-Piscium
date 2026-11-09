// Bloom DPF recipe f39980f47e927f18f8f749e9eeb10e27ce10e38c5588e6ed52e8b3a3d88e7dd1
vec3 bloom_filter(vec2 center, vec4 tile) {
    vec3 value = vec3(0.0);
#ifdef BLOOM_DOWNSAMPLE
    value += bloom_sample(center + vec2(0.000000000, 0.000000000), tile) * vec3(0.215983973, 0.215983973, 0.215983973);
    value += bloom_sample(center + vec2(-0.818871032, -0.818871032), tile) * vec3(0.196004007, 0.196004007, 0.196004007);
    value += bloom_sample(center + vec2(-0.818871032, 0.818871032), tile) * vec3(0.196004007, 0.196004007, 0.196004007);
    value += bloom_sample(center + vec2(0.818871032, -0.818871032), tile) * vec3(0.196004007, 0.196004007, 0.196004007);
    value += bloom_sample(center + vec2(0.818871032, 0.818871032), tile) * vec3(0.196004007, 0.196004007, 0.196004007);
#else
#if BLOOM_PASS == 1
    value += bloom_sample(center + vec2(-1.250000000, 0.000000000), tile) * vec3(0.067822946, 0.067822946, 0.067822946);
    value += bloom_sample(center + vec2(1.250000000, 0.000000000), tile) * vec3(0.067822946, 0.067822946, 0.067822946);
    value += bloom_sample(center + vec2(0.000000000, -1.250000000), tile) * vec3(0.067822946, 0.067822946, 0.067822946);
    value += bloom_sample(center + vec2(0.000000000, 1.250000000), tile) * vec3(0.067822946, 0.067822946, 0.067822946);
    value += bloom_sample(center + vec2(-0.457436126, -0.457436126), tile) * vec3(0.182177054, 0.182177054, 0.182177054);
    value += bloom_sample(center + vec2(-0.457436126, 0.457436126), tile) * vec3(0.182177054, 0.182177054, 0.182177054);
    value += bloom_sample(center + vec2(0.457436126, -0.457436126), tile) * vec3(0.182177054, 0.182177054, 0.182177054);
    value += bloom_sample(center + vec2(0.457436126, 0.457436126), tile) * vec3(0.182177054, 0.182177054, 0.182177054);
#elif BLOOM_PASS == 2
    value += bloom_sample(center + vec2(-1.250000000, 0.000000000), tile) * vec3(0.067822946, 0.067822946, 0.067822946);
    value += bloom_sample(center + vec2(1.250000000, 0.000000000), tile) * vec3(0.067822946, 0.067822946, 0.067822946);
    value += bloom_sample(center + vec2(0.000000000, -1.250000000), tile) * vec3(0.067822946, 0.067822946, 0.067822946);
    value += bloom_sample(center + vec2(0.000000000, 1.250000000), tile) * vec3(0.067822946, 0.067822946, 0.067822946);
    value += bloom_sample(center + vec2(-0.457436126, -0.457436126), tile) * vec3(0.182177054, 0.182177054, 0.182177054);
    value += bloom_sample(center + vec2(-0.457436126, 0.457436126), tile) * vec3(0.182177054, 0.182177054, 0.182177054);
    value += bloom_sample(center + vec2(0.457436126, -0.457436126), tile) * vec3(0.182177054, 0.182177054, 0.182177054);
    value += bloom_sample(center + vec2(0.457436126, 0.457436126), tile) * vec3(0.182177054, 0.182177054, 0.182177054);
#elif BLOOM_PASS == 3
    value += bloom_sample(center + vec2(-1.250000000, 0.000000000), tile) * vec3(0.067822946, 0.067822946, 0.067822946);
    value += bloom_sample(center + vec2(1.250000000, 0.000000000), tile) * vec3(0.067822946, 0.067822946, 0.067822946);
    value += bloom_sample(center + vec2(0.000000000, -1.250000000), tile) * vec3(0.067822946, 0.067822946, 0.067822946);
    value += bloom_sample(center + vec2(0.000000000, 1.250000000), tile) * vec3(0.067822946, 0.067822946, 0.067822946);
    value += bloom_sample(center + vec2(-0.457436126, -0.457436126), tile) * vec3(0.182177054, 0.182177054, 0.182177054);
    value += bloom_sample(center + vec2(-0.457436126, 0.457436126), tile) * vec3(0.182177054, 0.182177054, 0.182177054);
    value += bloom_sample(center + vec2(0.457436126, -0.457436126), tile) * vec3(0.182177054, 0.182177054, 0.182177054);
    value += bloom_sample(center + vec2(0.457436126, 0.457436126), tile) * vec3(0.182177054, 0.182177054, 0.182177054);
#elif BLOOM_PASS == 4
    value += bloom_sample(center + vec2(-1.250000000, 0.000000000), tile) * vec3(0.067822946, 0.067822946, 0.067822946);
    value += bloom_sample(center + vec2(1.250000000, 0.000000000), tile) * vec3(0.067822946, 0.067822946, 0.067822946);
    value += bloom_sample(center + vec2(0.000000000, -1.250000000), tile) * vec3(0.067822946, 0.067822946, 0.067822946);
    value += bloom_sample(center + vec2(0.000000000, 1.250000000), tile) * vec3(0.067822946, 0.067822946, 0.067822946);
    value += bloom_sample(center + vec2(-0.457436126, -0.457436126), tile) * vec3(0.182177054, 0.182177054, 0.182177054);
    value += bloom_sample(center + vec2(-0.457436126, 0.457436126), tile) * vec3(0.182177054, 0.182177054, 0.182177054);
    value += bloom_sample(center + vec2(0.457436126, -0.457436126), tile) * vec3(0.182177054, 0.182177054, 0.182177054);
    value += bloom_sample(center + vec2(0.457436126, 0.457436126), tile) * vec3(0.182177054, 0.182177054, 0.182177054);
#elif BLOOM_PASS == 5
    value += bloom_sample(center + vec2(-1.250000000, 0.000000000), tile) * vec3(0.067822946, 0.067822946, 0.067822946);
    value += bloom_sample(center + vec2(1.250000000, 0.000000000), tile) * vec3(0.067822946, 0.067822946, 0.067822946);
    value += bloom_sample(center + vec2(0.000000000, -1.250000000), tile) * vec3(0.067822946, 0.067822946, 0.067822946);
    value += bloom_sample(center + vec2(0.000000000, 1.250000000), tile) * vec3(0.067822946, 0.067822946, 0.067822946);
    value += bloom_sample(center + vec2(-0.457436126, -0.457436126), tile) * vec3(0.182177054, 0.182177054, 0.182177054);
    value += bloom_sample(center + vec2(-0.457436126, 0.457436126), tile) * vec3(0.182177054, 0.182177054, 0.182177054);
    value += bloom_sample(center + vec2(0.457436126, -0.457436126), tile) * vec3(0.182177054, 0.182177054, 0.182177054);
    value += bloom_sample(center + vec2(0.457436126, 0.457436126), tile) * vec3(0.182177054, 0.182177054, 0.182177054);
#elif BLOOM_PASS == 6
    value += bloom_sample(center + vec2(-1.250000000, 0.000000000), tile) * vec3(0.067822946, 0.067822946, 0.067822946);
    value += bloom_sample(center + vec2(1.250000000, 0.000000000), tile) * vec3(0.067822946, 0.067822946, 0.067822946);
    value += bloom_sample(center + vec2(0.000000000, -1.250000000), tile) * vec3(0.067822946, 0.067822946, 0.067822946);
    value += bloom_sample(center + vec2(0.000000000, 1.250000000), tile) * vec3(0.067822946, 0.067822946, 0.067822946);
    value += bloom_sample(center + vec2(-0.457436126, -0.457436126), tile) * vec3(0.182177054, 0.182177054, 0.182177054);
    value += bloom_sample(center + vec2(-0.457436126, 0.457436126), tile) * vec3(0.182177054, 0.182177054, 0.182177054);
    value += bloom_sample(center + vec2(0.457436126, -0.457436126), tile) * vec3(0.182177054, 0.182177054, 0.182177054);
    value += bloom_sample(center + vec2(0.457436126, 0.457436126), tile) * vec3(0.182177054, 0.182177054, 0.182177054);
#elif BLOOM_PASS == 7
    value += bloom_sample(center + vec2(-1.250000000, 0.000000000), tile) * vec3(0.067822946, 0.067822946, 0.067822946);
    value += bloom_sample(center + vec2(1.250000000, 0.000000000), tile) * vec3(0.067822946, 0.067822946, 0.067822946);
    value += bloom_sample(center + vec2(0.000000000, -1.250000000), tile) * vec3(0.067822946, 0.067822946, 0.067822946);
    value += bloom_sample(center + vec2(0.000000000, 1.250000000), tile) * vec3(0.067822946, 0.067822946, 0.067822946);
    value += bloom_sample(center + vec2(-0.457436126, -0.457436126), tile) * vec3(0.182177054, 0.182177054, 0.182177054);
    value += bloom_sample(center + vec2(-0.457436126, 0.457436126), tile) * vec3(0.182177054, 0.182177054, 0.182177054);
    value += bloom_sample(center + vec2(0.457436126, -0.457436126), tile) * vec3(0.182177054, 0.182177054, 0.182177054);
    value += bloom_sample(center + vec2(0.457436126, 0.457436126), tile) * vec3(0.182177054, 0.182177054, 0.182177054);
#elif BLOOM_PASS == 8
    value += bloom_sample(center + vec2(-1.250000000, 0.000000000), tile) * vec3(0.067822946, 0.067822946, 0.067822946);
    value += bloom_sample(center + vec2(1.250000000, 0.000000000), tile) * vec3(0.067822946, 0.067822946, 0.067822946);
    value += bloom_sample(center + vec2(0.000000000, -1.250000000), tile) * vec3(0.067822946, 0.067822946, 0.067822946);
    value += bloom_sample(center + vec2(0.000000000, 1.250000000), tile) * vec3(0.067822946, 0.067822946, 0.067822946);
    value += bloom_sample(center + vec2(-0.457436126, -0.457436126), tile) * vec3(0.182177054, 0.182177054, 0.182177054);
    value += bloom_sample(center + vec2(-0.457436126, 0.457436126), tile) * vec3(0.182177054, 0.182177054, 0.182177054);
    value += bloom_sample(center + vec2(0.457436126, -0.457436126), tile) * vec3(0.182177054, 0.182177054, 0.182177054);
    value += bloom_sample(center + vec2(0.457436126, 0.457436126), tile) * vec3(0.182177054, 0.182177054, 0.182177054);
#endif
#endif
    return value;
}

#if !defined(BLOOM_DOWNSAMPLE) && BLOOM_PASS == 2
void bloom_prepareInjection() {}
vec3 bloom_injection(vec2 center, vec4 tile) {
    return bloom_sampleRaw(center + vec2(0.000000000, -0.000000000), tile) * vec3(0.125000000, 0.125000000, 0.125000000);
}
#elif !defined(BLOOM_DOWNSAMPLE) && BLOOM_PASS == 3
void bloom_prepareInjection() {}
vec3 bloom_injection(vec2 center, vec4 tile) {
    return bloom_sampleRaw(center + vec2(0.000000000, -0.000000000), tile) * vec3(0.125000000, 0.125000000, 0.125000000);
}
#elif !defined(BLOOM_DOWNSAMPLE) && BLOOM_PASS == 4
void bloom_prepareInjection() {}
vec3 bloom_injection(vec2 center, vec4 tile) {
    return bloom_sampleRaw(center + vec2(0.000000000, -0.000000000), tile) * vec3(0.125000000, 0.125000000, 0.125000000);
}
#elif !defined(BLOOM_DOWNSAMPLE) && BLOOM_PASS == 5
void bloom_prepareInjection() {}
vec3 bloom_injection(vec2 center, vec4 tile) {
    return bloom_sampleRaw(center + vec2(0.000000000, -0.000000000), tile) * vec3(0.125000000, 0.125000000, 0.125000000);
}
#elif !defined(BLOOM_DOWNSAMPLE) && BLOOM_PASS == 6
void bloom_prepareInjection() {}
vec3 bloom_injection(vec2 center, vec4 tile) {
    return bloom_sampleRaw(center + vec2(0.000000000, -0.000000000), tile) * vec3(0.125000000, 0.125000000, 0.125000000);
}
#elif !defined(BLOOM_DOWNSAMPLE) && BLOOM_PASS == 7
void bloom_prepareInjection() {}
vec3 bloom_injection(vec2 center, vec4 tile) {
    return bloom_sampleRaw(center + vec2(0.000000000, -0.000000000), tile) * vec3(0.125000000, 0.125000000, 0.125000000);
}
#elif !defined(BLOOM_DOWNSAMPLE) && BLOOM_PASS == 8
void bloom_prepareInjection() {}
vec3 bloom_injection(vec2 center, vec4 tile) {
    return bloom_sampleRaw(center + vec2(0.000000000, -0.000000000), tile) * vec3(0.125000000, 0.125000000, 0.125000000);
}
#endif
#ifdef BLOOM_FIRST_UPSAMPLE
const vec3 bloom_deepestWeight = vec3(0.125000000, 0.125000000, 0.125000000);
#endif
