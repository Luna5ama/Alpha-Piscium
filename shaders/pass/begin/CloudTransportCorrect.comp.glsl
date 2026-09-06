#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable
#define CT_STORAGE restrict buffer
#include "/techniques/atmospherics/clouds/Transport.glsl"

#ifdef CT_COARSE
layout(local_size_x = CT_COARSE_LANES) in;
#else
layout(local_size_x = 64) in;
#endif
#if defined(CT_REDUCE) || defined(CT_COARSE)
const ivec3 workGroups = ivec3(1, 1, 1);
#elif CT_LEVEL == 0
const ivec3 workGroups = ivec3(CT_GROUPS, 1, 1);
#elif CT_LEVEL == 1
#if CT_N == 64
const ivec3 workGroups = ivec3(512, 1, 1);
#else
const ivec3 workGroups = ivec3(64, 1, 1);
#endif
#elif CT_LEVEL == 2
#if CT_N == 64
const ivec3 workGroups = ivec3(64, 1, 1);
#else
const ivec3 workGroups = ivec3(8, 1, 1);
#endif
#elif CT_LEVEL == 3 && CT_N == 64
const ivec3 workGroups = ivec3(8, 1, 1);
#else
const ivec3 workGroups = ivec3(1, 1, 1);
#endif

#if defined(CT_INIT) || defined(CT_RESIDUAL) || defined(CT_RESOLVE) || defined(CT_DOT) || defined(CT_APPLY)
shared vec3 angularMean[CT_ANGLES * 64];

void ct_cacheMean(ivec3 p, bool correction) {
    int lane = int(gl_LocalInvocationIndex);
    int i = ct_index(p, CT_N);
    for (int angle = 0; angle < CT_ANGLES; ++angle) {
        #ifdef CT_INIT
        angularMean[angle * 64 + lane] = ctSky(angle * CT_CELLS + i).rgb;
        #elif defined(CT_RESIDUAL)
        angularMean[angle * 64 + lane] = ctLifted(angle * CT_CELLS + i).rgb;
        #else
        angularMean[angle * 64 + lane] = ct_cellMean(p, angle, correction);
        #endif
    }
    barrier();
}
#endif

#if defined(CT_INIT) || defined(CT_RESIDUAL) || defined(CT_RESOLVE)
void main() {
    if (ct_skipIteration()) return;
    int i = int(gl_GlobalInvocationID.x);
    ivec3 p = ct_position(i, CT_N);
    float density = ctCoefficient(i, 0).a;
    ct_cacheMean(p, false);
    vec3 residual = vec3(0.0);
    for (int angle = 0; angle < CT_ANGLES; ++angle) {
        vec3 convolution = vec3(0.0);
        for (int incoming = 0; incoming < CT_ANGLES; ++incoming) {
            convolution += ctPhase(angle * CT_ANGLES + incoming).rgb * angularMean[incoming * 64 + int(gl_LocalInvocationIndex)];
        }
        int a = angle * CT_CELLS + i;
        #ifdef CT_INIT
        ctSky(a) = vec4(convolution, 0.0);
        ctSource(a) = vec4(ct_external(i, angle) + density * CT_SCATTERING * ctRadiance(a).rgb, 0.0);
        #elif defined(CT_RESOLVE)
        // P S still requires one scattering at the high-resolution point.
        ctSource(a) = vec4(convolution, 0.0);
        ctLifted(a) = vec4(angularMean[angle * 64 + int(gl_LocalInvocationIndex)], 0.0);
        #else
        vec3 reduced = convolution - ct_forwardFraction(angle) * angularMean[angle * 64 + int(gl_LocalInvocationIndex)];
        vec3 q = ct_external(i, angle) + density * CT_SCATTERING * reduced;
        ctSource(a) = vec4(q, 0.0);
        residual += CT_W * (q - ct_streaming(p, angle, false));
        #endif
    }
    #ifdef CT_RESIDUAL
    vec3 bs = vec3(0.0), bf = vec3(0.0), xs = vec3(0.0), xf = vec3(0.0);
    if (density > 0.0) {
        vec3 w = ctCoefficient(i, 2).rgb;
        bs = residual * w * ct_volume(p, 0);
        bf = residual * (1.0 - w) * ct_volume(p, 0);
        vec3 ds, df, exchange;
        ct_diagonal(p, 0, ds, df, exchange);
        ct_blockSolve(bs, bf, ds, df, exchange, xs, xf);
    }
    ctRhs(2 * i) = vec4(bs, 0.0);
    ctRhs(2 * i + 1) = vec4(bf, 0.0);
    ctError(2 * i) = vec4(xs, 0.0);
    ctError(2 * i + 1) = vec4(xf, 0.0);
    #endif
}

#elif defined(CT_RESTRICT)
void main() {
    if (ct_skipIteration()) return;
    int i = int(gl_GlobalInvocationID.x);
    int n = CT_N >> CT_LEVEL;
    if (i >= n * n * n) return;
    ivec3 p = ct_position(i, n);
    int destination = ct_levelOffset(CT_LEVEL) + i;
    vec3 bs = vec3(0.0), bf = vec3(0.0), xs = vec3(0.0), xf = vec3(0.0);
    if (ctCoefficient(destination, 0).a > 0.0) {
        for (int corner = 0; corner < 8; ++corner) {
            ivec3 q = p * 2 + (ivec3(corner) >> ivec3(0, 1, 2) & 1);
            int j = ct_levelOffset(CT_LEVEL - 1) + ct_index(q, n * 2);
            vec3 as, af;
            ct_apply(q, CT_LEVEL - 1, as, af);
            bs += ctRhs(2 * j).rgb - as;
            bf += ctRhs(2 * j + 1).rgb - af;
        }
        vec3 ds, df, exchange;
        ct_diagonal(p, CT_LEVEL, ds, df, exchange);
        ct_blockSolve(bs, bf, ds, df, exchange, xs, xf);
    }
    ctRhs(2 * destination) = vec4(bs, 0.0);
    ctRhs(2 * destination + 1) = vec4(bf, 0.0);
    ctError(2 * destination) = vec4(xs, 0.0);
    ctError(2 * destination + 1) = vec4(xf, 0.0);
}

#elif defined(CT_PROLONG)
void main() {
    if (ct_skipIteration()) return;
    int i = int(gl_GlobalInvocationID.x);
    int n = CT_N >> CT_LEVEL;
    ivec3 p = ct_position(i, n);
    int destination = ct_levelOffset(CT_LEVEL) + i;
    if (ctCoefficient(destination, 0).a == 0.0) return;
    ivec3 parent = p / 2;
    ivec3 base = parent - ivec3(equal(p & 1, ivec3(0)));
    vec3 point = ct_center(p, CT_LEVEL);
    ivec3 low = clamp(base, ivec3(0), ivec3(n / 2 - 1));
    ivec3 high = clamp(base + 1, ivec3(0), ivec3(n / 2 - 1));
    vec3 lowCenter = ct_center(low, CT_LEVEL + 1);
    vec3 highCenter = ct_center(high, CT_LEVEL + 1);
    vec3 f = vec3(0.0);
    for (int axis = 0; axis < 3; ++axis) {
        if (low[axis] != high[axis]) f[axis] = (point[axis] - lowCenter[axis]) / (highCenter[axis] - lowCenter[axis]);
    }
    vec3 xs = vec3(0.0), xf = vec3(0.0);
    for (int corner = 0; corner < 8; ++corner) {
        ivec3 offset = ivec3(corner) >> ivec3(0, 1, 2) & 1;
        ivec3 q = mix(low, high, bvec3(offset));
        int j = ct_levelOffset(CT_LEVEL + 1) + ct_index(q, n / 2);
        vec3 weights = mix(1.0 - f, f, vec3(offset));
        float weight = weights.x * weights.y * weights.z;
        xs += weight * ctError(2 * j).rgb;
        xf += weight * ctError(2 * j + 1).rgb;
    }
    ctError(2 * destination).rgb += xs;
    ctError(2 * destination + 1).rgb += xf;
}

#elif defined(CT_SMOOTH)
void main() {
    if (ct_skipIteration()) return;
    int i = int(gl_GlobalInvocationID.x);
    int n = CT_N >> CT_LEVEL;
    ivec3 p = ct_position(i, n);
    if (((p.x + p.y + p.z) & 1) != CT_COLOR) return;
    int base = ct_levelOffset(CT_LEVEL);
    i += base;
    if (ctCoefficient(i, 0).a == 0.0) return;
    vec3 bs = ctRhs(2 * i).rgb, bf = ctRhs(2 * i + 1).rgb;
    for (int f = 0; f < 6; ++f) {
        ivec3 offset = ct_neighbor(f);
        ivec3 q = p + offset;
        if (!ct_inside(q, n)) continue;
        int j = base + ct_index(q, n);
        if (ctCoefficient(j, 0).a == 0.0) continue;
        vec3 gs, gf;
        ct_face(p, CT_LEVEL, offset, gs, gf);
        bs += gs * ctError(2 * j).rgb;
        bf += gf * ctError(2 * j + 1).rgb;
    }
    vec3 ds, df, exchange, xs, xf;
    ct_diagonal(p, CT_LEVEL, ds, df, exchange);
    ct_blockSolve(bs, bf, ds, df, exchange, xs, xf);
    ctError(2 * i) = vec4(xs, 0.0);
    ctError(2 * i + 1) = vec4(xf, 0.0);
}

#elif defined(CT_COARSE)
shared vec3 cgPs[CT_COARSE_LANES], cgPf[CT_COARSE_LANES], cgReduction[CT_COARSE_LANES];
shared vec3 cgFaceS[3 * CT_COARSE_N * CT_COARSE_N * CT_COARSE_N];
shared vec3 cgFaceF[3 * CT_COARSE_N * CT_COARSE_N * CT_COARSE_N];

vec3 ct_sum(vec3 value) {
    value = subgroupAdd(value);
    if (subgroupElect()) cgReduction[gl_SubgroupID] = value;
    barrier();
    if (gl_SubgroupID == 0u) {
        value = vec3(0.0);
        for (uint i = gl_SubgroupInvocationID; i < gl_NumSubgroups; i += gl_SubgroupSize) value += cgReduction[i];
        value = subgroupAdd(value);
        if (subgroupElect()) cgReduction[0] = value;
    }
    barrier();
    vec3 result = cgReduction[0];
    barrier();
    return result;
}

vec3 ct_quotient(vec3 a, vec3 b) {
    return vec3(b.x > 0.0 ? a.x / b.x : 0.0, b.y > 0.0 ? a.y / b.y : 0.0, b.z > 0.0 ? a.z / b.z : 0.0);
}

void main() {
    if (ct_skipIteration()) return;
    int lane = int(gl_LocalInvocationIndex);
    int level = CT_COARSE_LEVEL;
    int base = ct_levelOffset(level);
    int i = base + lane;
    ivec3 p = ct_position(lane, CT_COARSE_N);
    bool occupied = ctCoefficient(i, 0).a > 0.0;
    vec3 xs = vec3(0.0), xf = vec3(0.0);
    vec3 rs = ctRhs(2 * i).rgb;
    vec3 rf = ctRhs(2 * i + 1).rgb;
    vec3 ds = vec3(1.0), df = vec3(1.0), exchange = vec3(0.0);
    if (occupied) ct_diagonal(p, level, ds, df, exchange);
    // Each lane owns its solution/residual. Only search directions and one
    // copy of each shared face need shared memory.
    for (int axis = 0; axis < 3; ++axis) {
        vec3 gs = vec3(0.0), gf = vec3(0.0);
        ivec3 offset = ivec3(equal(ivec3(axis), ivec3(0, 1, 2)));
        ivec3 q = p + offset;
        if (occupied && ct_inside(q, CT_COARSE_N) && ctCoefficient(base + ct_index(q, CT_COARSE_N), 0).a > 0.0) ct_face(p, level, offset, gs, gf);
        cgFaceS[lane * 3 + axis] = gs;
        cgFaceF[lane * 3 + axis] = gf;
    }
    vec3 zs, zf;
    ct_blockSolve(rs, rf, ds, df, exchange, zs, zf);
    cgPs[lane] = zs;
    cgPf[lane] = zf;
    vec3 rz = ct_sum(rs * zs + rf * zf);
    vec3 rhsNorm = ct_sum(rs * rs + rf * rf);
    vec3 residualNorm = rhsNorm;
    int iteration = 0;
    for (; iteration < 96; ++iteration) {
        if (all(lessThanEqual(residualNorm, rhsNorm * 1e-10))) break;
        vec3 aps = ds * cgPs[lane] - exchange * cgPf[lane];
        vec3 apf = df * cgPf[lane] - exchange * cgPs[lane];
        for (int f = 0; f < 6; ++f) {
            ivec3 q = p + ct_neighbor(f);
            if (!occupied || !ct_inside(q, CT_COARSE_N)) continue;
            int j = ct_index(q, CT_COARSE_N);
            int face = ((f & 1) == 0 ? lane : j) * 3 + f / 2;
            aps -= cgFaceS[face] * cgPs[j];
            apf -= cgFaceF[face] * cgPf[j];
        }
        vec3 pap = ct_sum(cgPs[lane] * aps + cgPf[lane] * apf);
        vec3 alpha = ct_quotient(rz, pap);
        xs += alpha * cgPs[lane];
        xf += alpha * cgPf[lane];
        rs -= alpha * aps;
        rf -= alpha * apf;
        ct_blockSolve(rs, rf, ds, df, exchange, zs, zf);
        vec3 rzNew = ct_sum(rs * zs + rf * zf);
        vec3 beta = ct_quotient(rzNew, rz);
        cgPs[lane] = zs + beta * cgPs[lane];
        cgPf[lane] = zf + beta * cgPf[lane];
        rz = rzNew;
        residualNorm = ct_sum(rs * rs + rf * rf);
    }
    ctError(2 * i) = vec4(xs, 0.0);
    ctError(2 * i + 1) = vec4(xf, 0.0);
    if (lane == 0) ctStats(16 + CT_ITERATION) = vec4(sqrt(ct_quotient(residualNorm, rhsNorm)), float(iteration));
}

#elif defined(CT_DOT) || defined(CT_REDUCE)
shared CTPartial partial[64];

CTPartial ct_zeroPartial() {
    CTPartial p;
    p.rd = vec4(0.0);
    p.dd = vec4(0.0);
    p.rr = vec4(0.0);
    p.limit = vec4(1.0);
    p.bb = vec4(0.0);
    p.inputPower = vec4(0.0);
    p.outputPower = vec4(0.0);
    return p;
}

CTPartial ct_addPartial(CTPartial a, CTPartial b) {
    a.rd += b.rd;
    a.dd += b.dd;
    a.rr += b.rr;
    a.limit = min(a.limit, b.limit);
    a.bb += b.bb;
    a.inputPower += b.inputPower;
    a.outputPower += b.outputPower;
    return a;
}

void main() {
    if (ct_skipIteration()) return;
    int lane = int(gl_LocalInvocationIndex);
    CTPartial sums = ct_zeroPartial();
    #ifdef CT_DOT
    int i = int(gl_GlobalInvocationID.x);
    ivec3 p = ct_position(i, CT_N);
    float volume = ct_volume(p, 0);
    float density = ctCoefficient(i, 0).a;
    vec3 h = ct_size(p, 0);
    ct_cacheMean(p, true);
    for (int angle = 0; angle < CT_ANGLES; ++angle) {
        vec3 omega = ct_direction(angle);
        vec3 rate = abs(omega) / h;
        vec3 s = ctRadiance(angle * CT_CELLS + i).rgb;
        vec3 r = ctSource(angle * CT_CELLS + i).rgb - ct_streaming(p, angle, false);
        vec3 b = ct_external(i, angle);
        vec3 delta = ctLifted(angle * CT_CELLS + i).rgb;
        vec3 phaseDelta = vec3(0.0);
        for (int incoming = 0; incoming < CT_ANGLES; ++incoming) phaseDelta += ctPhase(angle * CT_ANGLES + incoming).rgb * angularMean[incoming * 64 + lane];
        phaseDelta -= ct_forwardFraction(angle) * angularMean[angle * 64 + lane];
        vec3 ad = ct_streaming(p, angle, true) - density * CT_SCATTERING * phaseDelta;
        vec3 escape = vec3(0.0);
        vec3 correctionEscape = vec3(0.0);
        for (int axis = 0; axis < 3; ++axis) {
            ivec3 stepV = ivec3(equal(ivec3(axis), ivec3(0, 1, 2))) * ivec3(sign(omega));
            if (!ct_inside(p + stepV, CT_N)) {
                escape += rate[axis] * s;
                correctionEscape += rate[axis] * delta;
            }
        }
        sums.rd.rgb += r * ad * volume * CT_W;
        sums.dd.rgb += ad * ad * volume * CT_W;
        sums.rr.rgb += r * r * volume * CT_W;
        sums.bb.rgb += b * b * volume * CT_W;
        sums.inputPower.rgb += b * volume * CT_W;
        sums.outputPower.rgb += (density * CT_ABSORPTION * ct_cellMean(p, angle, false) + escape) * volume * CT_W;
        vec3 correctionPower = (density * CT_ABSORPTION * angularMean[angle * 64 + lane] + correctionEscape) * volume * CT_W;
        sums.rd.a += correctionPower.r;
        sums.dd.a += correctionPower.g;
        sums.rr.a += correctionPower.b;
        for (int c = 0; c < 3; ++c) if (delta[c] < 0.0) sums.limit[c] = min(sums.limit[c], -0.99 * s[c] / delta[c]);
    }
    #else
    for (int i = lane; i < CT_GROUPS; i += 64) {
        CTPartial p = CTPartial(ctPartial(i, 0), ctPartial(i, 1), ctPartial(i, 2), ctPartial(i, 3), ctPartial(i, 4), ctPartial(i, 5), ctPartial(i, 6));
        sums = ct_addPartial(sums, p);
    }
    #endif
    partial[lane] = sums;
    barrier();
    for (int stride = 32; stride > 0; stride >>= 1) {
        if (lane < stride) partial[lane] = ct_addPartial(partial[lane], partial[lane + stride]);
        barrier();
    }
    if (lane != 0) return;
    #ifdef CT_DOT
    int destination = int(gl_WorkGroupID.x);
    ctPartial(destination, 0) = partial[0].rd;
    ctPartial(destination, 1) = partial[0].dd;
    ctPartial(destination, 2) = partial[0].rr;
    ctPartial(destination, 3) = partial[0].limit;
    ctPartial(destination, 4) = partial[0].bb;
    ctPartial(destination, 5) = partial[0].inputPower;
    ctPartial(destination, 6) = partial[0].outputPower;
    #else
    sums = partial[0];
    vec3 gamma = vec3(0.0);
    vec3 relative = vec3(0.0);
    vec3 predicted = vec3(0.0);
    for (int c = 0; c < 3; ++c) {
        if (sums.dd[c] > 0.0) gamma[c] = clamp(sums.rd[c] / sums.dd[c], 0.0, sums.limit[c]);
        if (sums.bb[c] > 0.0) {
            relative[c] = sqrt(sums.rr[c] / sums.bb[c]);
            predicted[c] = sqrt(max(sums.rr[c] - 2.0 * gamma[c] * sums.rd[c] + gamma[c] * gamma[c] * sums.dd[c], 0.0) / sums.bb[c]);
        }
    }
    #ifdef CT_FINAL
    ctStats(13) = vec4(relative, 0.0);
    ctStats(14) = sums.inputPower;
    ctStats(15) = sums.outputPower;
    #else
    ctStats(0) = vec4(gamma, 0.0);
    ctStats(1 + CT_ITERATION) = vec4(relative, 0.0);
    ctStats(5 + CT_ITERATION) = vec4(predicted, 0.0);
    ctStats(9 + CT_ITERATION) = vec4(gamma, 0.0);
    vec3 correctedPower = sums.outputPower.rgb + gamma * vec3(sums.rd.a, sums.dd.a, sums.rr.a);
    ctStats(20 + CT_ITERATION) = vec4((correctedPower - sums.inputPower.rgb) / max(sums.inputPower.rgb, vec3(1e-20)), 0.0);
    ctStats(24).x = float(CT_ITERATION + 1);
    #endif
    #endif
}

#elif defined(CT_APPLY)
void main() {
    if (ct_skipIteration()) return;
    int i = int(gl_GlobalInvocationID.x);
    ct_cacheMean(ct_position(i, CT_N), true);
    vec3 gamma = ctStats(0).rgb;
    for (int angle = 0; angle < CT_ANGLES; ++angle) {
        int a = angle * CT_CELLS + i;
        vec3 phaseDelta = vec3(0.0);
        for (int incoming = 0; incoming < CT_ANGLES; ++incoming) phaseDelta += ctPhase(angle * CT_ANGLES + incoming).rgb * angularMean[incoming * 64 + int(gl_LocalInvocationIndex)];
        phaseDelta -= ct_forwardFraction(angle) * angularMean[angle * 64 + int(gl_LocalInvocationIndex)];
        ctRadiance(a).rgb += gamma * ctLifted(a).rgb;
        ctSource(a).rgb += ctCoefficient(i, 0).a * CT_SCATTERING * gamma * phaseDelta;
    }
}
#endif
