// Screen Space Indirect Lighting with Visiblity Bitmask
// References:
// https://arxiv.org/pdf/2301.11376
// https://cdrinmatane.github.io/posts/ssaovb-code/
// https://cybereality.com/screen-space-indirect-lighting-with-visibility-bitmask-improvement-to-gtao-ssao-real-time-ambient-occlusion-algorithm-glsl-shader-implementation/
// https://www.shadertoy.com/view/XXGSDd
#include "../_Util.glsl"

uniform sampler2D usam_viewZ;
uniform sampler2D usam_temp1;
uniform sampler2D usam_temp2;

const vec2 RADIUS_SQ = vec2(SETTING_SSVBIL_RADIUS * SETTING_SSVBIL_RADIUS, SETTING_SSVBIL_MAX_RADIUS * SETTING_SSVBIL_MAX_RADIUS);

in vec2 frag_texCoord;

/* RENDERTARGETS:11,14 */
layout(location = 0) out vec4 rt_bentNormal;
layout(location = 1) out vec4 rt_out;

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
    #else
    #error "Invalid SSVBIL_SAMPLE_STEPS"
    #endif
    y = clamp(y, SSVBIL_SAMPLE_STEPS, 32768.0);
    return saturate(a0 * log2(a1 * y + a2));
}

float lodTexelSize(float lod) {
    return exp2(lod);
}

// https://cdrinmatane.github.io/posts/ssaovb-code/
const float SECTOR_COUNT_F = 32.0;
const uint SECTOR_COUNT = 32u;
uint calcSectorBits(float minHorizon, float maxHorizon) {
    uint startBit = uint(minHorizon * SECTOR_COUNT_F);
    uint horizonAngle = uint(ceil((maxHorizon - minHorizon) * SECTOR_COUNT_F));
    uint angleBit = uint(0xFFFFFFFFu >> (SECTOR_COUNT - horizonAngle));
    angleBit = uint(-int(horizonAngle > 0u)) & angleBit;
    uint currentBitfield = angleBit << startBit;
    return currentBitfield;
}

#define NOISE_FRAME frameCounter
//#define NOISE_FRAME 0u

const float WEIGHTS[32] = float[](
    0.049067674,
    0.048949466,
    0.048713334,
    0.048359848,
    0.047889858,
    0.047304497,
    0.046605176,
    0.045793579,
    0.044871661,
    0.043841643,
    0.042706007,
    0.041467489,
    0.040129071,
    0.03869398,
    0.037165671,
    0.035547826,
    0.033844344,
    0.032059328,
    0.030197078,
    0.028262081,
    0.026258998,
    0.024192654,
    0.022068029,
    0.019890239,
    0.017664533,
    0.015396271,
    0.013090917,
    0.010754027,
    0.0083912296,
    0.0060082167,
    0.0036107295,
    0.001204543
);

// [Eberly2014] GPGPU Programming for Games and Science
float acosFastPositive(float inX) {
    float x = abs(inX);
    float res = -0.156583f * x + fsl_HALF_PI;
    res *= sqrt(1.0f - x);
    return res;
}

vec2 acosFastPositive(vec2 inX) {
    vec2 x = abs(inX);
    vec2 res = -0.156583f * x + fsl_HALF_PI;
    res *= sqrt(1.0f - x);
    return res;
}

vec3 acosFastPositive(vec3 inX) {
    vec3 x = abs(inX);
    vec3 res = -0.156583f * x + fsl_HALF_PI;
    res *= sqrt(1.0f - x);
    return res;
}

float calcHorizonAngles(vec3 projNormal, vec3 pos) {
    float cosH = dot(pos, projNormal) * inversesqrt(dot(pos, pos));
    cosH = saturate(cosH);
    return acosFastPositive(cosH) * RCP_PI_HALF_CONST;
}

void main() {
    rt_bentNormal = vec4(0.0, 1.0, 0.0, 0.0);
    rt_out = vec4(0.0, 0.0, 0.0, 1.0);

    float centerViewZ = textureLod(usam_viewZ, frag_texCoord, 0.0).r;

    if (centerViewZ < 0.0) {
        vec3 centerViewCoord = coords_toViewCoord(frag_texCoord, centerViewZ, gbufferProjectionInverse);
        vec3 centerViewNormal = textureLod(usam_temp1, frag_texCoord, 0.0).rgb;
        rt_bentNormal.xyz = centerViewNormal;
        vec3 centerViewDir = normalize(-centerViewCoord);

        float sampleAngleDelta = 2.0 * PI_CONST / SSVBIL_SAMPLE_SLICES;
        float initialAngle = rand_IGN(gl_FragCoord.xy, NOISE_FRAME) * sampleAngleDelta;
        vec2 sphereRadius = (SETTING_SSVBIL_RADIUS / -centerViewCoord.z) * vec2(gbufferProjection[0][0], gbufferProjection[1][1]);
        sphereRadius *= textureSize(usam_viewZ, 0).xy;

        uvec2 hashKey = (uvec2(gl_FragCoord.xy) & uvec2(31u)) ^ (NOISE_FRAME & 0xFFFFFFF0u);
        uint r2Index = (rand_hash21(hashKey) & 65535u) + NOISE_FRAME;
        float baseSampleLod = rand_r2Seq1(r2Index) * 0.5 - 0.25;

        rt_out.a = 0.0;

        for (uint sliceIndex = 0; sliceIndex < SSVBIL_SAMPLE_SLICES; sliceIndex++) {
            float sampleAngle = initialAngle + sampleAngleDelta * float(sliceIndex);
            vec2 sampleDir = vec2(cos(sampleAngle), sin(sampleAngle));

            vec3 planeNormal = normalize(cross(vec3(sampleDir, 0.0), centerViewDir));
            vec3 sliceTangent = cross(centerViewDir, planeNormal);
            vec3 projNormal = normalize(centerViewNormal - planeNormal * dot(centerViewNormal, planeNormal));
            vec3 realTangent = -cross(projNormal, planeNormal);

            float maxDist = length(sampleDir * sphereRadius);

            float lodStep = radiusToLodStep(maxDist);
            float maxLod = float(textureQueryLevels(usam_viewZ)) - 1.0;
            float sampleLod = baseSampleLod;

            float sampleTexelDist = 0.5;
            uint aoSectionBits = 0u;
            uint giSectionBits = 0u;

            for (int stepIndex = 0; stepIndex < SSVBIL_SAMPLE_STEPS; stepIndex++) {
                float sampleLodTexelSize = lodTexelSize(sampleLod) * 1.0;
                float stepTexelSize = sampleLodTexelSize * 0.5;
                sampleTexelDist += stepTexelSize;

                vec2 sampleTexelCoord = sampleDir * sampleTexelDist + gl_FragCoord.xy;
                vec2 sampleUV = sampleTexelCoord / textureSize(usam_viewZ, 0).xy;

                float sampleViewZ = textureLod(usam_viewZ, sampleUV, round(sampleLod * 0.5)).r;
                vec3 sampleViewXYZ = coords_toViewCoord(sampleUV, sampleViewZ, gbufferProjectionInverse);
                vec3 diff = sampleViewXYZ - centerViewCoord;
                float distSq = dot(diff, diff);

                if (distSq <= RADIUS_SQ.y) {
                    vec3 frontPos = diff;
                    float frontH = calcHorizonAngles(projNormal, frontPos);

                    uint aoStepSectorBits;
                    {
                        vec3 backOffset = centerViewDir * SETTING_SSVBIL_AO_THICKNESS;
                        vec3 backPos = diff - backOffset;
                        float backH = calcHorizonAngles(projNormal, backPos);
                        aoStepSectorBits = calcSectorBits(min(frontH, backH), max(frontH, backH));
                    }
                    aoSectionBits |= aoStepSectorBits;

                    uint giStepSectorBits;
                    {
                        vec3 backOffset = centerViewDir * SETTING_SSVBIL_GI_THICKNESS * float(stepIndex);
                        vec3 backPos = diff - backOffset;
                        float backH = calcHorizonAngles(projNormal, backPos);
                        giStepSectorBits = calcSectorBits(min(frontH, backH), max(frontH, backH));
                    }

                    uint ilBit = bitCount(giStepSectorBits & ~giSectionBits);
                    giSectionBits |= giStepSectorBits;

                    uint ilCond = uint(ilBit != 0);
                    ilCond &= uint(all(greaterThanEqual(sampleUV, vec2(0.0))));
                    ilCond &= uint(all(lessThanEqual(sampleUV, vec2(1.0))));

                    if (bool(ilCond)) {
                        vec4 sample1 = textureLod(usam_temp1, sampleUV, round(sampleLod * 0.5));
                        vec4 sample2 = textureLod(usam_temp2, sampleUV, round(sampleLod * 0.5));
                        vec3 sampleNormal = sample1.rgb;
                        vec3 direct = sample2.rgb;

                        float ilBitCoeff = float(ilBit) * (1.0 / 32.0);
                        float rcpDist = fastRcpSqrtNR0(distSq);
                        float emitterCos = dot(sampleNormal, -diff) * rcpDist;
                        emitterCos = mix(0.5 + linearStep(-0.6, 0.0, emitterCos), emitterCos, step(sample1.a, 0.0));
                        emitterCos = saturate(emitterCos);
                        float receiverCos = saturate(dot(centerViewNormal, diff) * rcpDist);

                        rt_out.rgb += ilBitCoeff * emitterCos * receiverCos * direct;
                    }
                }
                sampleLod = min(sampleLod + lodStep, maxLod);
                sampleTexelDist += stepTexelSize;
            }

            // Cosine weighted hemisphere as mentioned in GT-VBAO but blute forced
            for (uint bitIndex = 0u; bitIndex < 32; bitIndex++) {
                float indexF = float(bitIndex) / 31.0;
                float bitFlag = float((aoSectionBits >> bitIndex) & 1u);
                rt_bentNormal.xyz += bitFlag * normalize(mix(realTangent, projNormal, indexF));
                rt_out.a += bitFlag * WEIGHTS[bitIndex];
            }
        }

        rt_bentNormal.xyz = normalize(rt_bentNormal.xyz);
        rt_bentNormal.xyz = mat3(gbufferModelViewInverse) * rt_bentNormal.xyz;
        rt_bentNormal.xyz = rt_bentNormal.xyz * 0.5 + 0.5;

        rt_out.rgb /= float(SSVBIL_SAMPLE_SLICES);
        rt_out.rgb *= 16.0;

        rt_out.a /= float(SSVBIL_SAMPLE_SLICES);
        rt_out.a = linearStep(0.1, 1.0, rt_out.a);
        rt_out.a = saturate(1.0 - rt_out.a);
        rt_out.a = pow(rt_out.a, SETTING_SSVBIL_AO_STRENGTH);
    }
}