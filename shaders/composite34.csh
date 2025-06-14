#version 460 compatibility

#include "/util/Rand.glsl"
#include "/util/Math.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

uniform sampler2D usam_temp1;
uniform sampler2D usam_gbufferViewZ;

layout(rgba16f) uniform restrict image2D uimg_main;

#define SAMPLE_COUNT 8
#define DISK_RADIUS 16.0
const float INV_SQRT_SAMPLE_COUNT = inversesqrt(float(SAMPLE_COUNT));
const float BASE_WEIGHT = rcp(SAMPLE_COUNT);

vec4 sampleBokeh(vec2 centerTexelPos, vec2 sampleTexelPos, float centerCoc, float centerViewZ) {
    vec2 sampleScreenPos = sampleTexelPos * global_mainImageSizeRcp;
    vec4 sampleData = texture(usam_temp1, sampleScreenPos);
    vec3 sampleColor = sampleData.rgb;
    float sampleCoc = sampleData.a;
    float sampleViewZ = texture(usam_gbufferViewZ, sampleScreenPos, 0).r;
    vec2 texelPosDiff = sampleTexelPos - centerTexelPos;
    float texelDistSq = dot(texelPosDiff, texelPosDiff);
    float weight = BASE_WEIGHT;
    weight *= smoothstep(pow2(sampleCoc + 0.1), pow2(sampleCoc - 0.1), texelDistSq);
    weight *= saturate(exp2(-4.0 * exp2(-centerCoc) * (centerViewZ - sampleViewZ)));
    return vec4(sampleColor * weight, weight);
}

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    if (all(lessThan(texelPos, global_mainImageSizeI))) {
        vec2 centerTexelPos = vec2(texelPos) + 0.5;

        float baseRotation = rand_stbnVec1(texelPos, frameCounter) * PI_2;

        vec4 centerColor = imageLoad(uimg_main, texelPos);
        float centerCoc = texelFetch(usam_temp1, texelPos, 0).a;
        float centerViewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;

        vec4 sum = vec4(centerColor.rgb * BASE_WEIGHT, BASE_WEIGHT);

        for (int i = 0; i < SAMPLE_COUNT; i++) {
            float theta = GOLDEN_ANGLE * i + baseRotation;
            float r = sqrt(i + 0.5) * INV_SQRT_SAMPLE_COUNT;
            vec2 u = r * vec2(cos(theta), sin(theta));
            vec2 offset = u * DISK_RADIUS;

            vec2 sampleTexelPos = centerTexelPos + offset;
            sum += sampleBokeh(centerTexelPos, sampleTexelPos, centerCoc, centerViewZ);
        }

        sum.rgb /= sum.a;

        imageStore(uimg_main, texelPos, vec4(sum.rgb, centerColor.a));
    }
}