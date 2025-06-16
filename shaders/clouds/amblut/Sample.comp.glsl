#include "Common.glsl"
#include "../Common.glsl"

#define ATMOSPHERE_RAYMARCHING_SKY a
#include "/atmosphere/Raymarching.glsl"

#include "/util/Celestial.glsl"
#include "/util/Rand.glsl"

layout(local_size_x = 256) in;
const ivec3 workGroups = ivec3(1, 1, 1);

void main() {
    AtmosphereParameters atmosphere = getAtmosphereParameters();

    float lutHeight = atmosphere.bottom + 9.0;

    vec3 randV = rand_r2Seq3((gl_GlobalInvocationID.x + SAMPLE_COUNT * frameCounter) & 0xFFFFu);
    float randA = randV.x;
    float randB = randV.y;
    float theta = 2.0 * PI * randA;
    float phi = acos(1.0 - 2.0 * randB);
    float cosPhi = cos(phi);
    float sinPhi = sin(phi);
    float cosTheta = cos(theta);
    float sinTheta = sin(theta);

    vec3 rayDir = vec3(cosTheta * sinPhi, sinTheta * sinPhi, cosPhi);

    RaymarchParameters params = raymarchParameters_init();
    params.rayStart = vec3(0.0, lutHeight, 0.0);
    params.steps = 64u;

    LightParameters sunParams = lightParameters_init(atmosphere, SUN_ILLUMINANCE * PI, uval_sunDirWorld, rayDir);
    LightParameters moonParams = lightParameters_init(atmosphere, MOON_ILLUMINANCE, uval_moonDirWorld, rayDir);
    ScatteringParameters scatteringParams = scatteringParameters_init(sunParams, moonParams, 1.0);

    ScatteringResult result = scatteringResult_init();
    const vec3 earthCenter = vec3(0.0);
    if (setupRayEnd(atmosphere, params, rayDir)) {
        result = raymarchSky(atmosphere, params, scatteringParams);
    }

    ssbo_ambLUTWorkingBuffer.rayDir[gl_GlobalInvocationID.x] = vec2(phi, theta);
    ssbo_ambLUTWorkingBuffer.inSctr[gl_GlobalInvocationID.x] = result.inScattering;
}
