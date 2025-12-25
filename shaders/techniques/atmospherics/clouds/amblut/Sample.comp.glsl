#include "Common.glsl"

/*const*/
#define ATMOSPHERE_RAYMARCHING_SKY a
/*const*/
#include "/techniques/atmospherics/air/Raymarching.glsl"

#include "/util/Celestial.glsl"
#include "/util/Rand.glsl"

layout(local_size_x = 256) in;
const ivec3 workGroups = ivec3(SAMPLE_COUNT_D256, 1, 1);

void main() {
    AtmosphereParameters atmosphere = getAtmosphereParameters();

    int layerIndex = clouds_amblut_currLayerIndex();
    float lutHeight = atmosphere.bottom + clouds_amblut_height(layerIndex);

    vec3 randV = rand_r2Seq3((gl_GlobalInvocationID.x + SAMPLE_COUNT * (frameCounter / 6)) & 0xFFFFu);
    float randA = randV.x;
    float randB = randV.y;
    float theta = 2.0 * PI * randA;
    float phi = acos(1.0 - 2.0 * randB);
    float cosPhi = cos(phi);
    float sinPhi = sin(phi);
    float cosTheta = cos(theta);
    float sinTheta = sin(theta);

    vec3 rayDir = vec3(cosTheta * sinPhi, cosPhi, sinTheta * sinPhi);

    RaymarchParameters params = raymarchParameters_init();
    params.rayStart = vec3(0.0, lutHeight, 0.0);
    params.steps = 64u;

    LightParameters sunParams = lightParameters_init(atmosphere, SUN_ILLUMINANCE, uval_sunDirWorld, rayDir);
    LightParameters moonParams = lightParameters_init(atmosphere, MOON_ILLUMINANCE, uval_moonDirWorld, rayDir);
    ScatteringParameters scatteringParams = scatteringParameters_init(sunParams, moonParams, 1.0);

    ScatteringResult result = scatteringResult_init();
    const vec3 earthCenter = vec3(0.0);
    if (setupRayEnd(atmosphere, params, rayDir)) {
        result = raymarchSky(atmosphere, params, scatteringParams);

        const vec3 GROUND_ALBEDO_BASE = vec3(ivec3(SETTING_ATM_GROUND_ALBEDO_R, SETTING_ATM_GROUND_ALBEDO_G, SETTING_ATM_GROUND_ALBEDO_B)) / 255.0;
        vec3 groundAlbedo = colors2_colorspaces_convert(COLORS2_COLORSPACES_SRGB, COLORS2_WORKING_COLORSPACE, GROUND_ALBEDO_BASE);

        const float HEIGHT_EPS = 0.01;
        float rayEndHeight = length(params.rayEnd);
        if (abs(rayEndHeight - atmosphere.bottom) < HEIGHT_EPS) {
            vec3 upVector = params.rayEnd / rayEndHeight;
            float clampedGroundHeight = max(atmosphere.bottom + HEIGHT_EPS, rayEndHeight);
            {
                float cosLightZenith = dot(upVector, uval_sunDirWorld);
                vec3 tLightToGround = atmospherics_air_lut_sampleTransmittance(atmosphere, cosLightZenith, clampedGroundHeight);
                float tEarth = raySphereIntersectNearest(params.rayEnd, uval_sunDirWorld, earthCenter, atmosphere.bottom - HEIGHT_EPS);
                float earthShadow = float(tEarth < 0.0);
                float NDotL = saturate(dot(upVector, uval_sunDirWorld));
                float groundLightTerm1 = earthShadow * NDotL * RCP_PI;
                vec3 groundLightTerm3 = tLightToGround * result.transmittance * groundAlbedo;
                vec3 groundLighting = groundLightTerm1 * groundLightTerm3;
                result.inScattering += groundLighting * SUN_ILLUMINANCE;
            }
            {
                float cosLightZenith = dot(upVector, uval_moonDirWorld);
                vec3 tLightToGround = atmospherics_air_lut_sampleTransmittance(atmosphere, cosLightZenith, clampedGroundHeight);
                float tEarth = raySphereIntersectNearest(params.rayEnd, uval_moonDirWorld, earthCenter, atmosphere.bottom - HEIGHT_EPS);
                float earthShadow = float(tEarth < 0.0);
                float NDotL = saturate(dot(upVector, uval_moonDirWorld));
                float groundLightTerm1 = earthShadow * NDotL * RCP_PI;
                vec3 groundLightTerm3 = tLightToGround * result.transmittance * groundAlbedo;
                vec3 groundLighting = groundLightTerm1 * groundLightTerm3;
                result.inScattering += groundLighting * MOON_ILLUMINANCE;
            }
        }
    }

    ssbo_ambLUTWorkingBuffer.rayDir[gl_GlobalInvocationID.x] = vec2(phi, theta);
    ssbo_ambLUTWorkingBuffer.inSctr[gl_GlobalInvocationID.x] = result.inScattering;
}
