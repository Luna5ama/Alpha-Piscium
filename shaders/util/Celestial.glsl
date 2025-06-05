#ifndef INCLUDE_util_CelestialObjects_glsl
#define INCLUDE_util_CelestialObjects_glsl a

#include "BlackBody.glsl"
#include "Colors.glsl"
#include "Coords.glsl"
#include "/atmosphere/Common.glsl"

#ifdef SETTING_REAL_SUN_TEMPERATURE
#define SUN_TEMPERATURE 5772.0
#else
#define SUN_TEMPERATURE SETTING_SUN_TEMPERATURE
#endif

#define SUN_RADIUS (695700.0 * SETTING_SUN_RADIUS) // in km
#define SUN_DISTANCE (149597870.7 * SETTING_SUN_DISTANCE) // in km
#define SUN_ANGULAR_RADIUS atan(SUN_RADIUS / SUN_DISTANCE)
#define SUN_SOLID_ANGLE (2.0 * PI * (1.0 - sqrt(pow2(SUN_DISTANCE) - pow2(SUN_RADIUS)) / SUN_DISTANCE))

#define SUN_LUMINANCE blackBody_evalRadiance(SUN_TEMPERATURE)
#define SUN_ILLUMINANCE (SUN_LUMINANCE * SUN_SOLID_ANGLE)

#define MOON_RADIUS (1737.4 * SETTING_MOON_RADIUS) // in km
#define MOON_DISTANCE (384399 * SETTING_MOON_DISTANCE) // in km
#define MOON_ANGULAR_RADIUS atan(MOON_RADIUS / MOON_DISTANCE)
#define MOON_SOLID_ANGLE (2.0 * PI * (1.0 - sqrt(pow2(MOON_DISTANCE) - pow2(MOON_RADIUS)) / MOON_DISTANCE))

#define MOON_ALBEDO (SETTING_MOON_ALBEDO * vec3(SETTING_MOON_COLOR_R, SETTING_MOON_COLOR_G, SETTING_MOON_COLOR_B))
#define MOON_LUMINANCE (SUN_ILLUMINANCE * MOON_ALBEDO * RCP_PI)
#define MOON_ILLUMINANCE (MOON_LUMINANCE * MOON_SOLID_ANGLE)

uniform sampler2D usam_starmap;

float _celestial_circle(vec3 rayDir, vec3 objDir, float objAngularRadius) {
    float objCosTheta = cos(objAngularRadius);
    float cosTheta = saturate(dot(rayDir, objDir));
    return float(cosTheta >= objCosTheta);
}

const float _CELESTIAL_STARMAP_MILKYWAY_LUMINANCE = 0.1927554;
const float _CELESTIAL_REAL_MILKYWAY_LUMINANCE_MCD = 1.5;
const float _CELESTIAL_REAL_MILKYWAY_LUMINANCE_KCD = _CELESTIAL_REAL_MILKYWAY_LUMINANCE_MCD / 1000000.0;
const float _CELESTIAL_STARMAP_EXP = _CELESTIAL_REAL_MILKYWAY_LUMINANCE_KCD / _CELESTIAL_STARMAP_MILKYWAY_LUMINANCE;


// Rotation matrix to transform from world space to galactic coordinates
// This aligns +Y with galactic north pole and centers galaxy at phi=0
const mat3 WORLD_TO_GALATIC =mat3(
    cos(radians(40.0)), sin(radians(40.0)), 0.0,
    -sin(radians(40.0)), cos(radians(40.0)), 0.0,
    0.0, 0.0, 1.0
) * mat3(
    0.0, 0.0, 1.0,  // First column
    1.0, 0.0, 0.0,  // Second column
    0.0, 1.0, 0.0
);

const vec2 _CELESTIAL_STARMAP_SIZE = vec2(4096.0, 2048.0);

// from https://github.com/GameTechDev/TAA
vec4 BicubicSampling5(sampler2D samplerV, vec2 inHistoryST){
    const vec2 rcpResolution = rcp(_CELESTIAL_STARMAP_SIZE);
    const vec2 fractional = fract(inHistoryST - 0.5);
    const vec2 uv = (floor(inHistoryST - 0.5) + vec2(0.5f, 0.5f)) * rcpResolution;

    // 5-tap bicubic sampling (for Hermite/Carmull-Rom filter) -- (approximate from original 16->9-tap bilinear fetching)
    const vec2 t = vec2(fractional);
    const vec2 t2 = vec2(fractional * fractional);
    const vec2 t3 = vec2(fractional * fractional * fractional);
    const float s = float(0.5);
    const vec2 w0 = -s * t3 + float(2.f) * s * t2 - s * t;
    const vec2 w1 = (float(2.f) - s) * t3 + (s - float(3.f)) * t2 + float(1.f);
    const vec2 w2 = (s - float(2.f)) * t3 + (3 - float(2.f) * s) * t2 + s * t;
    const vec2 w3 = s * t3 - s * t2;
    const vec2 s0 = w1 + w2;
    const vec2 f0 = w2 / (w1 + w2);
    const vec2 m0 = uv + f0 * rcpResolution;
    const vec2 tc0 = uv - 1.f * rcpResolution;
    const vec2 tc3 = uv + 2.f * rcpResolution;

    const vec4 A = vec4(texture(samplerV, vec2(m0.x, tc0.y)));
    const vec4 B = vec4(texture(samplerV, vec2(tc0.x, m0.y)));
    const vec4 C = vec4(texture(samplerV, vec2(m0.x, m0.y)));
    const vec4 D = vec4(texture(samplerV, vec2(tc3.x, m0.y)));
    const vec4 E = vec4(texture(samplerV, vec2(m0.x, tc3.y)));
    const vec4 color = (float(0.5f) * (A + B) * w0.x + A * s0.x + float(0.5f) * (A + B) * w3.x) * w0.y + (B * w0.x + C * s0.x + D * w3.x) * s0.y + (float(0.5f) * (B + E) * w0.x + E * s0.x + float(0.5f) * (D + E) * w3.x) * w3.y;
    return color;
}

vec4 celestial_render(ivec2 texelPos) {
    vec2 screenPos = (vec2(texelPos) + 0.5 - global_taaJitter) * global_mainImageSizeRcp;
    vec3 viewCoord = coords_toViewCoord(screenPos, -far, gbufferProjectionInverse);

    vec3 viewDir = normalize(viewCoord);
    vec3 viewDirWorld = mat3(gbufferModelViewInverse) * viewDir;

    AtmosphereParameters atmosphere = getAtmosphereParameters();
    vec3 origin = atmosphere_viewToAtm(atmosphere, vec3(0.0));
    origin.y = max(origin.y, atmosphere.bottom + 0.5);
    vec3 earthCenter = vec3(0.0);
    float earthIntersect = raySphereIntersectNearest(origin, viewDirWorld, earthCenter, atmosphere.bottom);

    float sunV = _celestial_circle(viewDir, uval_sunDirView, SUN_ANGULAR_RADIUS * 2.0);
    float moonV = _celestial_circle(viewDir, uval_moonDirView, MOON_ANGULAR_RADIUS * 2.0);
    float moonDarkV = _celestial_circle(viewDir, uval_moonDirView, MOON_ANGULAR_RADIUS * 4.0);

    vec4 result = vec4(0.0, 0.0, 0.0, 1.0);
    result += sunV * vec4(SUN_LUMINANCE, 4.0);
    result += moonV * vec4(MOON_LUMINANCE, 0.0);
    result += moonDarkV * vec4(0.0, 0.0, 0.0, -0.99);

    #if SETTING_STARMAP_INTENSITY
    vec2 starmapCoords = coords_equirectanglarForward(normalize(WORLD_TO_GALATIC * viewDirWorld));
    // Adjust longitude direction (increases to the left)
    starmapCoords.x = 1.0 - starmapCoords.x;

    vec3 starmap = colors_LogLuv32ToSRGB(BicubicSampling5(usam_starmap, starmapCoords * _CELESTIAL_STARMAP_SIZE));
    starmap = pow(starmap, vec3(SETTING_STARMAP_GAMMA));
    result.rgb += starmap * _CELESTIAL_STARMAP_EXP * SETTING_STARMAP_INTENSITY;
    #endif

    result.rgb *= step(earthIntersect, 0.0);

    return result;
}

#endif