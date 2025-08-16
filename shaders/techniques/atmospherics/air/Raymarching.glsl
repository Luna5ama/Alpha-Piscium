#include "Common.glsl"
#include "RaymarchingBase.glsl"

#include "lut/API.glsl"

vec3 raymarchTransmittance(
    AtmosphereParameters atmosphere,
    RaymarchParameters params,
    float stepJitter
);

MultiScatteringResult raymarchMultiScattering(
    AtmosphereParameters atmosphere,
    RaymarchParameters params,
    LightParameters lightParams,
    float stepJitter
);

ScatteringResult raymarchSkySingle(
    AtmosphereParameters atmosphere,
    RaymarchParameters params,
    LightParameters lightParams,
    float bottomOffset
);

ScatteringResult raymarchSky(
    AtmosphereParameters atmosphere,
    RaymarchParameters params,
    ScatteringParameters scatteringParams
);

ScatteringResult raymarchAerialPerspective(
    AtmosphereParameters atmosphere,
    RaymarchParameters params,
    ScatteringParameters scatteringParams,
    vec3 shadowStart,
    vec3 shadowEnd,
    float stepJitter
);

#ifdef ATMOSPHERE_RAYMARCHING_TRANSMITTANCE
#define ATMOSPHERE_RAYMARCHING_FUNC_TYPE 0
#include "RaymarchingFunc.glsl"
#undef ATMOSPHERE_RAYMARCHING_FUNC_TYPE
#endif

#ifdef ATMOSPHERE_RAYMARCHING_MULTI_SCTR
#define ATMOSPHERE_RAYMARCHING_FUNC_TYPE 1
#include "RaymarchingFunc.glsl"
#undef ATMOSPHERE_RAYMARCHING_FUNC_TYPE
#endif

#ifdef ATMOSPHERE_RAYMARCHING_SKY_SINGLE
#define ATMOSPHERE_RAYMARCHING_FUNC_TYPE 2
#include "RaymarchingFunc.glsl"
#undef ATMOSPHERE_RAYMARCHING_FUNC_TYPE
#endif

#ifdef ATMOSPHERE_RAYMARCHING_SKY
#define ATMOSPHERE_RAYMARCHING_FUNC_TYPE 3
#include "RaymarchingFunc.glsl"
#undef ATMOSPHERE_RAYMARCHING_FUNC_TYPE
#endif

#ifdef ATMOSPHERE_RAYMARCHING_AERIAL_PERSPECTIVE
#define ATMOSPHERE_RAYMARCHING_FUNC_TYPE 4
#include "RaymarchingFunc.glsl"
#undef ATMOSPHERE_RAYMARCHING_FUNC_TYPE
#endif