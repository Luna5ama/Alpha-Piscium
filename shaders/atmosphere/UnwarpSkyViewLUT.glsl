
#include "Common.glsl"

ScatteringResult unwarpSkyView(vec2 screenPos, vec3 viewPos, float viewZ) {
    ScatteringResult result = scatteringResult_init();

    // TODO: Unwrap sky view data from usam_skyViewLUT_scattering & usam_skyViewLUT_transmittance
    result.inScattering = vec3(0.0);
    result.transmittance = vec3(1.0);

    return result;
}
