#ifndef INCLUDE_atmospherics_Common_glsl
#define INCLUDE_atmospherics_Common_glsl a

struct ScatteringResult {
    vec3 transmittance;
    vec3 inScattering;
};

ScatteringResult scatteringResult_init() {
    ScatteringResult result;
    result.transmittance = vec3(1.0);
    result.inScattering = vec3(0.0);
    return result;
}

vec3 scatteringResult_apply(ScatteringResult result, vec3 color) {
    return result.transmittance * color + result.inScattering;
}

ScatteringResult scatteringResult_blendLayer(ScatteringResult result, ScatteringResult layer, bool aboveLayer) {
    ScatteringResult newResult;
    if (aboveLayer) {
        newResult.inScattering = result.inScattering * layer.transmittance + layer.inScattering;
        newResult.transmittance = result.transmittance * layer.transmittance;
    } else {
        newResult.inScattering = layer.inScattering * result.transmittance + result.inScattering;
        newResult.transmittance = layer.transmittance * result.transmittance;
    }
    return newResult;
}

#endif
