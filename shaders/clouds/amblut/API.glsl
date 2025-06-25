#ifndef INCLUDE_clouds_amblut_API_glsl
#define INCLUDE_clouds_amblut_API_glsl a

#include "/util/Coords.glsl"
#include "/util/Math.glsl"

uniform sampler3D usam_cloudsAmbLUT;

#define CLOUDS_AMBLUT_LAYER_STRAUS 0.5
#define CLOUDS_AMBLUT_LAYER_CUMULUS 1.5
#define CLOUDS_AMBLUT_LAYER_ALTOCUMULUS 2.5
#define CLOUDS_AMBLUT_LAYER_CIRRUS 3.5
#define CLOUDS_AMBLUT_LAYER_CIRROSTRATUS 4.5
#define CLOUDS_AMBLUT_LAYER_NOCTILUCENT 5.5

#define _CLOUDS_AMBLUT_SIZE 16.0
#define _CLOUDS_AMBLUT_SIZE_RCP rcp(_CLOUDS_AMBLUT_SIZE)
#define _CLOUDS_AMBLUT_SIZE_LAYERS 6.0
#define _CLOUDS_AMBLUT_SIZE_LAYERS_RCP rcp(_CLOUDS_AMBLUT_SIZE_LAYERS)

vec2 cloods_amblut_uv(vec3 viewDir, vec2 jitter) {
    vec2 uv = coords_equirectanglarForwardHorizonBoost(viewDir);
    uv += (jitter - 0.5) * _CLOUDS_AMBLUT_SIZE_RCP;
    uv = fract(uv);
    return uv;
}

vec3 clouds_amblut_sample(vec2 uv, float layer) {
    return texture(usam_cloudsAmbLUT, vec3(uv, layer * _CLOUDS_AMBLUT_SIZE_LAYERS_RCP)).rgb;
}

#endif