#ifndef INCLUDE_clouds_ScreenRay_glsl
#define INCLUDE_clouds_ScreenRay_glsl

#include "CumulusRay.glsl"
#include "ss/Common.glsl"

ScatteringResult clouds_screen_front(ivec2 texel, float surfaceZ, float solidZ) {
    if (surfaceZ > max(-65536.0, solidZ)) {
        vec2 uv = coords_texelToUV(texel, uval_mainImageSizeRcp);
        vec3 view = coords_toViewCoord(uv, surfaceZ, global_camProjInverse);
        vec3 direction = normalize(mat3(gbufferModelViewInverse) * view);
        AtmosphereParameters atmosphere = getAtmosphereParameters();
        vec4 jitter = vec4(rand_stbnVec2(rand_newStbnPos(texel, 1), frameCounter),
            rand_stbnVec2(rand_newStbnPos(texel, 2), frameCounter));
        float meanDistance;
        return clouds_cu_render(atmosphere, clouds_cu_origin(gbufferModelViewInverse[3].xyz),
            atmosphere_viewToAtm(atmosphere, vec3(0.0)), direction, vec2(0.0, length(view) * 0.001),
            jitter, false, meanDistance);
    }
    CloudSSHistoryData cloud;
    clouds_ss_historyData_unpack(history_lowCloud_fetch(texel), cloud);
    return ScatteringResult(cloud.transmittance, cloud.inScattering);
}

// The background already contains C + T*B. This applies local air to B and keeps the cloud's own air prefix.
void clouds_screen_localAir(ScatteringResult cloud, inout ScatteringResult air) {
    air.inScattering = cloud.transmittance * air.inScattering + (1.0 - air.transmittance) * cloud.inScattering;
}

#endif
