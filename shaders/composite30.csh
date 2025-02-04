#version 460 compatibility

#include "atmosphere/UnwrapEpipolar.comp"

layout(local_size_x = 128, local_size_y = 1) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

uniform sampler2D usam_translucentColor;

void main() {
    ivec2 imgSize = imageSize(uimg_main);
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);

    if (all(lessThan(texelPos, imgSize))) {
        vec3 inScattering;
        vec3 transmittance;
        vec2 texCoord = (vec2(texelPos) + 0.5) / vec2(imgSize);
        float viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;
        UnwarpEpipolarInsctrImage(texCoord * 2.0 - 1.0, viewZ, inScattering, transmittance);

        vec4 color = imageLoad(uimg_main, texelPos);

        color.rgb *= transmittance;
        vec3 sunRadiance = global_sunRadiance.rgb * global_sunRadiance.a;
        float shadowIsSun = float(all(equal(sunPosition, shadowLightPosition)));
        sunRadiance *= mix(MOON_RADIANCE_MUL, vec3(1.0), shadowIsSun);
        color.rgb += sunRadiance * inScattering;

        vec4 translucentColorSample = texelFetch(usam_translucentColor, texelPos, 0);
        float luminanceC = colors_srgbLuma(color.rgb) * 4.0;
        float luminanceT = max(colors_srgbLuma(translucentColorSample.rgb), 1.0);
        color.rgb = mix(color.rgb, translucentColorSample.rgb * (luminanceC / luminanceT), translucentColorSample.a);


        imageStore(uimg_main, texelPos, color);
    }
}