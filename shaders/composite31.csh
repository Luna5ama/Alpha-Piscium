#version 460 compatibility

#include "/atmosphere/Scattering.glsl"
#include "/util/Coords.glsl"
#include "/util/Rand.glsl"
#include "/util/Material.glsl"

layout(local_size_x = 64) in;

uniform sampler2D usam_gbufferViewZ;

layout(rgba16f) restrict uniform image2D uimg_main;

void main() {
    if (gl_GlobalInvocationID.x < global_dispatchSize1.w) {
        uint texelPosEncoded = indirectComputeData[gl_GlobalInvocationID.x];
        ivec2 texelPos = ivec2(unpackUInt2x16(texelPosEncoded));

        float viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;
        vec2 screenPos = (vec2(texelPos) + 0.5) * global_mainImageSizeRcp;
        vec3 viewPos = coords_toViewCoord(screenPos, viewZ, gbufferProjectionInverse);

        float ignValue = rand_IGN(texelPos, frameCounter);
        AtmosphereParameters atmosphere = getAtmosphereParameters();
        ScatteringResult sctrResult = computeSingleScattering(atmosphere, vec3(0.0), viewPos, ignValue);

        vec4 outputColor = imageLoad(uimg_main, texelPos);
        outputColor.rgb *= sctrResult.transmittance;
        outputColor.rgb += sctrResult.inScattering;
        imageStore(uimg_main, texelPos, outputColor);
    }
}