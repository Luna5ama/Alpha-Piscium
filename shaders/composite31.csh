#version 460 compatibility

#include "/util/Coords.glsl"
#include "/util/Lighting.glsl"
#include "/util/Rand.glsl"
#include "/util/Material.glsl"

layout(local_size_x = 64) in;

#include "/atmosphere/RaymarchScreenViewAtmosphere.glsl"

layout(rgba16f) restrict uniform image2D uimg_main;

void main() {
    if (gl_GlobalInvocationID.x < global_dispatchSize1.w) {
        uint texelPosEncoded = indirectComputeData[gl_GlobalInvocationID.x];
        ivec2 texelPos = ivec2(unpackUInt2x16(texelPosEncoded));
        float viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;
        float noiseV = rand_stbnVec1(texelPos, frameCounter + 1);

        ScatteringResult sctrResult = raymarchScreenViewAtmosphere(texelPos, viewZ, noiseV);

        vec4 outputColor = imageLoad(uimg_main, texelPos);
        outputColor.rgb *= sctrResult.transmittance;
        outputColor.rgb += sctrResult.inScattering;
        imageStore(uimg_main, texelPos, outputColor);
    }
}