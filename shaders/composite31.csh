#version 460 compatibility

#include "/util/Coords.glsl"
#include "/util/Lighting.glsl"
#include "/util/Rand.glsl"
#include "/util/Material.glsl"

layout(local_size_x = 64) in;

uniform sampler2D usam_rtwsm_imap;
const bool shadowHardwareFiltering0 = true;
uniform sampler2DShadow shadowtex0HW;
uniform sampler2D usam_gbufferViewZ;
uniform usampler2D usam_packedZN;

#include "/atmosphere/Scattering.glsl"

layout(rgba16f) restrict uniform image2D uimg_main;

void main() {
    if (gl_GlobalInvocationID.x < global_dispatchSize1.w) {
        uint texelPosEncoded = indirectComputeData[gl_GlobalInvocationID.x];
        ivec2 texelPos = ivec2(unpackUInt2x16(texelPosEncoded));
        vec2 screenPos = (vec2(texelPos) + 0.5) * global_mainImageSizeRcp;
        float viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;
        float noiseV = rand_stbnVec1(texelPos, frameCounter + 1);

        ScatteringResult sctrResult = computeSingleScattering(screenPos, viewZ, noiseV);

        vec4 outputColor = imageLoad(uimg_main, texelPos);
        outputColor.rgb *= sctrResult.transmittance;
        outputColor.rgb += sctrResult.inScattering;
        imageStore(uimg_main, texelPos, outputColor);
    }
}