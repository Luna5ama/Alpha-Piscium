#include "/util/BitPacking.glsl"
#include "/util/Coords.glsl"
#include "/util/Rand.glsl"

layout(local_size_x = 64) in;

#include "/techniques/atmospherics/air/RaymarchScreenViewAtmosphere.glsl"

layout(rgba16f) restrict uniform image2D uimg_main;

void main() {
    if (gl_GlobalInvocationID.x < global_dispatchSize1.w) {
        uint texelPosEncoded = indirectComputeData[gl_GlobalInvocationID.x];
        ivec2 texelPos = ivec2(unpackUInt2x16(texelPosEncoded));
        float viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;
        float noiseV = rand_stbnVec1(texelPos, frameCounter + 1);

        AtmosphereParameters atmosphere = getAtmosphereParameters();
        ScatteringResult sctrResult = raymarchScreenViewAtmosphere(
            texelPos,
            0.0,
            viewZ, 
            SETTING_LIGHT_SHAFT_DEPTH_BREAK_CORRECTION_SAMPLES, 
            noiseV
        );

        vec4 outputColor = imageLoad(uimg_main, texelPos);
        outputColor.rgb *= sctrResult.transmittance;
        outputColor.rgb += sctrResult.inScattering;
        imageStore(uimg_main, texelPos, outputColor);
    }
}