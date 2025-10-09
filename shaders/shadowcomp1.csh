#version 460 compatibility

#include "/util/Coords.glsl"
#include "/util/Rand.glsl"
#include "/techniques/WaterWave.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const ivec3 workGroups = ivec3(128, 128, 1);

layout(rgb10_a2) uniform image2D uimg_shadow_waterNormal;

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    // Iris bug, must have some reference to the sampler for shadow image to work
    if (all(lessThan(texelPos, textureSize(usam_shadow_waterNormal, 0)))) {
        float waterMask = texelFetch(usam_shadow_waterMask, texelPos, 0).r;
        if (waterMask > 0.5) {
            vec2 screenPos = texelFetch(usam_shadow_unwarpedUV, texelPos, 0).rg;

            float shadowDepth = texelFetch(shadowtex0, texelPos, 0).r;
            vec3 shadowScreenPos = vec3(screenPos, shadowDepth);

            vec3 shadowNDCPos = shadowScreenPos * 2.0 - 1.0;
            vec4 shadowViewPos = global_shadowProjInversePrev * vec4(shadowNDCPos, 1.0);
            shadowViewPos /= shadowViewPos.w;
            vec4 scenePos = global_shadowViewInverse * global_shadowRotationMatrixInverse * shadowViewPos;

            vec3 cameraPosWaveSpace = vec3(cameraPositionInt >> 5) + ldexp(vec3(cameraPositionInt & ivec3(31)), ivec3(-5));
            cameraPosWaveSpace = cameraPositionFract * WAVE_POS_BASE + cameraPosWaveSpace * 0.736;
            vec3 waveWorldPos = scenePos.xyz * WAVE_POS_BASE + cameraPosWaveSpace;

            const float NORMAL_EPS = 0.05;
            const float NORMAL_WEIGHT = SETTING_WATER_NORMAL_SCALE;
            float waveHeightC = waveHeight(waveWorldPos, true, true);
            float waveHeightX = waveHeight(waveWorldPos + vec3(NORMAL_EPS * WAVE_POS_BASE, 0.0, 0.0), true, true);
            float waveHeightZ = waveHeight(waveWorldPos + vec3(0.0, 0.0, NORMAL_EPS * WAVE_POS_BASE), true, true);
            vec3 waveNormal = vec3(
                waveHeightX,
                NORMAL_EPS,
                waveHeightZ
            );
            waveNormal.xz -= waveHeightC;
            waveNormal.xz *= NORMAL_WEIGHT;
            vec3 waterNormal = normalize(waveNormal);

            imageStore(uimg_shadow_waterNormal, texelPos, vec4(waterNormal * 0.5 + 0.5, 1.0));
        }
    }
}