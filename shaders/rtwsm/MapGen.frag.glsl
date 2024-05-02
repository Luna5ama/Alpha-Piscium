#version 460 core

#include "../GlobalCamera.glsl"
#include "../Utils/Rand.glsl"
#include "../Utils/R2Seqs.glsl"

#ifdef CUTOUT
uniform sampler2D usam_blockAlbedoAtlas;
uniform sampler2D usam_blockSpecularAtlas;

//struct VertData {
//    vec2 texCoord;
//    float lodMul;
//};
//in VertData fragData;
in vec3 texCoordLodMul;
in vec3 fWorldCoord;

void main() {
//    float sampleLod = textureQueryLod(usam_blockAlbedoAtlas, fragData.texCoord).y * fragData.lodMul;
//    float alpha = textureLod(usam_blockAlbedoAtlas, fragData.texCoord, sampleLod).a;
    const float sssValueMin = 65.0 / 255.0;
    float sssStrength = step(sssValueMin, textureLod(usam_blockSpecularAtlas, texCoordLodMul.xy, 0.0).b);
    float sampleLod = textureQueryLod(usam_blockAlbedoAtlas, texCoordLodMul.xy).y * texCoordLodMul.z;
    float alpha = textureLod(usam_blockAlbedoAtlas, texCoordLodMul.xy, sampleLod).a;
    uint coordRand = uint(rand(fWorldCoord) * 2048.0);

    alpha *= (1.0 - sssStrength) + r2Seq1(coordRand + ubo_cameraNoJitter.gFrameIndex) * sssStrength;
    if (alpha <= 0.5) discard;
}
#else
layout(early_fragment_tests) in;

void main() {}
#endif