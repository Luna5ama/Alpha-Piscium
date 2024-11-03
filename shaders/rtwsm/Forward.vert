#version 460 core

#include "../utils/Common.glsl"
#include "../GlobalCamera.glsl"
#include "ShadowParams.glsl"
//#include "ShadowSamplers.glsl"

uniform vec3 originOffset;
uniform float uBaseImportance;

#include "../TerrainVertAttrib.glsl"

out float fImportance;

const vec3 coordConvert = vec3(2.51773861295491E-4);

void main() {
    // gl_BaseInstance exploit
    vec3 worldCoord = fma(vPos, coordConvert, vChunkOffset * 16.0 + originOffset);
    float aoMultiplier = float((vMdlAttrib >> 2u) & 1u);
    worldCoord += (1.0 - aoMultiplier) * uShadowMap.lightDir * 0.05;

    gl_Position = uShadowMap.matrix * vec4(worldCoord, 1.0);
    gl_Position /= gl_Position.w;

    float pDist = max(length(worldCoord), 4.0) / 4.0;
    vec3 viewDir = normalize(worldCoord);

    float importance = uBaseImportance;

    // Distance to Eye Function
    importance *= 1.0 / pDist;

    // Surface Normal Function
    const float SN_BETA = 0.5;
    importance *= 1.0 + SN_BETA * max(dot(viewNormal, -viewDir), 0.0);

    const float LSN_BETA = 0.1;
    importance *= 1.0 + LSN_BETA * sqrt(pDist) * pow(1.0 - abs(dot(viewNormal, uShadowMap.lightDir)), 8.0);

    fImportance = importance;
}