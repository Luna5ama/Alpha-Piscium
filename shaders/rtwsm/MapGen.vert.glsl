#version 460 core

uniform sampler1DArray usam_warpingMap;

#include "../GlobalCamera.glsl"
#include "ShadowParams.glsl"

uniform vec3 originOffset;

#include "../TerrainVertAttrib.glsl"

#ifdef CUTOUT
//struct VertData {
//    vec2 texCoord;
//    float lodMul;
//    vec3 texCoordLodMul;
//};
//
//out VertData fragData;
out vec3 texCoordLodMul;
out vec3 fWorldCoord;
#endif

const vec3 coordConvert = vec3(2.51773861295491E-4);

void main() {
    float frontFacing = max(0.0, dot(uShadowMap.lightDir, vNormal));

    // gl_BaseInstance exploit
    vec3 worldCoord = fma(vPos, coordConvert, vChunkOffset * 16.0 + originOffset);

    #ifdef CUTOUT
    fWorldCoord = worldCoord;
    //    fragData.texCoord = vTexCoord;
    //    fragData.lodMul = float(vMdlAttrib & 1u);
    texCoordLodMul.xy = vTexCoord;
    texCoordLodMul.z = float(vMdlAttrib & 1u);
    #endif


    float aoMultiplier = 1.0 - float((vMdlAttrib >> 2u) & 1u);
//    worldCoord -= aoMultiplier * (vNormal * 0.005);
    worldCoord -= (1.0 - frontFacing) * aoMultiplier * (vNormal * 0.01);

//    worldCoord -= (1.0 - frontFacing) * (vNormal * 0.03);
//    worldCoord -= (frontFacing) * (vNormal * 0.01);

    gl_Position = uShadowMap.matrix * vec4(worldCoord, 1.0);
    gl_Position /= gl_Position.w;

    vec2 texelSize;
    vec2 vPosTS = gl_Position.xy * 0.5 + 0.5;
    vPosTS = rtwsm_warpTexCoordTexelSize(usam_warpingMap, vPosTS, texelSize);

    gl_Position.z -= (1.0 - frontFacing) * aoMultiplier * 0.0001;
//    gl_Position.z -= (1.0 - frontFacing) * aoMultiplier * 0.0001;
//    gl_Position.z += frontFacing * aoMultiplier * 0.0002;
//    gl_Position.z += frontFacing * ((1.0 * uShadowMap.boundSizeZ.y) / (4096.0 * (texelSize.x + texelSize.y)));
//    gl_Position.z -= (1.0 - frontFacing) * (0.0001);
//    gl_Position.z += (frontFacing) * (0.001);

    gl_Position.xy = vPosTS * 2.0 - 1.0;
}