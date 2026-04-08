#include "/util/Math.glsl"
#include "/util/GBufferData.glsl"
#include "/util/Coords.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba32ui) uniform restrict readonly uimage2D uimg_gbufferVoxySolidData;
layout(rgba32ui) uniform restrict uimage2D uimg_gbufferSolidData1;
layout(r32ui) uniform restrict uimage2D uimg_gbufferSolidData2;
layout(r32f) uniform restrict image2D uimg_gbufferSolidViewZ;

shared uvec4 shared_voxyData[18][18];

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);

    // Load 16x16 tile plus 1-pixel border into shared memory.
    ivec2 groupBase = ivec2(gl_WorkGroupID.xy) * 16 - 1;
    uint localIndex = gl_LocalInvocationIndex;
    for (uint i = localIndex; i < 324u; i += 256u) {
        ivec2 localPos = ivec2(i % 18u, i / 18u);
        ivec2 loadPos = clamp(groupBase + localPos, ivec2(0), uval_mainImageSizeI - 1);
        shared_voxyData[localPos.x][localPos.y] = imageLoad(uimg_gbufferVoxySolidData, loadPos);
    }

    barrier();

    if (any(greaterThanEqual(texelPos, uval_mainImageSizeI))) return;

    ivec2 localPos = ivec2(gl_LocalInvocationID.xy) + 1;
    uvec4 voxyCenter = shared_voxyData[localPos.x][localPos.y];
    if (voxyCenter.w == 0u && voxyCenter.x == 0u) return;

    float voxyZ = uintBitsToFloat(voxyCenter.w);
    float solidZ = imageLoad(uimg_gbufferSolidViewZ, texelPos).r;

    if (voxyZ > solidZ) {
        imageStore(uimg_gbufferSolidViewZ, texelPos, vec4(voxyZ));

        vec2 uvCenter = unpackUnorm2x16(voxyCenter.x);
        vec4 colorFace = unpackUnorm4x8(voxyCenter.y);
        vec3 albedo = colorFace.rgb;
        uint face = bitfieldExtract(voxyCenter.y, 24, 3);

        uint rawProp = voxyCenter.z;
        uint matID = (rawProp >> 16) & 0xFFFFu;
        vec2 lmCoord = vec2(rawProp & 0xFFu, (rawProp >> 8) & 0xFFu) / 255.0;

        vec2 screenPos = (vec2(texelPos) + 0.5) * uval_mainImageSizeRcp;
        vec3 viewPosCenter = coords_toViewCoord(screenPos, voxyZ, global_camProjInverse);

        vec2 dUVdx = vec2(0.0);
        vec2 dUVdy = vec2(0.0);
        vec3 dPdx = vec3(0.0);
        vec3 dPdy = vec3(0.0);
        float weightX = 0.0;
        float weightY = 0.0;

        for (int dx = -1; dx <= 1; dx += 2) {
            uvec4 voxyNeighbor = shared_voxyData[localPos.x + dx][localPos.y];
            if (voxyNeighbor.w != 0u || voxyNeighbor.x != 0u) {
                uint neighborFace = bitfieldExtract(voxyNeighbor.y, 24, 3);
                if (neighborFace == face) {
                    float neighborZ = uintBitsToFloat(voxyNeighbor.w);
                    if (abs(neighborZ - voxyZ) < 0.2) {
                        vec2 uvNeighbor = unpackUnorm2x16(voxyNeighbor.x);
                        vec2 neighborScreenPos = (vec2(texelPos + ivec2(dx, 0)) + 0.5) * uval_mainImageSizeRcp;
                        vec3 viewPosNeighbor = coords_toViewCoord(neighborScreenPos, neighborZ, global_camProjInverse);

                        dUVdx += (uvNeighbor - uvCenter) * float(dx);
                        dPdx += (viewPosNeighbor - viewPosCenter) * float(dx);
                        weightX += 1.0;
                    }
                }
            }
        }

        for (int dy = -1; dy <= 1; dy += 2) {
            uvec4 voxyNeighbor = shared_voxyData[localPos.x][localPos.y + dy];
            if (voxyNeighbor.w != 0u || voxyNeighbor.x != 0u) {
                uint neighborFace = bitfieldExtract(voxyNeighbor.y, 24, 3);
                if (neighborFace == face) {
                    float neighborZ = uintBitsToFloat(voxyNeighbor.w);
                    if (abs(neighborZ - voxyZ) < 0.2) {
                        vec2 uvNeighbor = unpackUnorm2x16(voxyNeighbor.x);
                        vec2 neighborScreenPos = (vec2(texelPos + ivec2(0, dy)) + 0.5) * uval_mainImageSizeRcp;
                        vec3 viewPosNeighbor = coords_toViewCoord(neighborScreenPos, neighborZ, global_camProjInverse);

                        dUVdy += (uvNeighbor - uvCenter) * float(dy);
                        dPdy += (viewPosNeighbor - viewPosCenter) * float(dy);
                        weightY += 1.0;
                    }
                }
            }
        }

        if (weightX > 0.5) {
            dUVdx /= weightX;
            dPdx /= weightX;
        }
        if (weightY > 0.5) {
            dUVdy /= weightY;
            dPdy /= weightY;
        }

        vec3 geomWorldNormal = normalize(vec3(uint((face >> 1) == 2), uint((face >> 1) == 0), uint((face >> 1) == 1)) * (float(int(face) & 1) * 2 - 1));
        vec3 geomViewNormal = coords_dir_worldToView(geomWorldNormal);

        vec3 tangent = vec3(1.0, 0.0, 0.0);
        vec3 bitangent = vec3(0.0, 1.0, 0.0);
        bool singlePixel = (weightX < 0.5 && weightY < 0.5);

        if (!singlePixel) {
            float det = dUVdx.x * dUVdy.y - dUVdy.x * dUVdy.y;
            if (abs(det) > 1e-6) {
                float invDet = 1.0 / det;
                tangent = (dPdx * dUVdy.y - dPdy * dUVdx.y) * invDet;
                bitangent = (dPdy * dUVdx.x - dPdx * dUVdy.x) * invDet;
                tangent = normalize(tangent - geomViewNormal * dot(geomViewNormal, tangent));
                bitangent = normalize(cross(geomViewNormal, tangent) * sign(det));
                bitangent = normalize(bitangent - geomViewNormal * dot(geomViewNormal, bitangent));
                bitangent = normalize(bitangent - tangent * dot(tangent, bitangent));
            } else {
                singlePixel = true;
            }
        }

        if (singlePixel) {
            vec3 worldTangent;
            if ((face >> 1) == 0u) worldTangent = vec3(1, 0, 0);
            else if ((face >> 1) == 1u) worldTangent = vec3(1, 0, 0);
            else worldTangent = vec3(0, 0, 1);

            tangent = coords_dir_worldToView(worldTangent);
            bitangent = normalize(cross(tangent, geomViewNormal));
            tangent = normalize(cross(geomViewNormal, bitangent));

            dUVdx = vec2(1.0 / 256.0, 0.0);
            dUVdy = vec2(0.0, 1.0 / 256.0);
        }

        GBufferData gData = gbufferData_init();
        gData.geomNormal = geomViewNormal;
        gData.geomTangent = tangent;
        gData.pbrSpecular = vec4(0.0);
        gData.normal = geomViewNormal;
        gData.lmCoord = lmCoord;
        gData.materialID = matID;

        gData.albedo = albedo;
        gData.isHand = false;
        gData.forceBuiltInPBR = true;
        gData.bitangentSign = (dot(cross(tangent, geomViewNormal), bitangent) > 0.0) ? 1 : -1;

        uvec4 d1;
        uvec4 d2;
        gbufferData1_pack(d1, gData);
        gbufferData2_pack(d2, gData);
        imageStore(uimg_gbufferSolidData1, texelPos, d1);
        imageStore(uimg_gbufferSolidData2, texelPos, d2);
    }
}
