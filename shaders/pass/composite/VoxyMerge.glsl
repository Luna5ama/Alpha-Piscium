#include "/util/Math.glsl"
#include "/util/GBufferData.glsl"
#include "/util/Coords.glsl"
#include "/util/MaterialIDConst.glsl"
#include "/techniques/textile/CSR32F.glsl"
#include "/techniques/WaterWave.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba32ui) uniform restrict uimage2D uimg_gbufferSolidData1;
layout(r32ui) uniform restrict uimage2D uimg_gbufferSolidData2;
layout(r32f) uniform restrict image2D uimg_gbufferSolidViewZ;
layout(rgba32ui) uniform restrict uimage2D uimg_gbufferTranslucentData1;
layout(r32ui) uniform restrict uimage2D uimg_gbufferTranslucentData2;
layout(rgba16f) uniform restrict image2D uimg_translucentColor;
layout(r32i) uniform restrict iimage2D uimg_csr32f;

shared uvec4 shared_voxyData[18][18];

vec3 voxy_faceWorldNormal(uint face) {
    return normalize(vec3(
        uint((face >> 1u) == 2u),
        uint((face >> 1u) == 0u),
        uint((face >> 1u) == 1u)
    ) * (float(int(face) & 1) * 2.0 - 1.0));
}

vec3 voxy_faceWorldTangent(uint face) {
    return ((face >> 1u) == 2u) ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
}

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 groupBase = ivec2(gl_WorkGroupID.xy) * 16 - 1;
    uint localIndex = gl_LocalInvocationIndex;
    for (uint i = localIndex; i < 324u; i += 256u) {
        ivec2 localPos = ivec2(i % 18u, i / 18u);
        ivec2 loadPos = clamp(groupBase + localPos, ivec2(0), uval_mainImageSizeI - 1);
        shared_voxyData[localPos.x][localPos.y] = texelFetch(usam_gbufferVoxySolidData, loadPos, 0);
    }

    barrier();

    if (any(greaterThanEqual(texelPos, uval_mainImageSizeI))) return;

    ivec2 localPos = ivec2(gl_LocalInvocationID.xy) + 1;
    uvec4 voxyCenter = shared_voxyData[localPos.x][localPos.y];
    bool hasSolid = (voxyCenter.w != 0u || voxyCenter.x != 0u);

    uvec4 voxyTransCenter = texelFetch(usam_gbufferVoxyTranslucentData, texelPos, 0);
    bool hasTranslucent = (voxyTransCenter.w != 0u || voxyTransCenter.x != 0u);

    if (!hasSolid && !hasTranslucent) return;

    float solidZ = imageLoad(uimg_gbufferSolidViewZ, texelPos).r;

    if (hasSolid) {
        float voxyZ = uintBitsToFloat(voxyCenter.w);
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

            vec3 geomWorldNormal = voxy_faceWorldNormal(face);
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
                vec3 worldTangent = voxy_faceWorldTangent(face);
                tangent = coords_dir_worldToView(worldTangent);
                bitangent = normalize(cross(tangent, geomViewNormal));
                tangent = normalize(cross(geomViewNormal, bitangent));
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

    if (hasTranslucent) {
        float voxyZ = uintBitsToFloat(voxyTransCenter.w);
        solidZ = imageLoad(uimg_gbufferSolidViewZ, texelPos).r;
        if (voxyZ > solidZ) {
            uint face = bitfieldExtract(voxyTransCenter.y, 24, 3);
            uint rawProp = voxyTransCenter.z;
            uint matID = (rawProp >> 16) & 0xFFFFu;
            bool isWater = matID == MATERIAL_ID_WATER;
            vec2 lmCoord = vec2(rawProp & 0xFFu, (rawProp >> 8) & 0xFFu) / 255.0;

            vec3 geomViewNormal = coords_dir_worldToView(voxy_faceWorldNormal(face));
            vec3 geomViewTangent = coords_dir_worldToView(voxy_faceWorldTangent(face));
            vec3 geomViewBitangent = normalize(cross(geomViewTangent, geomViewNormal));
            geomViewTangent = normalize(cross(geomViewNormal, geomViewBitangent));

            vec2 screenPos = (vec2(texelPos) + 0.5) * uval_mainImageSizeRcp;
            vec3 viewPos = coords_toViewCoord(screenPos, voxyZ, global_camProjInverse);
            float zOffset = 0.0;
            vec3 finalNormal = geomViewNormal;

            #if defined(SETTING_NORMAL_MAPPING)
            if (isWater) {
                vec3 scenePos = (gbufferModelViewInverse * vec4(viewPos, 1.0)).xyz;
                vec3 cameraPosWaveSpace = vec3(cameraPositionInt >> 5) + ldexp(vec3(cameraPositionInt & ivec3(31)), ivec3(-5));
                cameraPosWaveSpace = cameraPositionFract * WAVE_POS_BASE + cameraPosWaveSpace * 0.736;
                vec3 waveWorldPos = scenePos * WAVE_POS_BASE + cameraPosWaveSpace;

                #ifdef SETTING_WATER_PARALLAX
                const float UP_DIR_COS_EPSILON = 0.001;
                if (dot(geomViewNormal, uval_upDirView) > UP_DIR_COS_EPSILON) {
                    const uint PARALLAX_LINEAR_STEPS = uint(SETTING_WATER_PARALLAX_LINEAR_STEPS);
                    const float PARALLAX_STRENGTH = float(SETTING_WATER_PARALLAX_STRENGTH) / 0.83;

                    vec3 rayVector = scenePos / max(abs(scenePos.y), 1e-4);
                    float rayVectorLength = length(rayVector);
                    vec3 rayDir = rayVector / max(rayVectorLength, 1e-4);
                    const float MAX_WAVE_HEIGHT = -rcp(0.83);
                    vec2 prevXY = vec2(0.0, 1.0);

                    for (uint i = 0u; i < PARALLAX_LINEAR_STEPS; i++) {
                        float fi = float(i) / float(max(PARALLAX_LINEAR_STEPS - 1u, 1u));
                        fi = 1.0 - pow3(1.0 - fi);
                        fi *= rayVectorLength;
                        vec3 sampleDelta = rayDir * fi;

                        vec3 samplePos = waveWorldPos + sampleDelta * WAVE_POS_BASE * PARALLAX_STRENGTH;
                        samplePos.y = waveWorldPos.y;
                        float sampleHeight = saturate(waveHeight(samplePos, false) * MAX_WAVE_HEIGHT);
                        float currHeight = 1.0 + sampleDelta.y;

                        if (currHeight <= sampleHeight) {
                            vec2 xy1 = prevXY;
                            vec2 xy2 = vec2(fi, currHeight - sampleHeight);
                            for (uint j = 0u; j < 2u; j++) {
                                float x3 = xy2.x - xy2.y * (xy2.x - xy1.x) / (xy2.y - xy1.y);
                                fi = x3;
                                sampleDelta = rayDir * x3;

                                samplePos = waveWorldPos;
                                samplePos.xz += sampleDelta.xz * (WAVE_POS_BASE * PARALLAX_STRENGTH);
                                sampleHeight = saturate(waveHeight(samplePos, false) * MAX_WAVE_HEIGHT);
                                float depthDiff = (1.0 + sampleDelta.y) - sampleHeight;
                                if (depthDiff < 0.01) break;
                                xy1 = xy2;
                                xy2 = vec2(x3, depthDiff);
                            }

                            waveWorldPos = samplePos;
                            zOffset = fi * PARALLAX_STRENGTH * 0.5 + 0.4 / rayDir.y;
                            break;
                        }

                        prevXY = vec2(fi, currHeight - sampleHeight);
                    }
                }
                #endif

                const float NORMAL_EPS = 0.2;
                const float NORMAL_WEIGHT = SETTING_WATER_NORMAL_SCALE;
                float waveHeightC = waveHeight(waveWorldPos, true);
                float waveOffset = NORMAL_EPS * WAVE_POS_BASE;
                vec3 offsetTangentX = coords_dir_viewToWorld(geomViewTangent) * waveOffset;
                vec3 offsetTangentY = coords_dir_viewToWorld(geomViewBitangent) * waveOffset;
                float waveHeightX = waveHeight(waveWorldPos + offsetTangentX, true);
                float waveHeightY = waveHeight(waveWorldPos + offsetTangentY, true);

                vec3 tangentNormal = vec3(waveHeightX, waveHeightY, NORMAL_EPS);
                tangentNormal.xy -= waveHeightC;
                tangentNormal.xy *= NORMAL_WEIGHT;
                finalNormal = normalize(mat3(geomViewTangent, geomViewBitangent, geomViewNormal) * normalize(tangentNormal));
            }
            #endif

            float offsetViewZ = voxyZ - clamp(zOffset, -16.0, 16.0);
            ivec2 nearDepthTexelPos = isWater ? csr32f_tile1_texelToTexel(texelPos) : csr32f_tile3_texelToTexel(texelPos);
            ivec2 farDepthTexelPos = isWater ? csr32f_tile2_texelToTexel(texelPos) : csr32f_tile4_texelToTexel(texelPos);
            float waterNearViewZ = -texelFetch(usam_csr32f, csr32f_tile1_texelToTexel(texelPos), 0).r;
            float translucentNearViewZ = -texelFetch(usam_csr32f, csr32f_tile3_texelToTexel(texelPos), 0).r;
            float frontBefore = max(waterNearViewZ, translucentNearViewZ);

            imageAtomicMin(uimg_csr32f, nearDepthTexelPos, floatBitsToInt(-offsetViewZ));
            imageAtomicMax(uimg_csr32f, farDepthTexelPos, floatBitsToInt(-offsetViewZ));

            if (offsetViewZ >= frontBefore) {
                vec4 transmittanceV = texelFetch(usam_voxyTranslucentColor, texelPos, 0);

                GBufferData gData = gbufferData_init();
                gData.geomNormal = geomViewNormal;
                gData.geomTangent = geomViewTangent;
                gData.pbrSpecular = vec4(0.0);
                gData.normal = finalNormal;
                gData.lmCoord = lmCoord;
                gData.materialID = matID;
                gData.albedo = transmittanceV.rgb;
                gData.isHand = false;
                gData.forceBuiltInPBR = true;
                gData.bitangentSign = (dot(cross(geomViewTangent, geomViewNormal), geomViewBitangent) > 0.0) ? 1 : -1;

                uvec4 d1;
                uvec4 d2;
                gbufferData1_pack(d1, gData);
                gbufferData2_pack(d2, gData);
                imageStore(uimg_gbufferTranslucentData1, texelPos, d1);
                imageStore(uimg_gbufferTranslucentData2, texelPos, d2);
                imageStore(uimg_translucentColor, texelPos, transmittanceV);
            }
        }
    }
}
