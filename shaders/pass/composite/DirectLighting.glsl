#extension GL_KHR_shader_subgroup_ballot : enable

#define GLOBAL_DATA_MODIFIER buffer
#include "/techniques/atmospherics/air/Constants.glsl"
#include "/techniques/atmospherics/air/lut/API.glsl"
#include "/techniques/HiZCheck.glsl"
#include "/techniques/gi/Common.glsl"
#include "/techniques/Lighting.glsl"
#include "/util/Celestial.glsl"
#include "/util/Colors.glsl"
#include "/util/BitPacking.glsl"
#include "/util/NZPacking.glsl"
#include "/util/Morton.glsl"
#include "/util/ThreadGroupTiling.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform writeonly image2D uimg_main;
layout(rgba8) uniform restrict image2D uimg_overlays;
layout(rgba16f) uniform restrict image2D uimg_rgba16f;
layout(rgba16f) uniform restrict image2D uimg_temp3;
layout(rgba8) uniform restrict writeonly image2D uimg_rgba8;
layout(rg32ui) uniform restrict writeonly uimage2D uimg_rg32ui;
layout(rgba32ui) uniform restrict writeonly uimage2D uimg_rgba32ui;

#ifdef VOXY
#include "/util/VoxyFaceNormal.glsl"
layout(rgba32f) uniform writeonly image2D uimg_hiz;

// Require the pixel footprint on the native face to fit one block. A coarse
// observation must not turn the air beside a distant lamp into another lamp.
bool gi_singleObservedCell(vec3 pos, vec3 normal, float z, vec3 fraction) {
    vec3 ray = pos - gbufferModelViewInverse[3].xyz;
    float facing = dot(ray, normal);
    if (abs(facing) < 1e-5) return false;
    vec2 scale = -2.0 * z * uval_mainImageSizeRcp
               * vec2(global_camProjInverse[0][0], global_camProjInverse[1][1]);
    vec3 dx = gbufferModelViewInverse[0].xyz * scale.x;
    vec3 dy = gbufferModelViewInverse[1].xyz * scale.y;
    dx -= ray * (dot(dx, normal) / facing);
    dy -= ray * (dot(dy, normal) / facing);
    vec3 margin = min(fraction, 1.0 - fraction) + abs(normal);
    return all(greaterThan(margin, 0.5 * (abs(dx) + abs(dy)) + 0.001));
}
void gi_captureWideSurface(ivec2 pixel, float z, uvec2 outgoing) {
    if (global_giWideState != 1) return;
    uint normal = texelFetch(usam_gbufferSolidData1, pixel, 0).r;
    uvec4 raw = texelFetch(usam_gbufferVoxySolidData, pixel, 0);
    bool terrain = z < -nearPlane && z > -65536.0 && raw.w == floatBitsToUint(z);
    uint surfaceCode = terrain ? 65536u + ((raw.z >> 16u) << 3u) + bitfieldExtract(raw.y, 24, 3)
                               : normal & 65535u;
    // FP16Luv has a non-negative finite FP16 luminance in its high half.
    // Bias its bits past FP32 subnormals so the image store preserves dark values.
    uvec2 light = (terrain ? outgoing : uvec2(0u)) + 0x00800000u;
    imageStore(uimg_hiz, pixel + ivec2(0, 2 * uval_mainImageSizeI.y),
               vec4(z, float(surfaceCode), uintBitsToFloat(light)));
}
#endif

#include "/techniques/atmospherics/SkyComposite.glsl"

void doLighting(ivec2 texelPos, Material material, vec3 viewPos, vec3 N, inout vec3 mainOut, inout vec4 giOut1, inout vec4 giOut2) {
    vec3 emissiveV = material.emissive;

    AtmosphereParameters atmosphere = getAtmosphereParameters();

    material.albedo = max(material.albedo, 0.0001);

    vec3 feetPlayerPos = (gbufferModelViewInverse * vec4(viewPos, 1.0)).xyz;
    vec3 worldPos = feetPlayerPos + cameraPosition;
    vec3 atmPos = atmosphere_viewToAtm(atmosphere, viewPos);
    atmPos.y = max(atmPos.y, atmosphere.bottom + 0.1);
    float viewAltitude = length(atmPos);
    vec3 upVector = atmPos / viewAltitude;
    const vec3 earthCenter = vec3(0.0, 0.0, 0.0);
    vec3 V = normalize(-viewPos);

    vec4 shadow = transient_shadow_fetch(texelPos);

    const float decay = 256.0 / pow2(shadow.w);
    material.sss -= rand_stbnVec1(texelPos, frameCounter) * (1.0 - rcp(1.0 + pow2(viewPos.z / decay)));

    float shadowIsSun = float(all(equal(sunPosition, shadowLightPosition)));

    float cosSunZenith = dot(uval_sunDirWorld, vec3(0.0, 1.0, 0.0));
    vec3 tSun = atmospherics_air_lut_sampleTransmittance(atmosphere, cosSunZenith, viewAltitude);
    tSun *= float(raySphereIntersectNearest(atmPos, uval_sunDirWorld, earthCenter + PLANET_RADIUS_OFFSET * upVector, atmosphere.bottom) < 0.0);
    vec4 sunShadow = mix(vec4(1.0), shadow, shadowIsSun);
    vec3 sunIrradiance = SUN_ILLUMINANCE * tSun;
    LightingResult sunLighting = directLighting(lighting_gData, material, sunIrradiance, sunShadow, V, uval_sunDirView, N);

    float cosMoonZenith = dot(uval_moonDirWorld, vec3(0.0, 1.0, 0.0));
    vec3 tMoon = atmospherics_air_lut_sampleTransmittance(atmosphere, cosMoonZenith, viewAltitude);
    tMoon *= float(raySphereIntersectNearest(atmPos, uval_moonDirWorld, earthCenter + PLANET_RADIUS_OFFSET * upVector, atmosphere.bottom) < 0.0);
    vec4 moonShadow = mix(shadow, vec4(1.0), shadowIsSun);
    vec3 moonIrradiance = MOON_ILLUMINANCE * tMoon;
    LightingResult moonLighting = directLighting(lighting_gData, material, moonIrradiance, moonShadow, V, uval_moonDirView, N);

    LightingResult combinedLighting = lightingResult_add(sunLighting, moonLighting);

    mainOut += 0.00001 * material.albedo;
    mainOut += emissiveV;
    mainOut += combinedLighting.diffuse;
    mainOut += combinedLighting.specular;
    mainOut += combinedLighting.sss;

    giOut1.rgb += combinedLighting.diffuseLambertian;

    giOut2.rgb += emissiveV;
}

void main() {
    uint workGroupIdx = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    uvec2 swizzledWGPos = ssbo_threadGroupTiling[workGroupIdx];
    uvec2 workGroupOrigin = swizzledWGPos << 4u;
    uint threadIdx = gl_SubgroupID * gl_SubgroupSize + gl_SubgroupInvocationID;
    uvec2 mortonPos = morton_8bDecode(threadIdx);
    uvec2 mortonGlobalPosU = workGroupOrigin + mortonPos;
    ivec2 texelPos = ivec2(mortonGlobalPosU);

    if (all(lessThan(texelPos, uval_mainImageSizeI))) {

        vec4 mainOut = vec4(0.0);

        float viewZ = hiz_groupGroundCheckSubgroupLoadViewZ(swizzledWGPos.xy, 4, texelPos);
        vec2 screenPos = coords_texelToUV(texelPos, uval_mainImageSizeRcp) - uval_taaJitterUV;
        vec3 viewPos = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);
        ScatteringResult sctrResult = atmospherics_skyComposite(texelPos, viewPos);

        if (viewZ > -65536.0) {
            gbufferData1_unpack(texelFetch(usam_gbufferSolidData1, texelPos, 0), lighting_gData);
            gbufferData2_unpack(texelFetch(usam_gbufferSolidData2, texelPos, 0), lighting_gData);
            transient_lmCoord_store(texelPos, vec4(lighting_gData.lmCoord, 0.0, 0.0));
            Material material = material_decode(lighting_gData);
            vec4 glintColorData = texelFetch(usam_temp4, texelPos, 0);
            vec3 glintColor = colors_material_toWorkSpace(glintColorData.rgb) * glintColorData.a;
            glintColor = pow(glintColor, vec3(SETTING_EMISSIVE_ARMOR_GLINT_CURVE));
            glintColor *= exp2(SETTING_EMISSIVE_STRENGTH + SETTING_EMISSIVE_ARMOR_GLINT_MULT);
            material.emissive += glintColor + material.albedo * glintColor * 4.0;

            vec4 giOut1 = vec4(0.0);
            vec4 giOut2 = vec4(0.0);

            if (lighting_gData.materialID == 65534u) {
                mainOut = vec4(material.albedo * 0.01, 2.0);
                giOut1 = vec4(0.0);
            } else {
                // Specular MB later
                doLighting(texelPos, material, viewPos, lighting_gData.normal, mainOut.rgb, giOut1, giOut2);
            }
            giOut1.rgb += transient_gi2Reprojected_fetch(texelPos).rgb * min(material.albedo, 0.95);

            giOut1.rgb = clamp(giOut1.rgb, 0.0, FP16_MAX);
            giOut2.rgb = clamp(giOut2.rgb, 0.0, FP16_MAX);

            uvec4 giRadianceInput = uvec4(0u);
            giRadianceInput.x = colors_workingColorToFP16Luv(giOut1.rgb);
            giRadianceInput.y = colors_workingColorToFP16Luv(giOut2.rgb);
            transient_giRadianceInputs_store(texelPos, giRadianceInput);
            #ifdef VOXY
            gi_captureWideSurface(texelPos, viewZ, giRadianceInput.xy);
            // Cycle one sample per 8x8 tile. Only native LOD surfaces identify a block state.
            ivec2 phase = ivec2(frameCounter, frameCounter >> 3) & 7;
            if (global_giWideState != 2 && global_camProj[1][1] == global_prevCamProj[1][1]
                && all(equal(texelPos & 7, phase))) {
                uvec4 raw = texelFetch(usam_gbufferVoxySolidData, texelPos, 0);
                uint materialID = raw.z >> 16u;
                if (raw.w == floatBitsToUint(viewZ) && materialID > 0u) {
                    // Voxy applies taaOffset in NDC, so its UV displacement is half the patch value.
                    vec2 uv = coords_texelToUV(texelPos, uval_mainImageSizeRcp) - 0.5 * uval_taaJitterUV;
                    vec3 vp = coords_toViewCoord(uv, viewZ, global_camProjInverse);
                    vec3 scene = (gbufferModelViewInverse * vec4(vp, 1.0)).xyz;
                    vec3 normal = voxy_faceWorldNormal(bitfieldExtract(raw.y, 24, 3));
                    vec3 inward = cameraPositionFract + scene - normal * 0.0625;
                    ivec3 cell = cameraPositionInt - global_giModelOrigin + ivec3(floor(inward));
                    // Discard ambiguous tangential edge pixels rather than assign the adjacent block.
                    vec3 fraction = fract(inward);
                    if (uint(cell.x | cell.y | cell.z) < 32u && gi_singleObservedCell(scene, normal, viewZ, fraction)) {
                        int index = cell.x + 32 * (cell.y + 32 * cell.z);
                        atomicExchange(global_giObservedCells[index].x, materialID);
                        atomicExchange(global_giObservedCells[index].y, global_giModelEpoch);
                        #if SETTING_PBR_MATERIAL == 0
                        if (hardcodedpbr_decode(materialID).emissive > 0.0) {
                            vec2 planeUV = normal.x != 0.0 ? inward.zy : (normal.y != 0.0 ? inward.xz : inward.xy);
                            ivec2 tile = ivec2(fract(planeUV) * 8.0);
                            int sampleIndex = index * 64 + tile.y * 8 + tile.x;
                            atomicExchange(global_giLightSamples[sampleIndex].x, giRadianceInput.y);
                            atomicExchange(global_giLightSampleMaterial[sampleIndex], materialID);
                            atomicExchange(global_giLightSamples[sampleIndex].y, global_giModelEpoch);
                            ivec3 group = cell >> 2;
                            atomicExchange(global_giLightGroupEpoch[group.x + 8 * (group.y + 8 * group.z)], global_giModelEpoch);
                            atomicExchange(global_giLightEpoch, global_giModelEpoch);
                        }
                        #endif
                    }
                }
            }
            #endif
        } else {
            vec4 temp6Out = texelFetch(usam_overlays, texelPos, 0);
            mainOut = celestial_render(texelPos, temp6Out);

            #ifdef SETTING_CONSTELLATIONS
            imageStore(uimg_overlays, texelPos, temp6Out);
            #endif
            mainOut.rgb = scatteringResult_apply(sctrResult, mainOut.rgb);

            transient_giRadianceInputs_store(texelPos, uvec4(0u));
            #ifdef VOXY
            gi_captureWideSurface(texelPos, -65536.0, uvec2(0u));
            #endif
            transient_lmCoord_store(texelPos, vec4(0.0));
        }

        mainOut.rgb = clamp(mainOut.rgb, 0.0, FP16_MAX);
        imageStore(uimg_main, texelPos, mainOut);
    }
}
