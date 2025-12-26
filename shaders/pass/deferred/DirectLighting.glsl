#extension GL_KHR_shader_subgroup_ballot : enable

#include "/techniques/atmospherics/air/Constants.glsl"
#include "/techniques/atmospherics/air/lut/API.glsl"
#include "/techniques/gi/Common.glsl"
#include "/techniques/Lighting.glsl"
#include "/util/Celestial.glsl"
#include "/util/Colors2.glsl"
#include "/util/BitPacking.glsl"
#include "/util/NZPacking.glsl"
#include "/util/Morton.glsl"
#include "/techniques/HiZCheck.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);


layout(rgba16f) uniform writeonly image2D uimg_main;
layout(rgba16f) uniform restrict image2D uimg_rgba16f;
layout(rgba16f) uniform restrict image2D uimg_temp3;

ivec2 texelPos;

void doLighting(Material material, vec3 viewPos, vec3 N, inout vec3 directDiffuseOut, inout vec3 mainOut, inout vec4 giOut1, inout vec4 giOut2) {
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

    vec3 shadow = transient_shadow_fetch(texelPos).rgb;
    float shadowIsSun = float(all(equal(sunPosition, shadowLightPosition)));

    float cosSunZenith = dot(uval_sunDirWorld, vec3(0.0, 1.0, 0.0));
    vec3 tSun = atmospherics_air_lut_sampleTransmittance(atmosphere, cosSunZenith, viewAltitude);
    tSun *= float(raySphereIntersectNearest(atmPos, uval_sunDirWorld, earthCenter + PLANET_RADIUS_OFFSET * upVector, atmosphere.bottom) < 0.0);
    vec3 sunShadow = mix(vec3(1.0), shadow, shadowIsSun);
    vec4 sunIrradiance = vec4(SUN_ILLUMINANCE * tSun * sunShadow, colors2_colorspaces_luma(COLORS2_WORKING_COLORSPACE, sunShadow));
    LightingResult sunLighting = directLighting(material, sunIrradiance, V, uval_sunDirView, N);

    float cosMoonZenith = dot(uval_moonDirWorld, vec3(0.0, 1.0, 0.0));
    vec3 tMoon = atmospherics_air_lut_sampleTransmittance(atmosphere, cosMoonZenith, viewAltitude);
    tMoon *= float(raySphereIntersectNearest(atmPos, uval_moonDirWorld, earthCenter + PLANET_RADIUS_OFFSET * upVector, atmosphere.bottom) < 0.0);
    vec3 moonShadow = mix(shadow, vec3(1.0), shadowIsSun);
    vec4 moonIrradiance = vec4(MOON_ILLUMINANCE * tMoon * moonShadow, colors2_colorspaces_luma(COLORS2_WORKING_COLORSPACE, moonShadow));
    LightingResult moonLighting = directLighting(material, moonIrradiance, V, uval_moonDirView, N);

    LightingResult combinedLighting = lightingResult_add(sunLighting, moonLighting);

    mainOut += 0.00001 * material.albedo;
    mainOut += emissiveV * RCP_PI;
    mainOut += combinedLighting.diffuse;
    mainOut += combinedLighting.specular;
    mainOut += combinedLighting.sss;

    directDiffuseOut += combinedLighting.diffuse;
    directDiffuseOut += combinedLighting.sss;
    directDiffuseOut /= material.albedo;

    giOut1.rgb += combinedLighting.diffuseLambertian * 4.0; // idk this needed to match reference
    giOut1.rgb += combinedLighting.sss * 4.;

    giOut2.rgb += emissiveV;
}

void main() {
    uvec2 workGroupOrigin = gl_WorkGroupID.xy << 4;
    uint threadIdx = gl_SubgroupID * gl_SubgroupSize + gl_SubgroupInvocationID;
    uvec2 mortonPos = morton_8bDecode(threadIdx);
    uvec2 mortonGlobalPosU = workGroupOrigin + mortonPos;
    texelPos = ivec2(mortonGlobalPosU);

    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        ivec2 texelPos2x2 = texelPos >> 1;
        vec2 screenPos = (vec2(texelPos) + 0.5) * uval_mainImageSizeRcp;

        float viewZ = hiz_groupGroundCheckSubgroupLoadViewZ(gl_WorkGroupID.xy, 4, texelPos);
        if (viewZ > -65536.0) {
            gbufferData1_unpack(texelFetch(usam_gbufferData1, texelPos, 0), lighting_gData);
            gbufferData2_unpack(texelFetch(usam_gbufferData2, texelPos, 0), lighting_gData);
            Material material = material_decode(lighting_gData);
            vec4 glintColorData = texelFetch(usam_temp4, texelPos, 0);
            vec3 glintColor = colors2_material_toWorkSpace(glintColorData.rgb) * glintColorData.a;
            glintColor = pow(glintColor, vec3(SETTING_EMISSIVE_ARMOR_GLINT_CURVE));
            glintColor *= exp2(SETTING_EMISSIVE_STRENGTH + SETTING_EMISSIVE_ARMOR_GLINT_MULT);
            material.emissive += glintColor + material.albedo * glintColor * 4.0;

            vec3 viewPos = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);
            ivec2 texelPos2x2 = texelPos >> 1;
            ivec2 radianceTexelPos = texelPos2x2 + ivec2(0, global_mipmapSizesI[1].y);

            vec4 giOut1 = vec4(0.0);
            vec4 giOut2 = vec4(0.0);

            {
                vec3 moment1 = vec3(0.0);
                vec3 moment2 = vec3(0.0);

                for (int dy = -2; dy <= 2; ++dy) {
                    for (int dx = -2; dx <= 2; ++dx) {
                        if (dx == 0 && dy == 0) continue;
                        ivec2 neighborPos = texelPos + ivec2(dx, dy);
                        vec3 neighborData = colors_SRGBToYCoCg(transient_gi1Reprojected_load(neighborPos).rgb);
                        moment1 += neighborData;
                        moment2 += neighborData * neighborData;
                    }
                }

                moment1 /= 24.0;
                moment2 /= 24.0;

                vec3 mean = moment1;
                vec3 variance = max(moment2 - moment1 * moment1, 0.0);
                vec3 stddev = sqrt(variance);
                vec3 aabbMin = mean - stddev * 1.0;
                vec3 aabbMax = mean + stddev * 1.0;
                vec3 giOut1Clamped = colors_SRGBToYCoCg(transient_gi1Reprojected_load(texelPos).rgb);
                giOut1Clamped = clamp(giOut1Clamped, aabbMin, aabbMax);
                giOut1.rgb = colors_YCoCgToSRGB(giOut1Clamped);
            }

            vec4 mainOut = vec4(0.0, 0.0, 0.0, 1.0);
            vec3 directDiffuseOut = vec3(0.0);
            if (lighting_gData.materialID == 65534u) {
                mainOut = vec4(material.albedo * 0.01, 2.0);
                giOut1 = vec4(0.0);
            } else {
                giOut1.rgb *= material.albedo;
                giOut1.rgb *= GI_MB;
                doLighting(material, viewPos, lighting_gData.normal, directDiffuseOut, mainOut.rgb, giOut1, giOut2);
                float albedoLuma = colors2_colorspaces_luma(COLORS2_WORKING_COLORSPACE, colors2_material_toWorkSpace(material.albedo));
                float emissiveFlag = float(any(greaterThan(material.emissive, vec3(0.0))));
//                float albedoLumaWeight = pow(1.0 - pow(1.0 - albedoLuma, 16.0), 4.0);
//                albedoLumaWeight += pow(albedoLuma, 16.0);
//                mainOut.a *= albedoLumaWeight;
//                mainOut.a *= mix(1.0, -1.0, emissiveFlag);
            }

            mainOut.rgb = clamp(mainOut.rgb, 0.0, FP16_MAX);
            giOut1.rgb = clamp(giOut1.rgb, 0.0, FP16_MAX);
            giOut2.rgb = clamp(giOut2.rgb, 0.0, FP16_MAX);

            uvec4 packedZNOut = uvec4(0u);
            nzpacking_pack(packedZNOut.xy, lighting_gData.normal, viewZ);
//                transient_packedZN_store(texelPos + ivec2(0, uval_mainImageSizeI.y), packedZNOut);

//                uint ssgiOutWriteFlag = uint(vbgi_selectDownSampleInput(threadIdx));
//                ssgiOutWriteFlag &= uint(all(lessThan(texelPos2x2, global_mipmapSizesI[1])));
//                if (bool(ssgiOutWriteFlag)) {
////                    transient_packedZN_store(texelPos2x2, packedZNOut);
////                    transient_packedZN_store(radianceTexelPos, uvec4(packHalf2x16(ssgiOut.rg), packHalf2x16(ssgiOut.ba), 0u, 0u));
//                }

            imageStore(uimg_main, texelPos, mainOut);
            transient_giRadianceInput1_store(texelPos, giOut1);
            transient_giRadianceInput2_store(texelPos, giOut2);
//            imageStore(uimg_temp3, texelPos, giOut1);
            return;
        }

        {
            vec3 directDiffuseOut = vec3(0.0);

            uvec4 packedZNOut = uvec4(0u);
            packedZNOut.y = floatBitsToUint(-65536.0);
//            transient_packedZN_store(texelPos + ivec2(0, uval_mainImageSizeI.y), packedZNOut);

//            uint ssgiOutWriteFlag = uint(vbgi_selectDownSampleInput(threadIdx));
//            ssgiOutWriteFlag &= uint(all(lessThan(texelPos2x2, global_mipmapSizesI[1])));
//            if (bool(ssgiOutWriteFlag)) {
////                transient_packedZN_store(texelPos2x2, packedZNOut);
//            }
            transient_giRadianceInput1_store(texelPos, vec4(0.0));
            transient_giRadianceInput2_store(texelPos, vec4(0.0));
        }
    }
}