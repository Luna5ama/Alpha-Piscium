#version 460 compatibility

#extension GL_KHR_shader_subgroup_basic : enable

#include "/util/Celestial.glsl"
#include "/util/BitPacking.glsl"
#include "/util/NZPacking.glsl"
#include "/util/Morton.glsl"
#include "/atmosphere/Common.glsl"
#include "/general/Lighting.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

uniform sampler2D usam_temp5;
uniform usampler2D usam_gbufferData32UI;
uniform sampler2D usam_gbufferData8UN;
uniform sampler2D usam_gbufferViewZ;

layout(rgba16f) uniform writeonly image2D uimg_main;
layout(rgba16f) uniform restrict image2D uimg_temp4;
layout(rgba8) uniform restrict image2D uimg_temp6;
layout(rg32ui) uniform restrict uimage2D uimg_packedZN;
layout(r32ui) uniform writeonly uimage2D uimg_geometryNormal;

ivec2 texelPos;

void doLighting(Material material, vec3 N, inout vec3 directDiffuseOut, inout vec3 mainOut, inout vec3 ssgiOut) {
    vec3 emissiveV = material.emissive;

    AtmosphereParameters atmosphere = getAtmosphereParameters();

    float alpha = material.roughness * material.roughness;

    vec3 feetPlayerPos = (gbufferModelViewInverse * vec4(lighting_viewCoord, 1.0)).xyz;
    vec3 worldPos = feetPlayerPos + cameraPosition;
    vec3 atmPos = atmosphere_viewToAtm(atmosphere, lighting_viewCoord);
    atmPos.y = max(atmPos.y, atmosphere.bottom + 0.5);
    float viewAltitude = length(atmPos);
    vec3 upVector = atmPos / viewAltitude;
    const vec3 earthCenter = vec3(0.0, 0.0, 0.0);

    vec3 shadow = texelFetch(usam_temp5, texelPos, 0).rgb;
    float shadowIsSun = float(all(equal(sunPosition, shadowLightPosition)));

    float cosSunZenith = dot(uval_sunDirWorld, vec3(0.0, 1.0, 0.0));
    vec3 tSun = sampleTransmittanceLUT(atmosphere, cosSunZenith, viewAltitude);
    tSun *= float(raySphereIntersectNearest(atmPos, uval_sunDirWorld, earthCenter + PLANET_RADIUS_OFFSET * upVector, atmosphere.bottom) < 0.0);
    vec3 sunShadow = mix(vec3(1.0), shadow, shadowIsSun);
    vec4 sunIrradiance = vec4(SUN_ILLUMINANCE * tSun * sunShadow, colors_sRGB_luma(sunShadow));
    LightingResult sunLighting = directLighting(material, sunIrradiance, uval_sunDirView, N);

    float cosMoonZenith = dot(uval_moonDirWorld, vec3(0.0, 1.0, 0.0));
    vec3 tMoon = sampleTransmittanceLUT(atmosphere, cosMoonZenith, viewAltitude);
    tMoon *= float(raySphereIntersectNearest(atmPos, uval_moonDirWorld, earthCenter + PLANET_RADIUS_OFFSET * upVector, atmosphere.bottom) < 0.0);
    vec3 moonShadow = mix(shadow, vec3(1.0), shadowIsSun);
    vec4 moonIrradiance = vec4(MOON_ILLUMINANCE * tMoon * moonShadow, colors_sRGB_luma(moonShadow));
    LightingResult moonLighting = directLighting(material, moonIrradiance, uval_moonDirView, N);

    LightingResult combinedLighting = lightingResult_add(sunLighting, moonLighting);

    mainOut += 0.00001 * material.albedo;
    mainOut += emissiveV;
    mainOut += combinedLighting.specular;

    directDiffuseOut += combinedLighting.diffuse;
    directDiffuseOut += combinedLighting.sss;
    directDiffuseOut /= material.albedo;

    ssgiOut += emissiveV * 2.0;
    ssgiOut += combinedLighting.diffuseLambertian * 4.0;
    ssgiOut += combinedLighting.sss * 4.0;
}

void main() {
    uvec2 workGroupOrigin = gl_WorkGroupID.xy << 4;
    uint threadIdx = gl_SubgroupID * gl_SubgroupSize + gl_SubgroupInvocationID;
    uvec2 mortonPos = morton_8bDecode(threadIdx);
    uvec2 mortonGlobalPosU = workGroupOrigin + mortonPos;
    texelPos = ivec2(mortonGlobalPosU);

    if (all(lessThan(texelPos, global_mainImageSizeI))) {
        ivec2 texelPos2x2 = texelPos >> 1;
        vec2 screenPos = (vec2(texelPos) + 0.5) * global_mainImageSizeRcp;

        float viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;

        vec4 mainOut = vec4(0.0, 0.0, 0.0, 1.0);
        vec3 directDiffuseOut = vec3(0.0);

        if (viewZ != -65536.0) {
            gbufferData1_unpack(texelFetch(usam_gbufferData32UI, texelPos, 0), lighting_gData);
            gbufferData2_unpack(texelFetch(usam_gbufferData8UN, texelPos, 0), lighting_gData);
            Material material = material_decode(lighting_gData);

            lighting_init(coords_toViewCoord(screenPos, viewZ, gbufferProjectionInverse), texelPos);
            ivec2 texelPos2x2 = texelPos >> 1;
            ivec2 radianceTexelPos = texelPos2x2 + ivec2(0, global_mipmapSizesI[1].y);

            uvec2 radianceData = imageLoad(uimg_packedZN, radianceTexelPos).xy;
            vec4 ssgiOut = vec4(unpackHalf2x16(radianceData.x), unpackHalf2x16(radianceData.y));

            if (lighting_gData.materialID == 65534u) {
                mainOut = vec4(material.albedo * 0.01, 2.0);
            } else {
                doLighting(material, lighting_gData.normal, directDiffuseOut, mainOut.rgb, ssgiOut.rgb);
                float albedoLuma = colors_Rec601_luma(lighting_gData.albedo);
                float emissiveFlag = float(any(greaterThan(material.emissive, vec3(0.0))));
                mainOut.a += emissiveFlag * albedoLuma * SETTING_EXPOSURE_EMISSIVE_WEIGHTING;
                float albedoLumaWeight = pow(1.0 - pow(1.0 - albedoLuma, 16.0), 4.0);
                albedoLumaWeight += pow(albedoLuma, 16.0);
                mainOut.a *= albedoLumaWeight;
                mainOut.a *= mix(1.0, -1.0, emissiveFlag);
            }

            uint packedGeometryNormal = packSnorm3x10(lighting_gData.geometryNormal);
            imageStore(uimg_geometryNormal, texelPos, uvec4(packedGeometryNormal));

            uvec4 packedZNOut = uvec4(0u);
            nzpacking_pack(packedZNOut.xy, lighting_gData.normal, viewZ);
            imageStore(uimg_packedZN, texelPos + ivec2(0, global_mainImageSizeI.y), packedZNOut);

            uint ssgiOutWriteFlag = uint((threadIdx & 3u) == (uint(frameCounter) & 3u));
            ssgiOutWriteFlag &= uint(all(lessThan(texelPos2x2, global_mipmapSizesI[1])));
            if (bool(ssgiOutWriteFlag)) {
                imageStore(uimg_packedZN, texelPos2x2, packedZNOut);
                imageStore(uimg_packedZN, radianceTexelPos, uvec4(packHalf2x16(ssgiOut.rg), packHalf2x16(ssgiOut.ba), 0u, 0u));
            }
        } else {
            vec4 temp6Out = vec4(0.0);
            mainOut = celestial_render(texelPos, temp6Out);

            uvec4 packedZNOut = uvec4(0u);
            packedZNOut.y = floatBitsToUint(viewZ);
            imageStore(uimg_packedZN, texelPos + ivec2(0, global_mainImageSizeI.y), packedZNOut);

            uint ssgiOutWriteFlag = uint((threadIdx & 3u) == (uint(frameCounter) & 3u));
            ssgiOutWriteFlag &= uint(all(lessThan(texelPos2x2, global_mipmapSizesI[1])));
            if (bool(ssgiOutWriteFlag)) {
                imageStore(uimg_packedZN, texelPos2x2, packedZNOut);
            }

            #ifdef SETTING_CONSTELLATIONS
            vec4 prevTemp6Value = imageLoad(uimg_temp6, texelPos);
            temp6Out.rgb += temp6Out.rgb;
            temp6Out.a += temp6Out.a;
            imageStore(uimg_temp6, texelPos, temp6Out);
            #endif
        }

        imageStore(uimg_temp4, texelPos, vec4(directDiffuseOut, 0.0));
        imageStore(uimg_main, texelPos, mainOut);
    }
}