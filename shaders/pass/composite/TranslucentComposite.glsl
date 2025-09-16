#include "/techniques/SST.glsl"
#include "/util/FullScreenComp.glsl"
#include "/util/GBufferData.glsl"
#include "/util/Material.glsl"
#include "/util/Coords.glsl"
#include "/util/Colors.glsl"
#include "/util/Fresnel.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform restrict image2D uimg_main;
layout(rgba16f) uniform writeonly image2D uimg_csrgba16f;

vec3 SampleVNDFGGX(
vec3 viewerDirection, // Direction pointing towards the viewer, oriented such that +Z corresponds to the surface normal
vec2 alpha, // Roughness parameter along X and Y of the distribution
vec2 xy // Pair of uniformly distributed numbers in [0, 1)
) {
    // Transform viewer direction to the hemisphere configuration
    viewerDirection = normalize(vec3(alpha * viewerDirection.xy, viewerDirection.z));

    // Sample a reflection direction off the hemisphere
    const float tau = 6.2831853; // 2 * pi
    float phi = tau * xy.x;
    float cosTheta = fma(1.0 - xy.y, 1.0 + viewerDirection.z, -viewerDirection.z);
    float sinTheta = sqrt(clamp(1.0 - cosTheta * cosTheta, 0.0, 1.0));
    vec3 reflected = vec3(vec2(cos(phi), sin(phi)) * sinTheta, cosTheta);

    // Evaluate halfway direction
    // This gives the normal on the hemisphere
    vec3 halfway = reflected + viewerDirection;

    // Transform the halfway direction back to hemiellispoid configuation
    // This gives the final sampled normal
    return normalize(vec3(alpha * halfway.xy, halfway.z));
}

void main() {
    if (all(lessThan(texelPos, global_mainImageSizeI))) {
        vec4 outputColor = texelFetch(usam_temp1, texelPos, 0);

            ivec2 farDepthTexelPos = texelPos;
            ivec2 nearDepthTexelPos = texelPos;
            farDepthTexelPos.y += global_mainImageSizeI.y;
            nearDepthTexelPos += global_mainImageSizeI;

            float startViewZ = -texelFetch(usam_translucentDepthLayers, nearDepthTexelPos, 0).r;
//            float endViewZ = -texelFetch(usam_translucentDepthLayers, farDepthTexelPos, 0).r;
//            float startViewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;

        if (startViewZ > -65536.0) {
            vec2 screenPos = coords_texelToUV(texelPos, global_mainImageSizeRcp);
            vec3 startViewPos = coords_toViewCoord(screenPos, startViewZ, global_camProjInverse);

            GBufferData gData = gbufferData_init();
            gbufferData1_unpack(texelFetch(usam_gbufferData1, texelPos, 0), gData);
            gbufferData2_unpack(texelFetch(usam_gbufferData2, texelPos, 0), gData);

            Material material = material_decode(gData);

            vec3 viewDir = normalize(-startViewPos);

            vec3 bitangent = cross(gData.normal, gData.geomTangent) * float(gData.bitangentSign);
            mat3 tbn = mat3(gData.geomTangent, bitangent, gData.normal);
            mat3 tbnInv = inverse(tbn);
            vec3 localViewDir = normalize(tbnInv * viewDir);

            vec2 noiseV = rand_stbnVec2(texelPos, frameCounter);
            vec3 localMicroNormal = SampleVNDFGGX(localViewDir, vec2(material.roughness * 0.5), noiseV);
            vec3 microNormal = normalize(tbn * localMicroNormal);

            float rior = AIR_IOR / material.hardCodedIOR;
            vec3 refractDir = refract(-viewDir, microNormal, rior);

            SSTResult result = sst_trace(startViewPos, refractDir);
            if (result.hit) {
                vec2 coord = result.hitScreenPos.xy;
                outputColor = texture(usam_temp1, result.hitScreenPos.xy);
            }
        }

        vec4 translucentTransmittance = texelFetch(usam_translucentColor, texelPos, 0);
        outputColor.rgb *= translucentTransmittance.rgb;
        imageStore(uimg_main, texelPos, outputColor);

        #ifdef SETTING_DOF
        float viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;
        outputColor.a = abs(viewZ);
        imageStore(uimg_csrgba16f, csrgba16f_temp1_texelToTexel(texelPos), outputColor);
        #endif
    }
}