#define SKIP_UNIFORMS a
#include "/util/Colors2.glsl"
#include "/util/Translucent.glsl"

layout(location = 0) out uvec4 rt_gbufferData;
layout(location = 1) out vec4 rt_translucentColor;

void voxy_emitFragment(VoxyFragmentParameters parameters) {

    vec4 albedo = parameters.sampledColour * parameters.tinting;
    uint materialID = parameters.customId & 0xFFFFu;
    vec3 materialColor = colors2_material_toWorkSpace(albedo.rgb);
    vec4 transmittanceV = translucent_albedoToTransmittance(materialColor, albedo.a, materialID);
    rt_translucentColor = transmittanceV;

    uint packedUV = packUnorm2x16(parameters.uv);
    uint packedColorFace = packUnorm4x8(vec4(albedo.rgb, float(parameters.face) / 255.0));
    uint lmx = uint(clamp(parameters.lightMap.x, 0.0, 1.0) * 255.0);
    uint lmy = uint(clamp(parameters.lightMap.y, 0.0, 1.0) * 255.0);
    uint lmPacked = lmx | (lmy << 8);
    uint packedLMMat = lmPacked | (materialID << 16);

    float viewZ = -rcp(gl_FragCoord.w);
    rt_gbufferData = uvec4(packedUV, packedColorFace, packedLMMat, floatBitsToUint(viewZ));
}
