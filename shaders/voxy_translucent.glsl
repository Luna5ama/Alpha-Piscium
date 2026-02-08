#define SKIP_UNIFORMS a
#include "/util/Colors2.glsl"
#include "/util/Translucent.glsl"

layout(location = 0) out vec4 image18;
layout(location = 1) out uvec4 image17;

void voxy_emitFragment(VoxyFragmentParameters parameters) {
    vec4 albedo = parameters.sampledColour * parameters.tinting;
    uint materialID = parameters.customId & 0xFFFFu;
    vec3 materialColor = colors2_material_toWorkSpace(albedo.rgb);
    vec4 transmittanceV = translucent_albedoToTransmittance(materialColor, albedo.a, materialID);
    image18 = transmittanceV;
    uint packedUV = packUnorm2x16(parameters.uv);
    uint packedColorFace = packUnorm4x8(vec4(albedo.rgb, float(parameters.face) / 255.0));
    uint lmx = uint(clamp(parameters.lightMap.x, 0.0, 1.0) * 255.0);
    uint lmy = uint(clamp(parameters.lightMap.y, 0.0, 1.0) * 255.0);
    uint lmPacked = lmx | (lmy << 8);
    uint packedLMMat = lmPacked | (materialID << 16);
    vec2 screenUV = gl_FragCoord.xy / vec2(viewWidth, viewHeight);
    vec4 ndc = vec4(screenUV * 2.0 - 1.0, gl_FragCoord.z * 2.0 - 1.0, 1.0);
    vec4 viewP = gbufferProjectionInverse * ndc;
    float viewZ = viewP.z / viewP.w;
    image17 = uvec4(packedUV, packedColorFace, packedLMMat, floatBitsToUint(viewZ));
}
