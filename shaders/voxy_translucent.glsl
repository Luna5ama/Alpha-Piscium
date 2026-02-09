#define SKIP_UNIFORMS a
#define DISABLE_FP16 a
#include "/util/Math.glsl"
#include "/util/Colors2.glsl"
#include "/util/Translucent.glsl"

layout(location = 0) out uvec4 rt_gbufferData;
layout(location = 1) out vec4 rt_translucentColor;

void voxy_emitFragment(VoxyFragmentParameters parameters) {
    vec4 color = parameters.sampledColour * parameters.tinting;
    uint materialID = parameters.customId & 0xFFFFu;

    float alpha = color.a;
    vec3 materialColor = colors2_material_toWorkSpace(color.rgb);
    vec4 transmittanceV = translucent_albedoToTransmittance(materialColor, alpha, materialID);
    rt_translucentColor = transmittanceV;

    // R: UV (2x16 unorm)
    uint packedUV = packUnorm2x16(parameters.uv);

    // G: Color (3x8 unorm) + Face (3 bits in alpha slot)
    uint packedColorFace = packUnorm4x8(vec4(color.rgb, 0.0));
    packedColorFace = bitfieldInsert(packedColorFace, parameters.face, 24, 3);

    // B: Lightmap (2x8 unorm) + MaterialID (16 bits)
    vec2 lmCoord = parameters.lightMap;
    lmCoord.x = linearStep(0.0625, 0.96875, lmCoord.x);
    lmCoord.y = linearStep(0.125, 0.73438, lmCoord.y);
    uint lmPacked = packUnorm4x8(vec4(lmCoord, 0.0, 0.0));
    uint packedLMMat = bitfieldInsert(lmPacked, materialID, 16, 16);

    // A: ViewZ (float as uint)
    float viewZ = -rcp(gl_FragCoord.w);

    rt_gbufferData = uvec4(packedUV, packedColorFace, packedLMMat, floatBitsToUint(viewZ));
}
