#define SKIP_UNIFORMS a
#define DISABLE_FP16 a
#include "/util/Math.glsl"

layout(location = 0) out uvec4 rt_gbufferData;
layout(location = 1) out vec4 rt_test;

void voxy_emitFragment(VoxyFragmentParameters parameters) {
    vec3 base_color = parameters.sampledColour.rgb * parameters.tinting.rgb;

    // R: UV (2x16 unorm)
    uint packedUV = packUnorm2x16(parameters.uv);

    // G: Color (3x8 unorm) + Face (3 bits in alpha slot)
    uint packedColorFace = packUnorm4x8(vec4(base_color, 0.0));
    packedColorFace = bitfieldInsert(packedColorFace, parameters.face, 24, 3);

    // B: Lightmap (2x8 unorm) + MaterialID (16 bits)
    uint lmx = uint(clamp(parameters.lightMap.x, 0.0, 1.0) * 255.0);
    uint lmy = uint(clamp(parameters.lightMap.y, 0.0, 1.0) * 255.0);
    uint lmPacked = lmx | (lmy << 8);
    uint matID = parameters.customId & 0xFFFFu;
    uint packedLMMat = lmPacked | (matID << 16);

    // A: ViewZ (float as uint)
    float viewZ = -rcp(gl_FragCoord.w);

    rt_gbufferData = uvec4(packedUV, packedColorFace, packedLMMat, floatBitsToUint(viewZ));
    rt_test = gl_FragCoord;
}

