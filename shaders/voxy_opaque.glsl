#define SKIP_UNIFORMS a
#include "/Base.glsl"

layout(location = 0) out uvec4 image16;

void voxy_emitFragment(VoxyFragmentParameters parameters) {
    vec3 base_color = parameters.sampledColour.rgb * parameters.tinting.rgb;

    // R: UV (2x16 unorm)
    uint packedUV = packUnorm2x16(parameters.uv);

    // G: Color (3x8 unorm) + Face (3 bits in alpha slot)
    // Note: parameters.face is uint 0-5 typically.
    // We map 0-255 to 0-1 for packUnorm.
    // face -> float(face)/255.0
    uint packedColorFace = packUnorm4x8(vec4(base_color, float(parameters.face) / 255.0));

    // B: Lightmap (2x8 unorm) + MaterialID (16 bits)
    uint lmx = uint(clamp(parameters.lightMap.x, 0.0, 1.0) * 255.0);
    uint lmy = uint(clamp(parameters.lightMap.y, 0.0, 1.0) * 255.0);
    uint lmPacked = lmx | (lmy << 8);
    uint matID = parameters.customId & 0xFFFFu;
    uint packedLMMat = lmPacked | (matID << 16);

    // Calculate ViewZ
    vec2 screenUV = gl_FragCoord.xy / vec2(viewWidth, viewHeight);
    vec4 ndc = vec4(screenUV * 2.0 - 1.0, gl_FragCoord.z * 2.0 - 1.0, 1.0);
    vec4 viewP = gbufferProjectionInverse * ndc;
    float viewZ = viewP.z / viewP.w;

    image16 = uvec4(packedUV, packedColorFace, packedLMMat, floatBitsToUint(viewZ));
}

