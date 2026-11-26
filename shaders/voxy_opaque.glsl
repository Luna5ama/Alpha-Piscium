#define SKIP_UNIFORMS a
#define DISABLE_FP16 a
#include "/util/Math.glsl"

layout(location = 0) out uvec4 rt_gbufferData;

#include "/util/VoxyModelAtlas.glsl"

// Iris preprocesses this hook before Voxy appends its generated SSBO header.
void voxy_captureEmissionAtlas(uint key, uint modelId) {
    const uint rowCount = 6u * 16u;
    uint slot = voxy_atlasSlot(key);
    uint state = uint(GI_VOXY_ATLAS_OFFSET) / 16u + slot;
    if (apAtlasWords[state].x != key || apAtlasWords[state].y >= rowCount) return;
    uint row = atomicAdd(apAtlasWords[state].y, 1u);
    if (row >= rowCount) return;
    uint face = row >> 4u;
    ivec2 faceSize = textureSize(blockModelAtlas, 0) / ivec2(768, 512);
    ivec2 model = ivec2(modelId & 255u, modelId >> 8u);
    ivec2 base = (model * ivec2(3, 2) + ivec2(face >> 1u, face & 1u)) * faceSize;
    uint dest = uint(GI_VOXY_ATLAS_OFFSET) / 16u + GI_VOXY_MODEL_SLOTS
              + slot * 6u * GI_VOXY_FACE_WORDS + row * 4u;
    for (uint word = 0u; word < 4u; ++word) {
        uvec4 pixels;
        for (uint k = 0u; k < 4u; ++k) {
            ivec2 uv = ivec2(4u * word + k, row & 15u);
            pixels[k] = packUnorm4x8(texelFetch(blockModelAtlas, base + (2 * uv + 1) * faceSize / 32, 0));
        }
        apAtlasWords[dest + word] = pixels;
    }
    memoryBarrierBuffer();
    // No consumer may see a partial face, including when only a few fragments
    // of the emitter were visible and filling the atlas spanned several frames.
    if (atomicAdd(apAtlasWords[state].w, 1u) == rowCount - 1u) atomicExchange(apAtlasWords[state].z, key);
}

void voxy_emitFragment(VoxyFragmentParameters parameters) {
    vec4 color = parameters.sampledColour * parameters.tinting;
    uint materialID = parameters.customId & 0xFFFFu;

    // R: UV (2x16 unorm)
    uint packedUV = packUnorm2x16(parameters.uv);

    // G: Color (3x8 unorm) + Face (3 bits in alpha slot)
    uint packedColorFace = packUnorm4x8(vec4(color.rgb, 0.0));
    packedColorFace = bitfieldInsert(packedColorFace, parameters.face, 24, 3);

    // Disambiguate UV tile edges and flag unmodified model colors for GI.
    packedColorFace = bitfieldInsert(packedColorFace, parameters.modelId, 27, 2);
    packedColorFace = bitfieldInsert(packedColorFace, parameters.modelId >> 8u, 29, 2);
    if (all(equal(parameters.tinting.rgb, vec3(1.0)))) {
        packedColorFace |= 0x80000000u;
        if (materialID != 0u) voxy_captureEmissionAtlas((materialID << 16u) | parameters.modelId, parameters.modelId);
    }

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

