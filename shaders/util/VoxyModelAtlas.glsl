#ifndef INCLUDE_util_VoxyModelAtlas_glsl
#define INCLUDE_util_VoxyModelAtlas_glsl a

#define GI_VOXY_ATLAS_OFFSET 25446416
#define GI_VOXY_MODEL_SLOTS 256u
#define GI_VOXY_FACE_WORDS 64u

uint voxy_atlasSlot(uint key) {
    return (key * 0x9e3779b9u) >> 24u;
}

uint voxy_observedModel(uvec4 raw) {
    ivec2 tile = min(ivec2(unpackUnorm2x16(raw.x) * 256.0), ivec2(255));
    // The UV may round into the adjacent tile at a packed 16-bit boundary.
    ivec2 low = ivec2(bitfieldExtract(raw.y, 27, 2), bitfieldExtract(raw.y, 29, 2));
    tile += ((low - tile + 2) & 3) - 2;
    return uint(tile.x + 256 * tile.y);
}

#endif
