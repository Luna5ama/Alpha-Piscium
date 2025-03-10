#include "/util/Coords.glsl"

void nzpacking_pack(out uvec2 packedData, vec3 normal, float depth) {
    packedData.x = packSnorm2x16(coords_octEncode11(normal));
    packedData.y = floatBitsToUint(depth);
}

void nzpacking_unpack(uvec2 packedData, out vec3 normal, out float depth) {
    normal = coords_octDecode11(unpackSnorm2x16(packedData.x));
    depth = uintBitsToFloat(packedData.y);
}