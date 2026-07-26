#ifndef PARALLAX_COMMON_GLSL
#define PARALLAX_COMMON_GLSL

const ivec2 MATERIAL_DEPTH_MIP_IMAGE_SIZE = ivec2(8192, 12288);

ivec2 mipPackedSize(ivec2 baseSize, int level) {
    return ((baseSize - 1) >> level) + 1;
}

ivec2 mipPackedOffset(ivec2 baseSize, int level) {
    if (level <= 1) {
        return ivec2(0);
    }

    ivec2 offset = ivec2(0, mipPackedSize(baseSize, 1).y);
    for (int i = 2; i < level; i++) {
        offset.x += mipPackedSize(baseSize, i).x;
    }
    return offset;
}

#endif
