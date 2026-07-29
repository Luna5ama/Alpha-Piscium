#ifndef INCLUDE_techniques_parallax_Common_glsl
#define INCLUDE_techniques_parallax_Common_glsl a

const ivec2 MATERIAL_DEPTH_MIP_IMAGE_SIZE = ivec2(8192, 12288);

ivec2 parallax_mipPackedSize(ivec2 baseSize, int level) {
    return ((baseSize - 1) >> level) + 1;
}

ivec2 parallax_mipPackedOffset(ivec2 baseSize, int level) {
    if (level == 0) {
        return ivec2(0);
    }

    ivec2 offset = ivec2(0, parallax_mipPackedSize(baseSize, 0).y);
    for (int i = 1; i < level; i++) {
        offset.x += parallax_mipPackedSize(baseSize, i).x;
    }
    return offset;
}

#endif
