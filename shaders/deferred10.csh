#version 460 compatibility

#include "/util/FullScreenComp.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(r32f) uniform image2D uimg_causticsPhoton;

float readInput(vec2 readTexel) {
    readTexel = clamp(readTexel, vec2(0.5), uval_mainImageSize - vec2(0.5));
    readTexel.y *= 0.5;
    vec2 readUV = readTexel * uval_mainImageSizeRcp;
    return texture(usam_causticsPhoton, readUV).r;
}

void main() {
    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        vec2 readPos = vec2(texelPos);
        float filteredInput = readInput(readPos);
        filteredInput += readInput(readPos + vec2(1.0, 0.0));
        filteredInput += readInput(readPos + vec2(0.0, 1.0));
        filteredInput += readInput(readPos + vec2(1.0, 1.0));
        filteredInput *= 0.25;
        ivec2 writePos = texelPos;
        writePos.y += uval_mainImageSizeI.y;
        imageStore(uimg_causticsPhoton, writePos, vec4(filteredInput));
    }
}