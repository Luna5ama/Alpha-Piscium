#include "/util/FullScreenComp.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(r32f) uniform image2D uimg_r32f;

float readInput(vec2 readTexel) {
    readTexel = clamp(readTexel, vec2(0.5), uval_mainImageSize - vec2(0.5));
    vec2 readUV = readTexel * uval_mainImageSizeRcp;
    return transient_caustics_remapped_sample(readUV).r;
}

void main() {
    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        vec2 readPos = vec2(texelPos);
        float filteredInput = readInput(readPos);
        filteredInput += readInput(readPos + vec2(1.0, 0.0));
        filteredInput += readInput(readPos + vec2(0.0, 1.0));
        filteredInput += readInput(readPos + vec2(1.0, 1.0));
        filteredInput *= 0.25;
        transient_caustics_final_store(texelPos, vec4(filteredInput));
    }
}