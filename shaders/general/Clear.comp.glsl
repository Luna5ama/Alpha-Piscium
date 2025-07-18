#include "/Base.glsl"

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    uint bitFlag = uint(all(greaterThanEqual(texelPos, CLEAR_IMAGE_BOUND.xy)));
    bitFlag |= uint(all(lessThanEqual(texelPos, CLEAR_IMAGE_BOUND.zw)));
    if (bool(bitFlag)) {
        #ifdef CLEAR_IMAGE1
        imageStore(CLEAR_IMAGE1, texelPos, CLEAR_COLOR1);
        #endif
        #ifdef CLEAR_IMAGE2
        imageStore(CLEAR_IMAGE2, texelPos, CLEAR_COLOR2);
        #endif
        #ifdef CLEAR_IMAGE3
        imageStore(CLEAR_IMAGE3, texelPos, CLEAR_COLOR3);
        #endif
        #ifdef CLEAR_IMAGE4
        imageStore(CLEAR_IMAGE4, texelPos, CLEAR_COLOR4);
        #endif
        #ifdef CLEAR_IMAGE5
        imageStore(CLEAR_IMAGE5, texelPos, CLEAR_COLOR5);
        #endif
        #ifdef CLEAR_IMAGE6
        imageStore(CLEAR_IMAGE6, texelPos, CLEAR_COLOR6);
        #endif
        #ifdef CLEAR_IMAGE7
        imageStore(CLEAR_IMAGE7, texelPos, CLEAR_COLOR7);
        #endif
        #ifdef CLEAR_IMAGE8
        imageStore(CLEAR_IMAGE8, texelPos, CLEAR_COLOR8);
        #endif
    }
}
