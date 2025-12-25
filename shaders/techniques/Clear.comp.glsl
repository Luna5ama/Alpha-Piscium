#include "/Base.glsl"

/*const*/
#ifndef CLEAR_OFFSET1
#define CLEAR_OFFSET1 ivec2(0)
#endif
#ifndef CLEAR_OFFSET2
#define CLEAR_OFFSET2 ivec2(0)
#endif
#ifndef CLEAR_OFFSET3
#define CLEAR_OFFSET3 ivec2(0)
#endif
#ifndef CLEAR_OFFSET4
#define CLEAR_OFFSET4 ivec2(0)
#endif
#ifndef CLEAR_OFFSET5
#define CLEAR_OFFSET5 ivec2(0)
#endif
#ifndef CLEAR_OFFSET6
#define CLEAR_OFFSET6 ivec2(0)
#endif
#ifndef CLEAR_OFFSET7
#define CLEAR_OFFSET7 ivec2(0)
#endif
#ifndef CLEAR_OFFSET8
#define CLEAR_OFFSET8 ivec2(0)
#endif

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    if (all(lessThan(texelPos, CLEAR_IMAGE_SIZE))) {
        #ifdef CLEAR_IMAGE1
        imageStore(CLEAR_IMAGE1, texelPos + CLEAR_OFFSET1, CLEAR_COLOR1);
        #endif
        #ifdef CLEAR_IMAGE2
        imageStore(CLEAR_IMAGE2, texelPos + CLEAR_OFFSET2, CLEAR_COLOR2);
        #endif
        #ifdef CLEAR_IMAGE3
        imageStore(CLEAR_IMAGE3, texelPos + CLEAR_OFFSET3, CLEAR_COLOR3);
        #endif
        #ifdef CLEAR_IMAGE4
        imageStore(CLEAR_IMAGE4, texelPos + CLEAR_OFFSET4, CLEAR_COLOR4);
        #endif
        #ifdef CLEAR_IMAGE5
        imageStore(CLEAR_IMAGE5, texelPos + CLEAR_OFFSET5, CLEAR_COLOR5);
        #endif
        #ifdef CLEAR_IMAGE6
        imageStore(CLEAR_IMAGE6, texelPos + CLEAR_OFFSET6, CLEAR_COLOR6);
        #endif
        #ifdef CLEAR_IMAGE7
        imageStore(CLEAR_IMAGE7, texelPos + CLEAR_OFFSET7, CLEAR_COLOR7);
        #endif
        #ifdef CLEAR_IMAGE8
        imageStore(CLEAR_IMAGE8, texelPos + CLEAR_OFFSET8, CLEAR_COLOR8);
        #endif
    }
}
/*const*/