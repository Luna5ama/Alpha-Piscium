#include "/Base.glsl"

/*const*/
void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    if (all(lessThan(texelPos, CLEAR_IMAGE_SIZE))) {
        #ifdef CLEAR_1
        CLEAR_1(texelPos);
        #endif
        #ifdef CLEAR_2
        CLEAR_2(texelPos);
        #endif
        #ifdef CLEAR_3
        CLEAR_3(texelPos);
        #endif
        #ifdef CLEAR_4
        CLEAR_4(texelPos);
        #endif
        #ifdef CLEAR_5
        CLEAR_5(texelPos);
        #endif
        #ifdef CLEAR_6
        CLEAR_6(texelPos);
        #endif
        #ifdef CLEAR_7
        CLEAR_7(texelPos);
        #endif
        #ifdef CLEAR_8
        CLEAR_8(texelPos);
        #endif
    }
}
/*const*/