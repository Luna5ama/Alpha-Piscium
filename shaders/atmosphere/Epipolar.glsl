#if SETTING_EPIPOLAR_SLICES == 256

#define EPIPOLAR_SLICE_D16 16
#define EPIPOLAR_SLICE_D128 2

#elif SETTING_EPIPOLAR_SLICES == 512

#define EPIPOLAR_SLICE_D16 32
#define EPIPOLAR_SLICE_D128 4

#elif SETTING_EPIPOLAR_SLICES == 1024

#define EPIPOLAR_SLICE_D16 64
#define EPIPOLAR_SLICE_D128 8

#elif SETTING_EPIPOLAR_SLICES == 2048

#define EPIPOLAR_SLICE_D16 128
#define EPIPOLAR_SLICE_D128 16

#endif

#if SETTING_SLICE_SAMPLES == 128

#define SLICE_SAMPLE_D16 8

#elif SETTING_SLICE_SAMPLES == 256

#define SLICE_SAMPLE_D16 16

#elif SETTING_SLICE_SAMPLES == 512

#define SLICE_SAMPLE_D16 32

#elif SETTING_SLICE_SAMPLES == 1024

#define SLICE_SAMPLE_D16 64

#endif


#define INVALID_EPIPOLAR_LINE vec4(-1000.0, -1000.0, -100.0, -100.0)

bool isValidScreenLocation(vec2 f2XY) {
    const float SAFETY_EPSILON = 0.2f;
    return all(lessThanEqual(abs(f2XY), 1.0 - (1.0 - SAFETY_EPSILON) / vec2(global_mainImageSizeI)));
}

vec4 getOutermostScreenPixelCoords() {
    // The outermost visible screen pixels centers do not lie exactly on the boundary (+1 or -1), but are biased by
    // 0.5 screen pixel size inwards
    //
    //                                        2.0
    //    |<---------------------------------------------------------------------->|
    //
    //       2.0/Res
    //    |<--------->|
    //    |     X     |      X     |     X     |    ...    |     X     |     X     |
    //   -1     |                                                            |    +1
    //          |                                                            |
    //          |                                                            |
    //      -1 + 1.0/Res                                                  +1 - 1.0/Res
    //
    // Using shader macro is much more efficient than using constant buffer variable
    // because the compiler is able to optimize the code more aggressively
    return vec4(-1.0, -1.0, 1.0, 1.0) + vec4(1.0, 1.0, -1.0, -1.0) / global_mainImageSizeI.xyxy;
}
