#define GLOBAL_DATA_MODIFIER buffer

#include "/util/Math.glsl"

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
const ivec3 workGroups = ivec3(1, 1, 1);

const uint BEND_SSS_PARAMS_BASE = 128u;

void main() {
    // Single-thread setup pass for deferred25 indirect dispatch.
    vec4 lightClip = global_camProj * vec4(uval_shadowLightDirView, 0.0);
    float lightW = lightClip.w;
    float fpLimit = 0.000128;
    if (lightW >= 0.0 && lightW < fpLimit) {
        lightW = fpLimit;
    } else if (lightW < 0.0 && lightW > -fpLimit) {
        lightW = -fpLimit;
    }

    vec2 lightNdc = lightClip.xy / lightW;
    vec2 lightPixel = vec2(
        (lightNdc.x * 0.5 + 0.5) * uval_mainImageSize.x,
        (lightNdc.y * -0.5 + 0.5) * uval_mainImageSize.y
    );

    // Bend Studio SSS Dispatch Logic
    float xy_light_w = lightClip.w;
    if (xy_light_w >= 0.0 && xy_light_w < fpLimit) xy_light_w = fpLimit;
    else if (xy_light_w < 0.0 && xy_light_w > -fpLimit) xy_light_w = -fpLimit;

    vec4 lightCoordInternal;
    lightCoordInternal.x = ((lightClip.x / xy_light_w) * 0.5 + 0.5) * uval_mainImageSize.x;
    lightCoordInternal.y = ((lightClip.y / xy_light_w) * 0.5 + 0.5) * uval_mainImageSize.y;
    lightCoordInternal.z = lightClip.w == 0.0 ? 0.0 : (lightClip.z / lightClip.w);
    lightCoordInternal.w = lightClip.w > 0.0 ? 1.0 : -1.0;

    ivec2 lightXY = ivec2(round(lightCoordInternal.xy));
    ivec2 minRenderBounds = ivec2(0);
    ivec2 maxRenderBounds = uval_mainImageSizeI;

    ivec4 biased_bounds = ivec4(
        minRenderBounds.x - lightXY.x,
        -(maxRenderBounds.y - lightXY.y),
        maxRenderBounds.x - lightXY.x,
        -(minRenderBounds.y - lightXY.y)
    );

    uint dispatchCount = 0u;
    uint totalGroups = 0u;

    // We store dispatches at BEND_SSS_PARAMS_BASE + 8
    // Format per dispatch: [WaveCountY, WaveCountZ, WaveOffsetX, WaveOffsetY, GroupStart, GroupCount]
    uint dataPtr = BEND_SSS_PARAMS_BASE + 8u;

    int waveSize = 64;

    for (int q = 0; q < 4; q++) {
        bool vertical = (q == 0 || q == 3);

        int b0 = max(0, ((q & 1) != 0 ? biased_bounds[0] : -biased_bounds[2])) / waveSize;
        int b1 = max(0, ((q & 2) != 0 ? biased_bounds[1] : -biased_bounds[3])) / waveSize;
        int b2 = max(0, (((q & 1) != 0 ? biased_bounds[2] : -biased_bounds[0]) + waveSize * (vertical ? 1 : 2) - 1)) / waveSize;
        int b3 = max(0, (((q & 2) != 0 ? biased_bounds[3] : -biased_bounds[1]) + waveSize * (vertical ? 2 : 1) - 1)) / waveSize;

        if ((b2 - b0) > 0 && (b3 - b1) > 0) {
            int bias_x = (q == 2 || q == 3) ? 1 : 0;
            int bias_y = (q == 1 || q == 3) ? 1 : 0;

            int wcY = b2 - b0;
            int wcZ = b3 - b1;
            int woX = ((q & 1) != 0 ? b0 : -b2) + bias_x;
            int woY = ((q & 2) != 0 ? -b3 : b1) + bias_y;

            // Split logic
            int axis_delta = biased_bounds[0] - biased_bounds[1];
            if (q == 1) axis_delta = biased_bounds[2] + biased_bounds[1];
            if (q == 2) axis_delta = -biased_bounds[0] - biased_bounds[3];
            if (q == 3) axis_delta = -biased_bounds[2] + biased_bounds[3];

            axis_delta = (axis_delta + waveSize - 1) / waveSize;

            // Primary dispatch
            bool hasSplit = axis_delta > 0;

            // Should split?
            if (hasSplit) {
                 int wcY2 = wcY;
                 int wcZ2 = wcZ;
                 int woX2 = woX;
                 int woY2 = woY;

                 // Modify original (disp) and split (disp2)
                 if (q == 0) {
                     wcZ2 = min(wcZ, axis_delta);
                     wcZ -= wcZ2;
                     woY2 = woY + wcZ;
                     woX2--;
                     wcY2++;
                 } else if (q == 1) {
                     wcY2 = min(wcY, axis_delta);
                     wcY -= wcY2;
                     woX2 = woX + wcY;
                     wcZ2++;
                 } else if (q == 2) {
                     wcY2 = min(wcY, axis_delta);
                     wcY -= wcY2;
                     woX += wcY2;
                     wcZ2++;
                     woY2--;
                 } else if (q == 3) {
                     wcZ2 = min(wcZ, axis_delta);
                     wcZ -= wcZ2;
                     woY += wcZ2;
                     wcY2++;
                 }

                 // Add split dispatch if valid
                 if (wcY2 > 0 && wcZ2 > 0) {
                      uint gCount = uint(wcY2 * wcZ2 * waveSize);
                      indirectComputeData[dataPtr + 0u] = uint(wcY2);
                      indirectComputeData[dataPtr + 1u] = uint(wcZ2);
                      indirectComputeData[dataPtr + 2u] = uint(woX2 * waveSize);
                      indirectComputeData[dataPtr + 3u] = uint(woY2 * waveSize);
                      indirectComputeData[dataPtr + 4u] = totalGroups;
                      indirectComputeData[dataPtr + 5u] = gCount;
                      totalGroups += gCount;
                      dispatchCount++;
                      dataPtr += 6u;
                 }
            }

            // Add remaining original dispatch if valid
            if (wcY > 0 && wcZ > 0) {
                 uint gCount = uint(wcY * wcZ * waveSize);
                 indirectComputeData[dataPtr + 0u] = uint(wcY);
                 indirectComputeData[dataPtr + 1u] = uint(wcZ);
                 indirectComputeData[dataPtr + 2u] = uint(woX * waveSize);
                 indirectComputeData[dataPtr + 3u] = uint(woY * waveSize);
                 indirectComputeData[dataPtr + 4u] = totalGroups;
                 indirectComputeData[dataPtr + 5u] = gCount;
                 totalGroups += gCount;
                 dispatchCount++;
                 dataPtr += 6u;
            }
        }
    }

    uint limitX = 65535u;
    uint groupsX = totalGroups > limitX ? limitX : totalGroups;
    uint groupsY = (totalGroups + limitX - 1u) / limitX;
    if (totalGroups == 0u) { groupsX = 0u; groupsY = 0u; }

    global_dispatchSize3 = uvec4(groupsX, groupsY, 1u, 0u);

    indirectComputeData[BEND_SSS_PARAMS_BASE + 0u] = floatBitsToUint(lightCoordInternal.x);
    indirectComputeData[BEND_SSS_PARAMS_BASE + 1u] = floatBitsToUint(lightCoordInternal.y);
    indirectComputeData[BEND_SSS_PARAMS_BASE + 2u] = floatBitsToUint(lightCoordInternal.z);
    indirectComputeData[BEND_SSS_PARAMS_BASE + 3u] = floatBitsToUint(lightCoordInternal.w);
    indirectComputeData[BEND_SSS_PARAMS_BASE + 4u] = dispatchCount;
}
