/*
    References:
        [INT17] Intel Corporation. "Outdoor Light Scattering Sample". 2017.
            Apache License 2.0. Copyright (c) 2017 Intel Corporation.
            https://github.com/GameTechDev/OutdoorLightScattering

        You can find full license texts in /licenses
*/
#include "Common.glsl"

bool unwarpEpipolarInsctrImage(
int layerIndex,
vec2 ndcPos,
vec2 screenLayerViewZ,
out ScatteringResult result
){
    // Compute direction of the ray going from the light through the pixel
    vec2 f2RayDir = normalize(ndcPos - uval_sunNdcPos);

    // Find, which boundary the ray intersects. For this, we will
    // find which two of four half spaces the f2RayDir belongs to
    // Each of four half spaces is produced by the line connecting one of four
    // screen corners and the current pixel:
    //    ________________        _______'________           ________________
    //   |'            . '|      |      '         |         |                |
    //   | '       . '    |      |     '          |      .  |                |
    //   |  '  . '        |      |    '           |        '|.        hs1    |
    //   |   *.           |      |   *     hs0    |         |  '*.           |
    //   |  '   ' .       |      |  '             |         |      ' .       |
    //   | '        ' .   |      | '              |         |          ' .   |
    //   |'____________ '_|      |'_______________|         | ____________ '_.
    //                           '                                             '
    //                           ________________  .        '________________
    //                           |             . '|         |'               |
    //                           |   hs2   . '    |         | '              |
    //                           |     . '        |         |  '             |
    //                           | . *            |         |   *            |
    //                         . '                |         |    '           |
    //                           |                |         | hs3 '          |
    //                           |________________|         |______'_________|
    //                                                              '
    // The equations for the half spaces are the following:
    //bool hs0 = (screenPos.x - (-1)) * f2RayDir.y < f2RayDir.x * (screenPos.y - (-1));
    //bool hs1 = (screenPos.x -  (1)) * f2RayDir.y < f2RayDir.x * (screenPos.y - (-1));
    //bool hs2 = (screenPos.x -  (1)) * f2RayDir.y < f2RayDir.x * (screenPos.y -  (1));
    //bool hs3 = (screenPos.x - (-1)) * f2RayDir.y < f2RayDir.x * (screenPos.y -  (1));
    // Note that in fact the outermost visible screen pixels do not lie exactly on the boundary (+1 or -1), but are biased by
    // 0.5 screen pixel size inwards. Using these adjusted boundaries improves precision and results in
    // smaller number of pixels which require inscattering correction
    vec4 f4Boundaries = getOutermostScreenPixelCoords();//left, bottom, right, top
    vec4 f4HalfSpaceEquationTerms = (ndcPos.xxyy - f4Boundaries.xzyw) * f2RayDir.yyxx;
    uvec4 b4HalfSpaceFlags = uvec4(lessThan(f4HalfSpaceEquationTerms.xyyx, f4HalfSpaceEquationTerms.zzww));

    // Now compute mask indicating which of four sectors the f2RayDir belongs to and consiquently
    // which border the ray intersects:
    //    ________________
    //   |'            . '|         0 : hs3 && !hs0
    //   | '   3   . '    |         1 : hs0 && !hs1
    //   |  '  . '        |         2 : hs1 && !hs2
    //   |0  *.       2   |         3 : hs2 && !hs3
    //   |  '   ' .       |
    //   | '   1    ' .   |
    //   |'____________ '_|
    //
    uvec4 b4SectorFlags = b4HalfSpaceFlags.wxyz & (1u - b4HalfSpaceFlags.xyzw);
    // Note that b4SectorFlags now contains true (1) for the exit boundary and false (0) for 3 other

    // Compute distances to boundaries according to following lines:
    //float fDistToLeftBoundary   = abs(f2RayDir.x) > 1e-5 ? ( -1 - uval_sunNdcPos.x) / f2RayDir.x : -FLT_MAX;
    //float fDistToBottomBoundary = abs(f2RayDir.y) > 1e-5 ? ( -1 - uval_sunNdcPos.y) / f2RayDir.y : -FLT_MAX;
    //float fDistToRightBoundary  = abs(f2RayDir.x) > 1e-5 ? (  1 - uval_sunNdcPos.x) / f2RayDir.x : -FLT_MAX;
    //float fDistToTopBoundary    = abs(f2RayDir.y) > 1e-5 ? (  1 - uval_sunNdcPos.y) / f2RayDir.y : -FLT_MAX;
    vec4 f4DistToBoundaries = (f4Boundaries - uval_sunNdcPos.xyxy) / (f2RayDir.xyxy + vec4(lessThan(abs(f2RayDir.xyxy), vec4(1e-6))));
    // Select distance to the exit boundary:
    float fDistToExitBoundary = dot(vec4(b4SectorFlags), f4DistToBoundaries);
    // Compute exit point on the boundary:
    vec2 f2ExitPoint = uval_sunNdcPos + f2RayDir * fDistToExitBoundary;

    // Compute epipolar slice for each boundary:
    //if( LeftBoundary )
    //    fEpipolarSlice = 0.0  - (LeftBoudaryIntersecPoint.y   -   1 )/2 /4;
    //else if( BottomBoundary )
    //    fEpipolarSlice = 0.25 + (BottomBoudaryIntersecPoint.x - (-1))/2 /4;
    //else if( RightBoundary )
    //    fEpipolarSlice = 0.5  + (RightBoudaryIntersecPoint.y  - (-1))/2 /4;
    //else if( TopBoundary )
    //    fEpipolarSlice = 0.75 - (TopBoudaryIntersecPoint.x      - 1 )/2 /4;
    vec4 f4EpipolarSlice = vec4(0.0, 0.25, 0.5, 0.75) +
    saturate((f2ExitPoint.yxyx - f4Boundaries.wxyz) * vec4(-1.0, 1.0, 1.0, -1.0) / (f4Boundaries.wzwz - f4Boundaries.yxyx)) / 4.0;
    // Select the right value:
    float fEpipolarSlice = dot(vec4(b4SectorFlags), f4EpipolarSlice);

    // Now find two closest epipolar slices, from which we will interpolate
    // First, find index of the slice which precedes our slice
    // Note that 0 <= fEpipolarSlice <= 1, and both 0 and 1 refer to the first slice
    float fPrecedingSliceInd = floor(fEpipolarSlice * SETTING_EPIPOLAR_SLICES);

    // Compute EXACT texture coordinates of preceding and succeeding slices and their weights
    // Note that slice 0 is stored in the first texel which has exact texture coordinate 0.5/SETTING_EPIPOLAR_SLICES
    // (search for "fEpipolarSlice = saturate(f2UV.x - 0.5f / (float)SETTING_EPIPOLAR_SLICES)"):
    float fSrcSliceTexelX[2];
    fSrcSliceTexelX[0] = fPrecedingSliceInd + 0.5;
    fSrcSliceTexelX[1] = mod(fPrecedingSliceInd + 1.5, float(SETTING_EPIPOLAR_SLICES));

    // Compute slice weights
    float fSliceWeights[2];
    fSliceWeights[1] = (fEpipolarSlice * SETTING_EPIPOLAR_SLICES) - fPrecedingSliceInd;
    fSliceWeights[0] = 1 - fSliceWeights[1];

    result.inScattering = vec3(0.0);
    result.transmittance = vec3(0.0);
    float totalWeight = 0;

    // Add 1 to account for the endpoint data stored in the first texel row
    #define TEXEL_Y_OFFSET (1 + layerIndex * SETTING_SLICE_SAMPLES)

    for (int i = 0; i < 2; ++i) {
        // Load epipolar line endpoints
        uvec4 endPointData = texelFetch(usam_epipolarData, ivec2(int(fSrcSliceTexelX[i]), 0), 0);
        vec4 f4SliceEndpoints = uintBitsToFloat(endPointData);

        // Compute line direction on the screen
        vec2 f2SliceDir = f4SliceEndpoints.zw - f4SliceEndpoints.xy;
        float fSliceLenSqr = dot(f2SliceDir, f2SliceDir);

        // Project current pixel onto the epipolar line
        float fSamplePosOnLine = dot((ndcPos - f4SliceEndpoints.xy), f2SliceDir) / max(fSliceLenSqr, 1e-8);
        fSamplePosOnLine = saturate(fSamplePosOnLine);
        // Compute index of the slice on the line
        // Note that the first sample on the line (fSamplePosOnLine==0) is exactly the Entry Point, while
        // the last sample (fSamplePosOnLine==1) is exactly the Exit Point
        // (search for "fSamplePosOnEpipolarLine *= (float)SETTING_SLICE_SAMPLES / ((float)SETTING_SLICE_SAMPLES-1.f)")
        float fSampleInd = fSamplePosOnLine * float(SETTING_SLICE_SAMPLES - 1);

        // We have to manually perform bilateral filtering of the scattered radiance texture to
        // eliminate artifacts at depth discontinuities

        float fPrecedingSampleInd = floor(fSampleInd);
        // Get bilinear filtering weight
        float fUWeight = fSampleInd - fPrecedingSampleInd;
        // Get texture coordinate of the left source texel. Again, offset by 0.5 is essential
        // to align with the texel center
        float fPrecedingSampleTexelY = fPrecedingSampleInd + 0.5;

        vec2 epipolarDataTexelF = vec2(fSrcSliceTexelX[i], fPrecedingSampleInd + 0.5);

        // Gather 4 camera space z values
        // Note that we need to bias f2SctrColorUV by 0.5 texel size to refer the location between all four texels and
        // get the required values for sure
        // The values in vec4, which Gather() returns are arranged as follows:
        //   _______ _______
        //  |       |       |
        //  |   x   |   y   |
        //  |_______o_______|  o gather location
        //  |       |       |
        //  |   *w  |   z   |  * f2SctrColorUV
        //  |_______|_______|
        //  |<----->|
        //     1/f2ScatteredColorTexDim.x

        // x == g_tex2DEpipolarCamSpaceZ.SampleLevel(samPointClamp, f2SctrColorUV, 0, int2(0,1))
        // y == g_tex2DEpipolarCamSpaceZ.SampleLevel(samPointClamp, f2SctrColorUV, 0, int2(1,1))
        // z == g_tex2DEpipolarCamSpaceZ.SampleLevel(samPointClamp, f2SctrColorUV, 0, int2(1,0))
        // w == g_tex2DEpipolarCamSpaceZ.SampleLevel(samPointClamp, f2SctrColorUV, 0, int2(0,0))

        const vec2 EPIPOLAR_DATA_TEX_SIZE = vec2(SETTING_EPIPOLAR_SLICES, EPIPOLAR_DATA_Y_SIZE);
        const vec2 LAYER_SIZE = vec2(SETTING_EPIPOLAR_SLICES, SETTING_SLICE_SAMPLES);


        ivec2 epipolarDataTexelI = ivec2(epipolarDataTexelF);
        ivec2 sampleTexelPos1 = epipolarDataTexelI;
        sampleTexelPos1.y += int(TEXEL_Y_OFFSET);
        ivec2 sampleTexelPos2 = epipolarDataTexelI + ivec2(0, 1);
        sampleTexelPos2.y = min(sampleTexelPos2.y, SETTING_SLICE_SAMPLES - 1);
        sampleTexelPos2.y += int(TEXEL_Y_OFFSET);

        uint viewZData1 = texelFetch(usam_epipolarData, sampleTexelPos1, 0).a;
        uint viewZData2 = texelFetch(usam_epipolarData, sampleTexelPos2, 0).a;
        uvec2 viewZData = uvec2(viewZData1, viewZData2);
        uvec2 viewZTexelPosXs = bitfieldExtract(viewZData, 0, 16);
        uvec2 viewZTexelPosYs = bitfieldExtract(viewZData, 16, 16);
        ivec2 viewZTexelPos1 = ivec2(uvec2(viewZTexelPosXs.x, viewZTexelPosYs.x));
        ivec2 viewZTexelPos2 = ivec2(uvec2(viewZTexelPosXs.y, viewZTexelPosYs.y));
        vec2 viewZ1;
        vec2 viewZ2;
        if (layerIndex == 0) {
            viewZ1 = uintBitsToFloat(transient_translucentZLayer1_fetch(viewZTexelPos1).xy);
            viewZ1 = uintBitsToFloat(transient_translucentZLayer1_fetch(viewZTexelPos2).xy);
        } else if (layerIndex == 1) {
            viewZ1 = uintBitsToFloat(transient_translucentZLayer2_fetch(viewZTexelPos1).xy);
            viewZ2 = uintBitsToFloat(transient_translucentZLayer2_fetch(viewZTexelPos2).xy);
        } else {
            viewZ1 = uintBitsToFloat(transient_translucentZLayer3_fetch(viewZTexelPos1).xy);
            viewZ2 = uintBitsToFloat(transient_translucentZLayer3_fetch(viewZTexelPos2).xy);
        }
        viewZ1 = -abs(viewZ1);
        viewZ2 = -abs(viewZ2);

        vec2 startZs = viewZData1 == 0xFFFFFFFFu ? vec2(65536.0) : vec2(viewZ1.x, viewZ2.x);
        vec2 endZs = viewZData2 == 0xFFFFFFFFu ? vec2(65536.0) : vec2(viewZ1.y, viewZ2.y);

        const float refinementThreshold = 0.03;
        vec2 f2DepthWeights = vec2(1.0);
        {
            // Compute depth weights in a way that if the difference is less than the threshold, the weight is 1 and
            // the weights fade out to 0 as the difference becomes larger than the threshold:
            vec2 f2MaxZ = max(-startZs, max(-screenLayerViewZ.x, 1.0));
            f2DepthWeights *= saturate(refinementThreshold / max(abs(startZs - screenLayerViewZ.x) / f2MaxZ, refinementThreshold));
            // Note that if the sample is located outside the [-1,1]x[-1,1] area, the sample is invalid and fCurrCamSpaceZ == fInvalidCoordinate
            // Depth weight computed for such sample will be zero
        }
        {
            vec2 f2MaxZ = max(-endZs, max(-screenLayerViewZ.y, 1.0));
            f2DepthWeights *= saturate(refinementThreshold / max(abs(endZs - screenLayerViewZ.y) / f2MaxZ, refinementThreshold));
        }

        f2DepthWeights = pow4(f2DepthWeights);

        // Multiply bilinear weights with the depth weights:
        vec2 bilateralVWeights = vec2(1 - fUWeight, fUWeight) * f2DepthWeights * fSliceWeights[i];
        // If the sample projection is behind [0,1], we have to discard this slice
        // We however must take into account the fact that if at least one sample from the two
        // bilinear sources is correct, the sample can still be properly computed
        //
        //            -1       0       1                  N-2     N-1      N              Sample index
        // |   X   |   X   |   X   |   X   |  ......   |   X   |   X   |   X   |   X   |
        //         1-1/(N-1)   0    1/(N-1)                        1   1+1/(N-1)          fSamplePosOnLine
        //             |                                                   |
        //             |<-------------------Clamp range------------------->|
        //
        bilateralVWeights *= float(abs(fSamplePosOnLine - 0.5) < (0.5 + 1.0 / (SETTING_SLICE_SAMPLES - 1)));

        {
            ScatteringResult sampleResult;
            float viewZ;
            unpackEpipolarData(texelFetch(usam_epipolarData, sampleTexelPos1, 0), sampleResult, viewZ);

            result.inScattering += bilateralVWeights.x * sampleResult.inScattering;
            result.transmittance += bilateralVWeights.x * sampleResult.transmittance;
        }

        {
            ScatteringResult sampleResult;
            float viewZ;
            unpackEpipolarData(texelFetch(usam_epipolarData, sampleTexelPos2, 0), sampleResult, viewZ);

            result.inScattering += bilateralVWeights.y * sampleResult.inScattering;
            result.transmittance += bilateralVWeights.y * sampleResult.transmittance;
        }

        // Update total weight
        totalWeight += bilateralVWeights.x + bilateralVWeights.y;
    }

    result.inScattering /= totalWeight;
    result.transmittance /= totalWeight;
    return totalWeight >= 1e-2;
}