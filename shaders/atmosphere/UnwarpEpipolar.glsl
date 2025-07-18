/*
    References:
        [INT17] Intel Corporation. "Outdoor Light Scattering Sample". 2017.
            Apache License 2.0. Copyright (c) 2017 Intel Corporation.
            https://github.com/GameTechDev/OutdoorLightScattering

        You can find full license texts in /licenses
*/
#include "Common.glsl"


bool unwarpEpipolarInsctrImage(
vec2 screenPos,
in float screenViewZ,
out ScatteringResult result
){
    // Compute direction of the ray going from the light through the pixel
    vec2 f2RayDir = normalize(screenPos - uval_sunNdcPos);

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
    vec4 f4HalfSpaceEquationTerms = (screenPos.xxyy - f4Boundaries.xzyw) * f2RayDir.yyxx;
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
    float fPrecedingSliceInd = min(floor(fEpipolarSlice * SETTING_EPIPOLAR_SLICES), SETTING_EPIPOLAR_SLICES - 1);

    // Compute EXACT texture coordinates of preceding and succeeding slices and their weights
    // Note that slice 0 is stored in the first texel which has exact texture coordinate 0.5/SETTING_EPIPOLAR_SLICES
    // (search for "fEpipolarSlice = saturate(f2UV.x - 0.5f / (float)SETTING_EPIPOLAR_SLICES)"):
    float fSrcSliceV[2];
    // Compute V coordinate to refer exactly the center of the slice row
    fSrcSliceV[0] = (fPrecedingSliceInd + 0.5) / float(SETTING_EPIPOLAR_SLICES);
    // Use frac() to wrap around to the first slice from the next-to-last slice:
    fSrcSliceV[1] = fract(fSrcSliceV[0] + 1.0 / float(SETTING_EPIPOLAR_SLICES));

    // Compute slice weights
    float fSliceWeights[2];
    fSliceWeights[1] = (fEpipolarSlice * SETTING_EPIPOLAR_SLICES) - fPrecedingSliceInd;
    fSliceWeights[0] = 1 - fSliceWeights[1];

    result.inScattering = vec3(0.0);
    result.transmittance = vec3(0.0);
    float totalWeight = 0;

    for (int i = 0; i < 2; ++i) {
        // Load epipolar line endpoints
        vec4 f4SliceEndpoints = uintBitsToFloat(texture(usam_epipolarData, vec2(fSrcSliceV[i], EPIPOLAR_SLICE_END_POINTS_V)));

        // Compute line direction on the screen
        vec2 f2SliceDir = f4SliceEndpoints.zw - f4SliceEndpoints.xy;
        float fSliceLenSqr = dot(f2SliceDir, f2SliceDir);

        // Project current pixel onto the epipolar line
        float fSamplePosOnLine = dot((screenPos - f4SliceEndpoints.xy), f2SliceDir) / max(fSliceLenSqr, 1e-8);
        fSamplePosOnLine = sqrt(saturate(fSamplePosOnLine));
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
        float fPrecedingSampleU = (fPrecedingSampleInd + 1.5) / float(EPIPOLAR_DATA_Y_SIZE);

        vec2 f2SctrColorUV = vec2(fSrcSliceV[i], fPrecedingSampleU);

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

        const vec2 f2ScatteredColorTexDim = vec2(SETTING_EPIPOLAR_SLICES, EPIPOLAR_DATA_Y_SIZE);
        vec2 epipolarViewZ = uintBitsToFloat(textureGather(usam_epipolarData, f2SctrColorUV + vec2(0.5, 0.5) / f2ScatteredColorTexDim.xy, 3).wx);

        // Compute depth weights in a way that if the difference is less than the threshold, the weight is 1 and
        // the weights fade out to 0 as the difference becomes larger than the threshold:
        vec2 f2MaxZ = max(-epipolarViewZ, max(-screenViewZ, 1.0));
        const float refinementThreshold = 0.03;
        vec2 f2DepthWeights = saturate(refinementThreshold / max(abs(epipolarViewZ - screenViewZ) / f2MaxZ, refinementThreshold));
        // Note that if the sample is located outside the [-1,1]x[-1,1] area, the sample is invalid and fCurrCamSpaceZ == fInvalidCoordinate
        // Depth weight computed for such sample will be zero
        f2DepthWeights = pow(f2DepthWeights, vec2(4.0));

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
            unpackEpipolarData(texture(usam_epipolarData, f2SctrColorUV), sampleResult, viewZ);

            result.inScattering += bilateralVWeights.x * sampleResult.inScattering;
            result.transmittance += bilateralVWeights.x * sampleResult.transmittance;
        }

        {
            ScatteringResult sampleResult;
            float viewZ;
            unpackEpipolarData(textureOffset(usam_epipolarData, f2SctrColorUV, ivec2(0, 1)), sampleResult, viewZ);

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