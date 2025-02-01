uvec2 ndpacking_pack(vec3 normal, float depth) {
    uvec2 packedData;
    packedData.x = packSnorm2x16(coords_octEncode11(normal));
    packedData.y = floatBitsToUint(depth);
    return packedData;
}

void ndpacking_unpack(uvec2 packedData, out vec3 normal, out float depth) {
    normal = coords_octDecode11(unpackSnorm2x16(packedData.x));
    depth = uintBitsToFloat(packedData.y);
}

void ndpacking_updateProjReject(usampler2D lastNZTex, ivec2 texelCoord, vec2 screenCoord, vec3 currN, vec3 currView, out vec2 projReject) {
    vec3 cameraDelta = cameraPosition - previousCameraPosition;
    vec4 currScene = gbufferModelViewInverse * vec4(currView, 1.0);
    vec4 curr2PrevScene = coord_sceneCurrToPrev(currScene);
    vec4 curr2PrevView = gbufferPreviousModelView * curr2PrevScene;
    vec4 curr2PrevClip = gbufferProjection * curr2PrevView;

    {
        float prevZ;
        vec3 prevN;
        ndpacking_unpack(texelFetch(lastNZTex, texelCoord, 0).xy, prevN, prevZ);

        vec3 prevView = coords_toViewCoord(screenCoord, prevZ, gbufferPreviousProjectionInverse);
        vec4 prevScene = gbufferPreviousModelViewInverse * vec4(prevView, 1.0);
        vec4 prev2CurrScene = coord_scenePrevToCurr(prevScene);
        vec4 prev2CurrClip = gbufferPreviousProjection * gbufferModelView * prev2CurrScene;

        uint flag = 0u;
        flag |= uint(currView.z != 1.0) & uint(any(greaterThanEqual(abs(curr2PrevClip.xyz), curr2PrevClip.www)));
        flag |= uint(prevZ != 0.0) & uint(any(greaterThanEqual(abs(prev2CurrClip.xyz), prev2CurrClip.www)));

        projReject.x = float(flag);
    }

    {
        vec4 prevZs = uintBitsToFloat(textureGather(lastNZTex, screenCoord, 1));

        vec3 diff;
        float dotV = 0.0;
        diff = coords_toViewCoord(screenCoord, prevZs.x, gbufferPreviousProjectionInverse) - curr2PrevView.xyz;
        dotV += dot(currN, diff);
        diff = coords_toViewCoord(screenCoord, prevZs.y, gbufferPreviousProjectionInverse) - curr2PrevView.xyz;
        dotV += dot(currN, diff);
        diff = coords_toViewCoord(screenCoord, prevZs.z, gbufferPreviousProjectionInverse) - curr2PrevView.xyz;
        dotV += dot(currN, diff);
        diff = coords_toViewCoord(screenCoord, prevZs.w, gbufferPreviousProjectionInverse) - curr2PrevView.xyz;
        dotV += dot(currN, diff);
        dotV = step(3.0, abs(dotV));

        projReject.y = dotV;
    }
}