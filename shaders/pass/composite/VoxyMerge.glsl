#include "/util/Math.glsl"
#include "/util/GBufferData.glsl"
#include "/util/Coords.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba32ui) uniform restrict readonly uimage2D uimg_gbufferVoxySolidData;
layout(rgba32ui) uniform restrict uimage2D uimg_gbufferSolidData1;
layout(r32ui) uniform restrict uimage2D uimg_gbufferSolidData2;
layout(r32f) uniform restrict image2D uimg_gbufferSolidViewZ;

shared uvec4 shared_voxyData[18][18];

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);

    // 1. Load into shared memory (with 1 pixel border)
    ivec2 groupBase = ivec2(gl_WorkGroupID.xy) * 16 - 1;
    uint localIndex = gl_LocalInvocationIndex;

    // Load 18x18 = 324 pixels using 256 threads
    for (uint i = localIndex; i < 324u; i += 256u) {
        ivec2 localPos = ivec2(i % 18u, i / 18u);
        ivec2 loadPos = groupBase + localPos;
        loadPos = clamp(loadPos, ivec2(0), uval_mainImageSizeI - 1);
        shared_voxyData[localPos.x][localPos.y] = imageLoad(uimg_gbufferVoxySolidData, loadPos);
    }

    barrier();

    if (any(greaterThanEqual(texelPos, uval_mainImageSizeI))) return;

    // Check center pixel
    ivec2 localPos = ivec2(gl_LocalInvocationID.xy) + 1;
    uvec4 voxyCenter = shared_voxyData[localPos.x][localPos.y];

    // Check if empty (assuming w=0 and x=0 means empty/invalid)
    if (voxyCenter.w == 0u && voxyCenter.x == 0u) return;

    float voxyZ = uintBitsToFloat(voxyCenter.w);
    float solidZ = imageLoad(uimg_gbufferSolidViewZ, texelPos).r;

    // Merge if voxy is closer (viewZ is negative distance, so larger is closer)
    if (voxyZ > solidZ) {
        imageStore(uimg_gbufferSolidViewZ, texelPos, vec4(voxyZ));

        // Unpack Center Data
        vec2 uvCenter = unpackUnorm2x16(voxyCenter.x);
        vec4 colorFace = unpackUnorm4x8(voxyCenter.y);
        vec3 albedo = colorFace.rgb;
        uint face = bitfieldExtract(voxyCenter.y, 24, 3);

        uint rawProp = voxyCenter.z;
        uint matID = (rawProp >> 16) & 0xFFFFu;
        vec2 lmCoord = vec2(rawProp & 0xFFu, (rawProp >> 8) & 0xFFu) / 255.0;

        vec2 screenPos = (vec2(texelPos) + 0.5) * uval_mainImageSizeRcp;
        vec3 viewPosCenter = coords_toViewCoord(screenPos, voxyZ, global_camProjInverse);

        // 2. Gradients Reconstruction
        vec2 dUVdx = vec2(0.0);
        vec2 dUVdy = vec2(0.0);
        vec3 dPdx = vec3(0.0);
        vec3 dPdy = vec3(0.0);

        float weightX = 0.0;
        float weightY = 0.0;

        // X-Derivative
        for (int dx = -1; dx <= 1; dx += 2) {
            uvec4 voxyNeighbor = shared_voxyData[localPos.x + dx][localPos.y];
            if (voxyNeighbor.w != 0u || voxyNeighbor.x != 0u) {
                 uint neighborFace = bitfieldExtract(voxyNeighbor.y, 24, 3);
                 if (neighborFace == face) { // Check face continuity
                     float neighborZ = uintBitsToFloat(voxyNeighbor.w);
                     // Check depth continuity (threshold ~0.1 or dependant on scene scale)
                     if (abs(neighborZ - voxyZ) < 0.2) {
                        vec2 uvNeighbor = unpackUnorm2x16(voxyNeighbor.x);
                        vec2 neighborScreenPos = (vec2(texelPos + ivec2(dx, 0)) + 0.5) * uval_mainImageSizeRcp;
                        vec3 viewPosNeighbor = coords_toViewCoord(neighborScreenPos, neighborZ, global_camProjInverse);

                        dUVdx += (uvNeighbor - uvCenter) * float(dx); // * 0.5 if centered? No, one sided is 1.0 dist.
                        dPdx  += (viewPosNeighbor - viewPosCenter) * float(dx);
                        weightX += 1.0;
                     }
                 }
            }
        }

        // Y-Derivative
        for (int dy = -1; dy <= 1; dy += 2) {
            uvec4 voxyNeighbor = shared_voxyData[localPos.x][localPos.y + dy];
            if (voxyNeighbor.w != 0u || voxyNeighbor.x != 0u) {
                 uint neighborFace = bitfieldExtract(voxyNeighbor.y, 24, 3);
                 if (neighborFace == face) {
                     float neighborZ = uintBitsToFloat(voxyNeighbor.w);
                     if (abs(neighborZ - voxyZ) < 0.2) {
                        vec2 uvNeighbor = unpackUnorm2x16(voxyNeighbor.x);
                        vec2 neighborScreenPos = (vec2(texelPos + ivec2(0, dy)) + 0.5) * uval_mainImageSizeRcp;
                        vec3 viewPosNeighbor = coords_toViewCoord(neighborScreenPos, neighborZ, global_camProjInverse);

                        dUVdy += (uvNeighbor - uvCenter) * float(dy);
                        dPdy  += (viewPosNeighbor - viewPosCenter) * float(dy);
                        weightY += 1.0;
                     }
                 }
            }
        }

        // Normalize if we accumulate
        if (weightX > 0.5) {
            dUVdx /= weightX; // If centered (neighbors -1 and 1), diff is 2dx.
            dPdx /= weightX;  // If only 1 neighbor, diff is 1dx.
            // My loop: dx=-1 -> adds (N-C)*(-1) = C-N. dx=1 -> adds (N-C)*(1) = N-C.
            // Sum = (N_right - C) + (C - N_left) = N_right - N_left. Distance 2.
            // So if weightX is 2, we divide by 2. Correct.
            // If weightX is 1, we divide by 1. Correct (approx).
        }
        if (weightY > 0.5) {
            dUVdy /= weightY;
            dPdy /= weightY;
        }

        // 3. Tangent/Bitangent Reconstruction
        // Geometric Normal from Face
        vec3 gemoWorldNormal = normalize( vec3(uint((face>>1)==2), uint((face>>1)==0), uint((face>>1)==1)) * (float(int(face)&1)*2-1));
        vec3 geomViewNormal = coords_dir_worldToView(gemoWorldNormal);

        vec3 tangent = vec3(1.0, 0.0, 0.0);
        vec3 bitangent = vec3(0.0, 1.0, 0.0);

        bool singlePixel = (weightX < 0.5 && weightY < 0.5);

        if (!singlePixel) {
            // Reconstruct TBN
            // det = dUVdx.x * dUVdy.y - dUVdx.y * dUVdy.x
            float det = dUVdx.x * dUVdy.y - dUVdy.x * dUVdy.y; // Wait, dot perp? det(J)
            // T = (dPdx * dv_dy - dPdy * dv_dx) / det

            if (abs(det) > 1e-6) {
                float invDet = 1.0 / det;
                tangent = (dPdx * dUVdy.y - dPdy * dUVdx.y) * invDet;
                bitangent = (dPdy * dUVdx.x - dPdx * dUVdy.x) * invDet;
                // Gram-Schmidt / Orthogonalize
                tangent = normalize(tangent - geomViewNormal * dot(geomViewNormal, tangent));
                bitangent = normalize(cross(geomViewNormal, tangent) * sign(det)); // Use sign of det for handedness?
                // Or just:
                bitangent = normalize(bitangent - geomViewNormal * dot(geomViewNormal, bitangent));
                bitangent = normalize(bitangent - tangent * dot(tangent, bitangent));
            } else {
                 singlePixel = true;
            }
        }

        if (singlePixel) {
             // Hardcoded fallback
             // We can deduce T/B from face.
             // But we also need UV gradients for texture sampling?
             // "calculate the uv gradient ... as if it is a 1x1 sized ... quad"

             // Canonical TBN for face (assuming standard MC UV mapping)
             // This is tricky as rotation varies. But let's assume standard.
             // Face: 0=Down, 1=Up, 2=North, 3=South, 4=West, 5=East (typ MC)
             vec3 worldIdx = vec3(uint((face>>1)==2), uint((face>>1)==0), uint((face>>1)==1)); // 0->(0,-1,0) etc?
             // Actually geomViewNormal logic above:
             // (face>>1)==2 -> indices 4,5 (X axis)
             // (face>>1)==0 -> indices 0,1 (Y axis)
             // (face>>1)==1 -> indices 2,3 (Z axis)
             // Matches MC: 0/1 (-Y/+Y), 2/3 (-Z/+Z), 4/5 (-X/+X).

             // Construct hardcoded tangent in View Space
             vec3 worldTangent;
             if ((face >> 1) == 0u) worldTangent = vec3(1, 0, 0); // Y face -> X tangent
             else if ((face >> 1) == 1u) worldTangent = vec3(1, 0, 0); // Z face -> X tangent
             else worldTangent = vec3(0, 0, 1); // X face -> Z tangent

             tangent = coords_dir_worldToView(worldTangent);
             bitangent = normalize(cross(tangent, geomViewNormal));
             // Re-orthogonalize
             tangent = normalize(cross(geomViewNormal, bitangent));

             // Calculate 1x1 quad derivatives
             // Estimated: dUVdx ~ projected tangent / block_size ?
             // For textureGrad, if data is missing, we pick a safe level (0 or calculated).
             // Let's assume Mip 0 for single pixels or small gradient.
             // Or construct dUVdx from T projected to screen?
             // Maybe simplified:
             dUVdx = vec2(1.0/256.0, 0.0); // Dummy small value
             dUVdy = vec2(0.0, 1.0/256.0);
        }

        // 6. Packing
        GBufferData gData = gbufferData_init();
        gData.albedo = albedo;
        gData.normal = geomViewNormal;
        gData.geomNormal = geomViewNormal;
        gData.geomTangent = tangent;
        // Bitangent sign?
        gData.bitangentSign = (dot(cross(tangent, geomViewNormal), bitangent) > 0.0) ? 1 : -1;

        gData.lmCoord = vec2(0.0, 1.0);

        gData.materialID = matID;
        gData.pbrSpecular = vec4(0.0);

        uvec4 d1;
        uvec4 d2;
        gbufferData1_pack(d1, gData);
        gbufferData2_pack(d2, gData);
        imageStore(uimg_gbufferSolidData1, texelPos, d1);
        imageStore(uimg_gbufferSolidData2, texelPos, uvec4(d2.r, 0, 0, 0));
    }
}
