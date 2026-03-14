#include "/techniques/rtwsm/RTWSM.glsl"
#include "/util/Fresnel.glsl"
#define VOXEL_BRICK_DATA_MODIFIER buffer
#define VOXEL_MATERIAL_DATA_MODIFIER buffer
#include "/techniques/Voxelization.glsl"

layout(r32i) uniform iimage2D uimg_fr32f;

in vec2 mc_Entity;
in vec4 at_midBlock;

#if defined(SHADOW_PASS_ALPHA_TEST) || defined(SHADOW_PASS_TRANSLUCENT)
out vec2 vert_texcoord;
#if defined(SHADOW_PASS_TRANSLUCENT)
out vec3 vert_color;
#endif
#endif

out vec2 vert_screenPos;
out uint vert_worldNormalMaterialID;

out uint vert_survived;

void main() {
    #if defined(SHADOW_PASS_ALPHA_TEST) || defined(SHADOW_PASS_TRANSLUCENT)
    vert_texcoord = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
    #if defined(SHADOW_PASS_TRANSLUCENT)
    vert_color = gl_Color.rgb;
    #endif
    #endif

    vec3 worldNormal = normalize(mat3(shadowModelViewInverse) * (gl_NormalMatrix * gl_Normal.xyz));
    vec2 worldNormalOct = coords_octEncode11(worldNormal);
    uint materialID = uint(int(mc_Entity.x));
    vert_worldNormalMaterialID = packSnorm4x8(vec4(worldNormalOct, 0.0, 0.0));
    vert_worldNormalMaterialID = bitfieldInsert(vert_worldNormalMaterialID, materialID, 16, 16);

    vec4 shadowClipPos = ftransform();
    vec4 shadowViewPos = shadowProjectionInverse * shadowClipPos;
    vec4 scenePos = shadowModelViewInverse * shadowViewPos;
    vec4 shadowNDCPos = global_sceneToShadowNDC * scenePos;
    gl_Position = shadowNDCPos;
    vec2 shadowScreenPosUnwarpped = shadowNDCPos.xy * 0.5 + 0.5;
    vert_screenPos = shadowScreenPosUnwarpped;
    vec2 texelSize;
    gl_Position.xy = rtwsm_warpTexCoordTexelSize(shadowScreenPosUnwarpped, texelSize) * 2.0 - 1.0;

    shadowViewPos /= shadowViewPos.w;
    vec4 camViewPos = gbufferModelView * scenePos;
    camViewPos /= camViewPos.w;

    uint survived = uint(all(lessThan(abs(shadowNDCPos.xyz), shadowNDCPos.www)));
    vert_survived = survived;

    survived &= uint((gl_VertexID & 3) == 0);

    // -------------------------------------------------------------------
    // Voxelization: mark brick occupancy and write material data.
    // Only runs once per quad (gl_VertexID & 3 == 0) to reduce atomics.
    // Skipped for translucent geometry (water handled separately later).
    // -------------------------------------------------------------------
    #ifndef SHADOW_PASS_TRANSLUCENT
    if ((gl_VertexID & 3) == 0 && materialID != MATERIAL_ID_WATER) {
        // Absolute integer block position of the center of this block.
        // scenePos is camera-relative; add camera's integer + fractional parts.
        ivec3 blockWorldPos = ivec3(floor(scenePos.xyz + cameraPositionFract + at_midBlock.xyz / 64.0))
                              + cameraPositionInt;

        // Brick grid coordinate centered on the camera's brick
        ivec3 cameraBrickCoord = cameraPositionInt >> 4;
        ivec3 brickWorldCoord  = blockWorldPos >> 4;
        ivec3 brickRelCoord    = brickWorldCoord - cameraBrickCoord + ivec3(VOXEL_GRID_SIZE / 2);

        if (all(greaterThanEqual(brickRelCoord, ivec3(0))) &&
            all(lessThan(brickRelCoord, ivec3(VOXEL_GRID_SIZE)))) {

            uint brickMorton = voxel_brickMorton(brickRelCoord);

            // Mark brick occupied for this frame
            atomicOr(voxel_brickOccupancy[brickMorton], 1u);

            // Write material ID if the brick already has a valid alloc ID
            // (assigned by last frame's VoxelAllocator begin pass)
            uint allocID = voxel_brickAllocID[brickMorton];
            if (allocID != VOXEL_UNALLOCATED) {
                ivec3 blockInBrick = blockWorldPos & ivec3(VOXEL_BRICK_SIZE - 1);
                uint blockMorton   = voxel_blockMorton(blockInBrick);
                uint matIdx        = voxel_materialIndex(allocID, blockMorton);
                // Only write a non-zero material ID; 0 means "no entity mapping".
                // atomicMax ensures a real ID beats the cleared-to-0 state.
                if (materialID != 0u) {
                    atomicMax(voxel_materials[matIdx], materialID);
                } else {
                    // Block exists but has no material ID mapping: write a
                    // placeholder (1) so the tree knows the voxel is solid.
                    atomicMax(voxel_materials[matIdx], 1u);
                }
            }
        }
    }
    #endif

    #ifdef SETTING_RTWSM_F
    if (bool(survived)){
        ivec2 importanceTexelPos = ivec2(shadowScreenPosUnwarpped * vec2(RTWSM_IMAP_SIZE));

        float importance = SETTING_RTWSM_F_BASE * 16.0;

        float camDistanceSq = dot(camViewPos.xyz, camViewPos.xyz);

        if (isEyeInWater == 1 && materialID == MATERIAL_ID_WATER) {
            const float RIOR = AIR_IOR / WATER_IOR;
            vec3 refractDir = refract(-uval_shadowLightDirWorld, normalize(worldNormal), RIOR);

            // Calculate closest distance from refracted ray to origin (0,0,0)
            // Ray: P = camViewPos + t * refractDir
            // Closest point: t = -dot(camViewPos, refractDir) / dot(refractDir, refractDir)
            float t = -dot(camViewPos.xyz, refractDir) / dot(refractDir, refractDir);
            t = max(t, 0.0); // Clamp to ray origin side
            vec3 closestPointOnRay = camViewPos.xyz + t * refractDir;
            camDistanceSq = pow2(dot(closestPointOnRay, closestPointOnRay)) * 0.1;
        }

        camDistanceSq = max(4.0, camDistanceSq);

        // Distance function
        importance *= 1.0 / (1.0 + pow(camDistanceSq, SETTING_RTWSM_F_D));
        importance = max(importance, uval_rtwsmMin.x);
        persistent_rtwsm_importance2D_atomicMax(importanceTexelPos, floatBitsToInt(importance));
    }
    #endif
}