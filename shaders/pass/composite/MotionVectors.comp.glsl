#include "/techniques/textile/CSR32F.glsl"
#include "/util/GBufferData.glsl"

layout(local_size_x = 8, local_size_y = 8) in;
const vec2 workGroupsRender = vec2(RENDER_SCALE_FACTOR, RENDER_SCALE_FACTOR);

layout(rgba16f) uniform restrict writeonly image2D uimg_rgba16f;
layout(r32ui) uniform restrict writeonly uimage2D uimg_fsr3ReconstructedDepth;

bool hasTranslucentSurface(ivec2 texelPos, float solidViewZ) {
    float waterViewZ = -texelFetch(usam_csr32f, csr32f_tile1_texelToTexel(texelPos), 0).r;
    float translucentViewZ = -texelFetch(usam_csr32f, csr32f_tile3_texelToTexel(texelPos), 0).r;
    float frontViewZ = max(waterViewZ, translucentViewZ);
    return frontViewZ > -65536.0 && frontViewZ > solidViewZ;
}

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(texelPos, uval_mainImageSizeI))) return;

    vec2 screenUv = (vec2(texelPos) + 0.5) * uval_mainImageSizeRcp;
    vec2 currentUv = screenUv - uval_taaJitterUV;
    float viewZ = texelFetch(usam_gbufferSolidViewZ, texelPos, 0).r;
    bool isSky = viewZ <= -65535.0;

    GBufferData solidData = gbufferData_init();
    gbufferData2_unpack(texelFetch(usam_gbufferSolidData2, texelPos, 0), solidData);

    vec4 previousClip;
    if (isSky) {
        vec3 viewDirection = normalize(coords_toViewCoord(currentUv, -1.0, global_camProjInverse));
        vec3 worldDirection = coords_dir_viewToWorld(viewDirection);
        vec3 previousViewDirection = coords_dir_worldToViewPrev(worldDirection);
        previousClip = global_prevCamProj * vec4(previousViewDirection, 0.0);
    } else {
        vec3 currentViewPosition = coords_toViewCoord(currentUv, viewZ, global_camProjInverse);
        vec4 previousViewPosition = coord_viewCurrToPrev(vec4(currentViewPosition, 1.0), solidData.isHand);
        previousClip = global_prevCamProj * previousViewPosition;
    }

    bool validReprojection = frameCounter > 1;
    validReprojection = validReprojection && previousClip.w > 0.0;
    validReprojection = validReprojection && (isSky || previousClip.z > 0.0);
    validReprojection = validReprojection && all(lessThan(abs(previousClip.xy), previousClip.ww));

    vec2 previousUv = previousClip.xy / previousClip.w * 0.5 + 0.5;
    vec2 motionVector = validReprojection ? previousUv - currentUv : vec2(0.0);

    bool translucentSurface = hasTranslucentSurface(texelPos, viewZ);
    GBufferData translucentData = gbufferData_init();
    gbufferData2_unpack(texelFetch(usam_gbufferTranslucentData2, texelPos, 0), translucentData);
    float overlayCoverage = texelFetch(usam_overlays, texelPos, 0).a;

    float reactiveMask = max(float(solidData.temporalReactive), overlayCoverage);
    reactiveMask = max(reactiveMask, float(translucentSurface && translucentData.temporalReactive));
    reactiveMask = max(reactiveMask, float(!validReprojection));
    float compositionMask = max(float(translucentSurface || solidData.temporalReactive), overlayCoverage);

    history_fsr3Motion_store(texelPos, vec4(motionVector, reactiveMask, compositionMask));
    imageStore(uimg_fsr3ReconstructedDepth, texelPos, uvec4(0u));
}
