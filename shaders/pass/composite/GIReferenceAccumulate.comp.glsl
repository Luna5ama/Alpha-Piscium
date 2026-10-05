// Accumulates the one-sample Monte Carlo estimate of the initial GI candidate in FP32 while the camera is static.
// TranslucentBackComposite shows the running mean instead of the denoised GI, bypassing reuse and denoising.
#include "/Base.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba32f) uniform restrict image2D uimg_giReferenceDiff;
layout(rgba32f) uniform restrict image2D uimg_giReferenceSpec;

#include "/techniques/gi/InitialSample.glsl"
#include "/techniques/gi/Reservoir.glsl"

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(texelPos, uval_mainImageSizeI))) {
        return;
    }

    vec3 diffSample = vec3(0.0);
    vec3 specSample = vec3(0.0);
    float viewZ = texelFetch(usam_gbufferSolidViewZ, texelPos, 0).x;
    restir_InitialCandidate candidate = restir_initialCandidate_load(texelPos);
    if (viewZ > -65536.0 && candidate.pdf > 0.0) {
        vec3 geomNormal = normalize(transient_geomViewNormal_fetch(texelPos).xyz * 2.0 - 1.0);
        #if SETTING_GI_USE_REFERENCE == 3
        // Furnace: E[cos / (pi q)] = 1 over the sampled hemisphere when the pdf matches the sampler.
        diffSample = vec3(max(dot(geomNormal, candidate.rayDirView), 0.0) * RCP_PI / candidate.pdf);
        #else
        vec2 screenPos = coords_texelToUV(texelPos, uval_mainImageSizeRcp) - uval_taaJitterUV;
        vec3 V = normalize(-coords_toViewCoord(screenPos, viewZ, global_camProjInverse));
        vec3 normal = normalize(transient_viewNormal_fetch(texelPos).xyz * 2.0 - 1.0);
        ResampleMaterial material = resampleMaterial_unpack(transient_restir_resampleMaterial_fetch(texelPos));
        vec4 diffOut;
        vec4 specOut;
        restir_shadeSample(
            candidate.radiance,
            vec4(candidate.rayDirView, candidate.hitDistance),
            rcp(candidate.pdf),
            geomNormal,
            normal,
            V,
            V,
            material,
            texelPos,
            diffOut,
            specOut
        );
        diffSample = diffOut.rgb;
        specSample = specOut.rgb;
        #endif
    }

    bool resetHistory = frameCounter <= 1
        || gbufferModelView != gbufferPreviousModelView
        || gbufferProjection != gbufferPreviousProjection
        || cameraPosition != previousCameraPosition;

    // rgb: sum of samples, a: sum of squared diffuse luminance (diff) / sample count (spec).
    vec4 diffSum = vec4(diffSample, pow2(colors_colorspaces_luma(COLORS_WORKING_COLORSPACE, diffSample)));
    vec4 specSum = vec4(specSample, 1.0);
    if (!resetHistory) {
        diffSum += imageLoad(uimg_giReferenceDiff, texelPos);
        specSum += imageLoad(uimg_giReferenceSpec, texelPos);
    }
    imageStore(uimg_giReferenceDiff, texelPos, diffSum);
    imageStore(uimg_giReferenceSpec, texelPos, specSum);
}
