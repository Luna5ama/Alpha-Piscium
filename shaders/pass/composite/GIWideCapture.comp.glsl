#include "/util/GBufferData.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);
#ifdef VOXY
layout(rgba32ui) uniform restrict writeonly uimage2D uimg_giWideSurface;
layout(rg32f) uniform restrict writeonly image2D uimg_giWideBounds;
#endif

void main() {
    #ifdef VOXY
    if (global_giWideState != 1) return;
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(p, uval_mainImageSizeI))) return;
    float z = texelFetch(usam_gbufferSolidViewZ, p, 0).r;
    uint normal = texelFetch(usam_gbufferSolidData1, p, 0).r;
    bool terrain = z < 0.0 && z > -65536.0 && texelFetch(usam_gbufferVoxySolidData, p, 0).w == floatBitsToUint(z);
    // Ordinary geometry still occludes the ray. Only native terrain owns retained radiance.
    uvec2 light = terrain ? transient_giRadianceInputs_fetch(p).xy : uvec2(0u);
    imageStore(uimg_giWideSurface, p, uvec4(floatBitsToUint(z), normal, light));
    if (all(lessThan(p, imageSize(uimg_giWideBounds)))) {
        vec2 bounds = texelFetch(usam_hiz, p + ivec2(0, uval_mainImageSizeI.y), 0).ba;
        imageStore(uimg_giWideBounds, p, vec4(bounds, 0.0, 0.0));
    }
    #endif
}
