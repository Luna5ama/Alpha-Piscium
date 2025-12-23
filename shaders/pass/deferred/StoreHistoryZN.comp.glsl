#include "/util/Coords.glsl"
#include "/util/GBufferData.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(r32f) uniform restrict writeonly image2D uimg_r32f;
layout(rgb10_a2) uniform restrict writeonly image2D uimg_rgb10_a2;
layout(rgba8) uniform restrict writeonly image2D uimg_rgba8;

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        vec3 viewNormal = vec3(0.0, 1.0, 0.0);
        vec3 geomViewNomral = vec3(0.0, 1.0, 0.0);
        float viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;
        vec3 albedo = vec3(0.0);
        if (viewZ > -65536.0) {
            GBufferData gData = gbufferData_init();
            gbufferData1_unpack(texelFetch(usam_gbufferData1, texelPos, 0), gData);
            gbufferData2_unpack(texelFetch(usam_gbufferData2, texelPos, 0), gData);
            viewNormal = gData.normal;
            geomViewNomral = gData.geomNormal;
            albedo = gData.albedo;
        }

        history_viewZ_store(texelPos, vec4(viewZ));
        history_geomViewNormal_store(texelPos, vec4(geomViewNomral * 0.5 + 0.5, 0.0));
        history_viewNormal_store(texelPos, vec4(viewNormal * 0.5 + 0.5, 0.0));
        transient_solidAlbedo_store(texelPos, vec4(albedo, 1.0));
    }
}
