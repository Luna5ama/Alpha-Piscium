#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_KHR_shader_subgroup_ballot : enable

#define GLOBAL_DATA_MODIFIER buffer

#include "/Base.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(RENDER_SCALE_FACTOR, RENDER_SCALE_FACTOR);

#include "/util/FullScreenComp.glsl"
#include "/techniques/gi/DenoiserEdgeClassification.glsl"
#include "/techniques/atmospherics/VolumetricsDepthLayers.glsl"

void main() {
    classifyGIDenoiserEdges();

    float solid = loadSampleViewZ(ivec2(gl_LocalInvocationID.xy) + 1);
    updateVolumetricsDepthLayers(solid);
}
