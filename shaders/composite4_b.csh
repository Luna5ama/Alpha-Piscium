#version 460 compatibility

#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_vote : enable

#include "/util/Colors.glsl"
#include "/util/Morton.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

uniform usampler2D usam_tempRGBA32UI;
layout(rgba16f) uniform writeonly image2D uimg_temp3;
layout(rgba16f) uniform writeonly image2D uimg_temp4;

shared uvec4 shared_moments[20][20];
shared uvec4 shared_momentsV[16][20];

uvec2 groupOriginTexelPos;

uvec4 packMoments(vec3 moment1, vec3 moment2) {
    return uvec4(
        packHalf2x16(moment1.xy),
        packHalf2x16(vec2(moment1.z, moment2.x)),
        packHalf2x16(vec2(moment2.y, moment2.z)),
        0u
    );
}

void loadSharedData(uint index) {
    if (index < 400) {
        uvec2 sharedXY = uvec2(index % 20, index / 20);
        ivec2 srcXY = ivec2(groupOriginTexelPos) + ivec2(sharedXY) - 2;
        srcXY = clamp(srcXY, ivec2(0), ivec2(global_mainImageSize - 1));

        vec3 inputColor = colors_LogLuvToSRGB(unpackUnorm4x8(texelFetch(usam_tempRGBA32UI, srcXY, 0).y));
        inputColor = colors_SRGBToYCoCg(inputColor);
        vec3 moment1 = inputColor;
        vec3 moment2 = inputColor * inputColor;

        shared_moments[sharedXY.y][sharedXY.x] = packMoments(moment1, moment2);
    }
}

void updateMoments0(uvec2 originXY, ivec2 offset, inout vec3 moment1, inout vec3 moment2) {
    ivec2 sampleXY = ivec2(originXY) + offset;
    uvec4 packedData = shared_moments[sampleXY.y][sampleXY.x];
    vec2 temp1 = unpackHalf2x16(packedData.x);
    vec2 temp2 = unpackHalf2x16(packedData.y);
    vec2 temp3 = unpackHalf2x16(packedData.z);
    moment1 += vec3(temp1, temp2.x);
    moment2 += vec3(temp2.y, temp3.xy);
}

void sampleV(uint index) {
    if (index < 320) {
        uvec2 writeSharedXY = uvec2(index % 20, index / 20);
        uvec2 readSharedXY = writeSharedXY;
        readSharedXY.y += 2;
        vec3 moment1 = vec3(0.0);
        vec3 moment2 = vec3(0.0);
        updateMoments0(readSharedXY, ivec2(0, -2), moment1, moment2);
        updateMoments0(readSharedXY, ivec2(0, -1), moment1, moment2);
        updateMoments0(readSharedXY, ivec2(0, 0), moment1, moment2);
        updateMoments0(readSharedXY, ivec2(0, 1), moment1, moment2);
        updateMoments0(readSharedXY, ivec2(0, 2), moment1, moment2);
        moment1 /= 5.0;
        moment2 /= 5.0;
        shared_momentsV[writeSharedXY.y][writeSharedXY.x] = packMoments(moment1, moment2);
    }
}

void updateMoments1(uvec2 originXY, ivec2 offset, inout vec3 moment1, inout vec3 moment2) {
    ivec2 sampleXY = ivec2(originXY) + offset;
    uvec4 packedData = shared_momentsV[sampleXY.y][sampleXY.x];
    vec2 temp1 = unpackHalf2x16(packedData.x);
    vec2 temp2 = unpackHalf2x16(packedData.y);
    vec2 temp3 = unpackHalf2x16(packedData.z);
    moment1 += vec3(temp1, temp2.x);
    moment2 += vec3(temp2.y, temp3.xy);
}

void main() {
    groupOriginTexelPos = gl_WorkGroupID.xy << 4u;
    loadSharedData(gl_LocalInvocationIndex);
    loadSharedData(gl_LocalInvocationIndex + 256);
    barrier();

    sampleV(gl_LocalInvocationIndex);
    sampleV(gl_LocalInvocationIndex + 256);
    barrier();

    ivec2 texelPos = ivec2(groupOriginTexelPos) + ivec2(gl_LocalInvocationID.xy);
    if (all(lessThan(texelPos, global_mainImageSizeI))) {
        uvec2 readSharedXY = gl_LocalInvocationID.xy;
        readSharedXY.x += 2;
        vec3 moment1 = vec3(0.0);
        vec3 moment2 = vec3(0.0);
        updateMoments1(readSharedXY, ivec2(-2, 0), moment1, moment2);
        updateMoments1(readSharedXY, ivec2(-1, 0), moment1, moment2);
        updateMoments1(readSharedXY, ivec2(0, 0), moment1, moment2);
        updateMoments1(readSharedXY, ivec2(1, 0), moment1, moment2);
        updateMoments1(readSharedXY, ivec2(2, 0), moment1, moment2);
        moment1 /= 5.0;
        moment2 /= 5.0;
        const float EPS = 0.00001;
        vec3 variance = max(moment2 - moment1 * moment1, EPS);
        vec3 stddev = sqrt(variance);
        imageStore(uimg_temp3, texelPos, vec4(moment1, 0.0));
        imageStore(uimg_temp4, texelPos, vec4(stddev, 0.0));
    }
}