#version 460 compatibility

#include "/atmosphere/UnwarpEpipolar.glsl"
#include "/atmosphere/Scattering.glsl"
#include "/util/FullScreenComp.glsl"
#include "/util/Coords.glsl"
#include "/util/Rand.glsl"
#include "/util/Material.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(0.5, 0.5);

uniform sampler2D usam_temp4;

uniform usampler2D usam_gbufferData32UI;
uniform sampler2D usam_gbufferData8UN;
uniform sampler2D usam_translucentColor;

layout(rgba16f) writeonly uniform image2D uimg_temp3;

void updateMoments(vec3 colorSRGB, inout vec3 sum, inout vec3 sqSum) {
    vec3 color = colors_SRGBToYCoCg(colorSRGB);
    sum += color;
    sqSum += color * color;
}

void main() {
    if (all(lessThan(texelPos, global_mipmapSizesI[1]))) {
        vec3 curr3x3Avg = vec3(0.0);
        vec3 curr3x3SqAvg = vec3(0.0);
        updateMoments(texelFetchOffset(usam_temp4, texelPos, 0, ivec2(-1, 0)).rgb, curr3x3Avg, curr3x3SqAvg);
        updateMoments(texelFetchOffset(usam_temp4, texelPos, 0, ivec2(1, 0)).rgb, curr3x3Avg, curr3x3SqAvg);
        updateMoments(texelFetchOffset(usam_temp4, texelPos, 0, ivec2(0, -1)).rgb, curr3x3Avg, curr3x3SqAvg);
        updateMoments(texelFetchOffset(usam_temp4, texelPos, 0, ivec2(0, 1)).rgb, curr3x3Avg, curr3x3SqAvg);
        updateMoments(texelFetchOffset(usam_temp4, texelPos, 0, ivec2(-1, -1)).rgb, curr3x3Avg, curr3x3SqAvg);
        updateMoments(texelFetchOffset(usam_temp4, texelPos, 0, ivec2(1, -1)).rgb, curr3x3Avg, curr3x3SqAvg);
        updateMoments(texelFetchOffset(usam_temp4, texelPos, 0, ivec2(-1, 1)).rgb, curr3x3Avg, curr3x3SqAvg);
        updateMoments(texelFetchOffset(usam_temp4, texelPos, 0, ivec2(1, 1)).rgb, curr3x3Avg, curr3x3SqAvg);
        curr3x3Avg /= 9.0;
        curr3x3SqAvg /= 9.0;

        vec3 centerGI = texelFetch(usam_temp4, texelPos, 0).rgb;

        // Ellipsoid intersection clipping by Marty
        vec3 centerGIYCoCg = colors_SRGBToYCoCg(centerGI);
        vec3 stddev = sqrt(curr3x3SqAvg - curr3x3Avg * curr3x3Avg);
        vec3 delta = centerGIYCoCg - curr3x3Avg;
        const float clippingEps = 0.00001;
        delta /= max(1.0, length(delta / (stddev + clippingEps)));
        centerGIYCoCg = curr3x3Avg + delta;
        centerGI.rgb = colors_YCoCgToSRGB(centerGIYCoCg);

        imageStore(uimg_temp3, texelPos, vec4(centerGI, 0.0));
    }
}