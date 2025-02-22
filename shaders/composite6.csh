#version 460 compatibility

#include "/general/EnvProbe.glsl"
#include "/util/Morton.glsl"

layout(local_size_x = 128) in;
const ivec3 workGroups = ivec3(2048, 1, 1);

uniform sampler2D usam_temp2;
uniform usampler2D usam_gbufferData;
uniform sampler2D usam_gbufferViewZ;

layout(rgba32ui) uniform restrict uimage2D uimg_envProbe;

void main() {
    ivec2 texelPos = ivec2(morton_32bDecode(gl_GlobalInvocationID.x));
    ivec2 inputPos = texelPos;
    inputPos.x += 512;
    ivec2 outputPos = texelPos;

    EnvProbeData outputData;
    envProbe_initData(outputData);

    EnvProbeData inputDataC = envProbe_decode(imageLoad(uimg_envProbe, inputPos));
    if (envProbe_hasData(inputDataC)) {
        outputData = inputDataC;
    } else {
        uint count = 0u;
        EnvProbeData nearData;

        if ((frameCounter & 1) == 0) {
            nearData = envProbe_decode(imageLoad(uimg_envProbe, inputPos + ivec2(-1, 0)));
            if (envProbe_hasData(nearData)) {
                outputData = envProbe_add(outputData, nearData);
                count++;
            }
            nearData = envProbe_decode(imageLoad(uimg_envProbe, inputPos + ivec2(1, 0)));
            if (envProbe_hasData(nearData)) {
                outputData = envProbe_add(outputData, nearData);
                count++;
            }
            nearData = envProbe_decode(imageLoad(uimg_envProbe, inputPos + ivec2(0, -1)));
            if (envProbe_hasData(nearData)) {
                outputData = envProbe_add(outputData, nearData);
                count++;
            }
            nearData = envProbe_decode(imageLoad(uimg_envProbe, inputPos + ivec2(0, -1)));
            if (envProbe_hasData(nearData)) {
                outputData = envProbe_add(outputData, nearData);
                count++;
            }
        } else {
            nearData = envProbe_decode(imageLoad(uimg_envProbe, inputPos + ivec2(-1, -1)));
            if (envProbe_hasData(nearData)) {
                outputData = envProbe_add(outputData, nearData);
                count++;
            }
            nearData = envProbe_decode(imageLoad(uimg_envProbe, inputPos + ivec2(1, -1)));
            if (envProbe_hasData(nearData)) {
                outputData = envProbe_add(outputData, nearData);
                count++;
            }
            nearData = envProbe_decode(imageLoad(uimg_envProbe, inputPos + ivec2(-1, 1)));
            if (envProbe_hasData(nearData)) {
                outputData = envProbe_add(outputData, nearData);
                count++;
            }
            nearData = envProbe_decode(imageLoad(uimg_envProbe, inputPos + ivec2(1, 1)));
            if (envProbe_hasData(nearData)) {
                outputData = envProbe_add(outputData, nearData);
                count++;
            }
        }

        if (count > 0.0) {
            float rcpCount = 1.0 / float(count);
            outputData.radiance *= rcpCount;
            outputData.normal *= rcpCount;
            outputData.scenePos *= rcpCount;
        }
    }

    imageStore(uimg_envProbe, outputPos, envProbe_encode(outputData));
}