#include "/Base.glsl"

#define SSVBIL_SAMPLE_STEPS222 64

uint vbgi_downSampleInputMortonIndex() {
    return uint(frameCounter) & 3u;
}

bool vbgi_selectDownSampleInput(uint mortonIndex) {
    return (mortonIndex & 3u) == vbgi_downSampleInputMortonIndex();
}