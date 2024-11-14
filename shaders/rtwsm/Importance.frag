#version 460 core

#include "../_Util.glsl"

layout(early_fragment_tests) in;

in float fImportance;

out float outImportance;

void main() {
    outImportance = fImportance;
}