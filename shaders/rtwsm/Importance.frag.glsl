#version 460 core

layout(early_fragment_tests) in;

in float fImportance;

out float outImportance;

void main() {
    outImportance = fImportance;
}