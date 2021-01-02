#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject {
    dvec2 center;
    double scale;
    int iterations;
    double aspect;
    float colormap_index;
} ubo;

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec2 inTexCoord;

layout(location = 0) out vec2 outTexCoord;

void main() {
    double tx = (inTexCoord.x - 0.5) * ubo.aspect + 0.5;

    gl_Position = vec4(inPosition, 0.0, 1.0);
    outTexCoord = vec2(tx, inTexCoord.y);
}
