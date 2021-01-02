#version 450

layout (std140) uniform shader_data {
    dvec2 center;
    double scale;
    int iterations;
    double aspect;
    float colormap_index;
} ubo;

layout(location = 0) in vec2 in_position;
layout(location = 1) in vec2 in_texcoord;

out vec2 frag_texcoord;

void main() {
    double tx = (in_texcoord.x - 0.5) * ubo.aspect + 0.5;
    double ty = 1 - in_texcoord.y;

    gl_Position = vec4(in_position, 0, 1.0);
    frag_texcoord = vec2(tx, ty);
}