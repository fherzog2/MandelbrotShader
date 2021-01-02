#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject {
    dvec2 center;
    double scale;
    int iterations;
    double aspect;
    float colormap_index;
} ubo;

layout(binding = 1) uniform sampler2DArray colormap_array;

layout(location = 0) in vec2 inTexCoord;

layout(location = 0) out vec4 outColor;

void main() {
    dvec2 c = (inTexCoord - dvec2(0.5, 0.5)) * ubo.scale - ubo.center;

    int i;
    dvec2 z = c;
    for(i = 0; i < ubo.iterations; ++i){
	double x = (z.x * z.x - z.y * z.y) + c.x;
	double y = (z.y * z.x + z.x * z.y) + c.y;

	if((x * x + y * y) > 4.0)
	    break;
	z.x = x;
	z.y = y;
    }

    outColor = texture(colormap_array, vec3(float(i) / float(ubo.iterations), 0, ubo.colormap_index));
}