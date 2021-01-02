#version 450

layout (std140) uniform shader_data {
    dvec2 center;
    double scale;
    int iterations;
    double aspect;
    float colormap_index;
} ubo;

uniform sampler2DArray colormap_array;

in vec2 frag_texcoord;
out vec4 frag_colour;

void main() {
    dvec2 c = (frag_texcoord - dvec2(0.5, 0.5)) * ubo.scale - ubo.center;

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

    frag_colour = texture(colormap_array, vec3(float(i) / float(ubo.iterations), 0, ubo.colormap_index));
}