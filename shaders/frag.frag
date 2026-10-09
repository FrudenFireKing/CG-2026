#version 450

layout(binding = 1) uniform UBO_Color {
    vec4 color;
} uboColor;

layout(location = 0) in vec3 fragColor;
layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(fragColor, 1.0) * uboColor.color;
}
