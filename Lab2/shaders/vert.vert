#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 0) out vec3 worldPosition;
layout(location = 1) out vec3 worldNormal;

layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 view;
    mat4 proj;
    vec4 cameraPos;
    vec4 ambientColor;
    vec4 directionalDirection;
    vec4 directionalColor;
} scene;

layout(set = 1, binding = 0) uniform ModelUBO {
    mat4 model;
    vec4 albedo;
    vec4 specularColor;
    vec4 materialParams;
} material;

layout(push_constant) uniform DrawOverride {
    mat4 model;
    vec4 albedo;
    vec4 specularColor;
    vec4 materialParams;
} drawOverride;

void main() {
    bool useOverride = drawOverride.materialParams.w > 0.5;
    mat4 modelMatrix = useOverride ? drawOverride.model : material.model;
    vec4 world = modelMatrix * vec4(inPosition, 1.0);
    worldPosition = world.xyz;
    worldNormal = normalize(transpose(inverse(mat3(modelMatrix))) * inNormal);
    gl_Position = scene.proj * scene.view * world;
}