
#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

layout(location = 0) out vec3 worldPosition;
layout(location = 1) out vec3 worldNormal;

layout(set = 0, binding = 0, std140) uniform SceneData {
    mat4 viewProjection;
    vec4 cameraPosition;
    vec4 ambientColor;
    vec4 directionalDirection;
    vec4 directionalColorIntensity;
    uvec4 lightCounts;
} scene;

layout(set = 1, binding = 0, std140) uniform ModelData {
    mat4 model;
    vec4 albedo;
    vec4 specularColor;
    vec4 materialParameters;
} material;

layout(push_constant) uniform DrawOverride {
    vec4 overrideColor;
    uint unlit;
} drawOverride;

void main()
{
    vec4 worldPos = material.model * vec4(inPosition, 1.0);

    worldPosition = worldPos.xyz;

    worldNormal = normalize(
        transpose(inverse(mat3(material.model))) * inNormal
    );

    gl_Position = scene.viewProjection * worldPos;
}
