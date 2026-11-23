
#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec3 worldPosition;
layout(location = 1) out vec3 worldNormal;
layout(location = 2) out vec2 uv;

layout(set = 0, binding = 0, std140) uniform SceneData {
    mat4 viewProjection;
    vec4 cameraPosition;
    vec4 ambientColor;
    vec4 directionalDirection;
    vec4 directionalColorIntensity;
    uvec4 lightCounts;
    // Матрицы "источник света -> текстура глубины": ортографическая
    // проекция солнца и перспективная проекция прожектора.
    mat4 sunViewProjection;
    mat4 spotViewProjection;
    // x = размер текселя теневой карты (1 / размер карты).
    vec4 shadowSettings;
} scene;

// Общий динамический uniform-буфер моделей. Конкретный ModelData
// выбирается динамическим смещением, переданным при привязке
// набора дескрипторов vkCmdBindDescriptorSets.
layout(set = 1, binding = 0, std140) uniform ModelData {
    mat4 model;
    vec4 albedo;
    vec4 specularColor;
    vec4 materialParameters;
} material;

layout(push_constant) uniform DrawOverride {
    vec4 overrideColor;
    float time;
    float textureEffectStrength;
    uint unlit;
    uint textureMode;
} drawOverride;

void main()
{
    vec4 worldPos = material.model * vec4(inPosition, 1.0);

    worldPosition = worldPos.xyz;

    worldNormal = normalize(
        transpose(inverse(mat3(material.model))) * inNormal
    );

    uv = inUV;

    gl_Position = scene.viewProjection * worldPos;
}
