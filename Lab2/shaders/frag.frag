#version 450

layout(location = 0) in vec3 worldPosition;
layout(location = 1) in vec3 worldNormal;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 view;
    mat4 proj;
    vec4 cameraPos;
    vec4 ambientColor;
    vec4 directionalDirection;
    vec4 directionalColor;
} scene;

struct PointLight { vec4 position; vec4 colorIntensity; };
layout(std430, set = 0, binding = 1) readonly buffer PointLightBuffer { PointLight lights[]; } pointBuffer;

struct SpotLight { vec4 position; vec4 direction; vec4 colorIntensity; vec4 coneParams; };
layout(std430, set = 0, binding = 2) readonly buffer SpotLightBuffer { SpotLight lights[]; } spotBuffer;

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

vec3 blinnPhong(vec3 N, vec3 V, vec3 L, vec3 radiance, vec3 albedo, vec3 specularColor, float shininess) {
    float ndotl = max(dot(N, L), 0.0);
    vec3 diffuse = albedo * ndotl;
    vec3 H = normalize(L + V);
    float spec = ndotl > 0.0 ? pow(max(dot(N, H), 0.0), max(shininess, 1.0)) : 0.0;
    return (diffuse + specularColor * spec) * radiance;
}

void main() {
    bool useOverride = drawOverride.materialParams.w > 0.5;
    vec4 albedo = useOverride ? drawOverride.albedo : material.albedo;
    vec3 specularColor = useOverride ? drawOverride.specularColor.rgb : material.specularColor.rgb;
    vec4 params = useOverride ? drawOverride.materialParams : material.materialParams;
    if (params.y > 0.5) {
        outColor = vec4(albedo.rgb, albedo.a);
        return;
    }

    vec3 N = normalize(worldNormal);
    vec3 V = normalize(scene.cameraPos.xyz - worldPosition);
    vec3 result = albedo.rgb * scene.ambientColor.rgb;
    vec3 directionalL = normalize(-scene.directionalDirection.xyz);
    result += blinnPhong(N, V, directionalL, scene.directionalColor.rgb * scene.directionalColor.a,
                         albedo.rgb, specularColor, params.x);

    for (uint i = 0; i < pointBuffer.lights.length(); ++i) {
        PointLight light = pointBuffer.lights[i];
        vec3 delta = light.position.xyz - worldPosition;
        float d2 = max(dot(delta, delta), 0.0001);
        vec3 L = delta * inversesqrt(d2);
        vec3 radiance = light.colorIntensity.rgb * light.colorIntensity.a / d2;
        result += blinnPhong(N, V, L, radiance, albedo.rgb, specularColor, params.x);
    }

    for (uint i = 0; i < spotBuffer.lights.length(); ++i) {
        SpotLight light = spotBuffer.lights[i];
        vec3 delta = light.position.xyz - worldPosition;
        float d2 = max(dot(delta, delta), 0.0001);
        vec3 L = delta * inversesqrt(d2);
        float theta = dot(normalize(light.direction.xyz), -L);
        float innerCos = light.coneParams.x, outerCos = light.coneParams.y;
        float epsilon = max(innerCos - outerCos, 0.0001);
        float cone = clamp((theta - outerCos) / epsilon, 0.0, 1.0);
        cone = cone * cone * (3.0 - 2.0 * cone);
        vec3 radiance = light.colorIntensity.rgb * light.colorIntensity.a * cone / d2;
        result += blinnPhong(N, V, L, radiance, albedo.rgb, specularColor, params.x);
    }
    result = result / (vec3(1.0) + result);
    outColor = vec4(result, albedo.a);
}