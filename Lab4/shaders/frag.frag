
#version 450

layout(location = 0) in vec3 worldPosition;
layout(location = 1) in vec3 worldNormal;
layout(location = 2) in vec2 uv;

layout(location = 0) out vec4 outColor;

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

struct PointLight {
    vec4 position;
    vec4 colorIntensity;
};

layout(set = 0, binding = 1, std430) readonly buffer PointLightBuffer {
    PointLight lights[];
} pointBuffer;

struct SpotLight {
    vec4 position;
    vec4 direction;
    vec4 colorIntensity;
    vec4 cone;
};

layout(set = 0, binding = 2, std430) readonly buffer SpotLightBuffer {
    SpotLight lights[];
} spotBuffer;

layout(set = 1, binding = 0, std140) uniform ModelData {
    mat4 model;
    vec4 albedo;
    vec4 specularColor;
    // x = shininess, y = сила зеркальной составляющей,
    // z = сила свечения (emissive).
    vec4 materialParameters;
} material;

// Карты теней: сравнивающие сэмплеры (sampler2DShadow).
// Оба набора объединяют одну и ту же карту глубины, заполненную
// динамическим рендерингом, и сэмплер с включённым сравнением
// глубины (VK_COMPARE_OP_LESS) и границей CLAMP_TO_BORDER с
// непрозрачным белым цветом: точки вне теневой карты считаются
// освещёнными.
layout(set = 0, binding = 3) uniform sampler2DShadow sunShadowMap;
layout(set = 0, binding = 4) uniform sampler2DShadow spotShadowMap;

// Три карты материала: цветовая (albedo), зеркальная (specular)
// и карта свечения (emissive).
layout(set = 1, binding = 1) uniform sampler2D albedoTexture;
layout(set = 1, binding = 2) uniform sampler2D specularTexture;
layout(set = 1, binding = 3) uniform sampler2D emissiveTexture;

// Push-константы управляют включаемым анимированным сэмплированием
// и служебными режимами отрисовки (маркеры источников света,
// процедурная окраска уровня лабораторной №2).
layout(push_constant) uniform DrawOverride {
    vec4 overrideColor;
    float time;
    float textureEffectStrength;
    uint unlit;
    uint textureMode;
} drawOverride;

// Вычисление видимости для карты теней. Точка переводится в
// пространство текстуры источника света, глубина сравнивается
// аппаратным сравнивающим сэмплером с глубиной из карты теней.
// Для смягчения краёв выполняется PCF 3x3 с шагом в один тексель.
float shadowVisibility(
    sampler2DShadow shadowMap,
    mat4 lightMatrix,
    vec3 position
) {
    vec4 clip = lightMatrix * vec4(position, 1.0);

    if (clip.w <= 0.0) {
        return 1.0;
    }

    vec3 projected = clip.xyz / clip.w;

    vec2 shadowUV = projected.xy * 0.5 + 0.5;

    // За пределами объёма, покрываемого проекцией источника света,
    // поверхность считается освещённой (граница карты - белый цвет).
    if (any(lessThan(shadowUV, vec2(0.0))) ||
        any(greaterThan(shadowUV, vec2(1.0))) ||
        projected.z <= 0.0 || projected.z >= 1.0) {
        return 1.0;
    }

    float visibility = 0.0;

    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            vec2 offset = vec2(x, y) * scene.shadowSettings.x;

            visibility += texture(
                shadowMap,
                vec3(shadowUV + offset, projected.z)
            );
        }
    }

    return visibility / 9.0;
}

vec3 blinnPhong(
    vec3 N,
    vec3 V,
    vec3 L,
    vec3 radiance,
    vec3 albedoColor,
    vec3 specular,
    float shininess
) {
    float NdotL = max(dot(N, L), 0.0);

    vec3 diffuse = albedoColor * NdotL;

    vec3 H = normalize(V + L);

    float spec = 0.0;

    if (NdotL > 0.0) {
        spec = pow(
            max(dot(N, H), 0.0),
            max(shininess, 1.0)
        );
    }

    return (diffuse + specular * spec) * radiance;
}

void main()
{
    // Маркеры источников света отображаются без освещения.
    if (drawOverride.unlit != 0) {
        outColor = drawOverride.overrideColor;
        return;
    }

    vec3 albedoColor = material.albedo.rgb;
    // Карта зеркальности маскирует блик: 1.0 соответствует
    // процедурной окраске лабораторной №2.
    float specularMask = 1.0;
    vec3 emissive = vec3(0.0);

    if (drawOverride.textureMode != 0) {
        vec4 albedoSample = texture(albedoTexture, uv);
        vec3 sampledAlbedo = albedoSample.rgb;

        // Анимированное сэмплирование: та же albedo-карта сэмплируется
        // во вторых, изменяющихся со временем текстурных координатах.
        // Результат смешивается с базовым и модулируется.
        if (drawOverride.textureEffectStrength > 0.0) {
            vec2 drift = 0.18 * vec2(
                sin(drawOverride.time * 0.73),
                cos(drawOverride.time * 0.51)
            );

            vec2 warped = uv * 2.0 + drift
                + 0.05 * sin(vec2(uv.y, uv.x) * 18.0
                             + drawOverride.time * 2.0);

            vec3 second = texture(albedoTexture, warped).rgb;

            float pulse = 0.5 + 0.5 * sin(
                drawOverride.time * 1.8 + (uv.x + uv.y) * 9.0
            );

            float blend =
                drawOverride.textureEffectStrength * (0.25 + 0.45 * pulse);

            sampledAlbedo = mix(sampledAlbedo, second, blend);

            sampledAlbedo *= 1.0
                + drawOverride.textureEffectStrength * 0.25
                * sin(drawOverride.time * 2.4 + uv.x * 12.0);
        }

        albedoColor = sampledAlbedo * material.albedo.rgb;

        // Красный канал карты зеркальности задаёт силу блика.
        specularMask = texture(specularTexture, uv).r;

        // Карта свечения добавляется независимо от освещения.
        emissive = texture(emissiveTexture, uv).rgb;
    }

    vec3 N = normalize(worldNormal);
    vec3 V = normalize(scene.cameraPosition.xyz - worldPosition);

    float shininess = material.materialParameters.x;
    float specularStrength = material.materialParameters.y;
    float emissiveStrength = material.materialParameters.z;

    vec3 specular = material.specularColor.rgb * specularMask * specularStrength;

    // Окружающее освещение.
    vec3 result =
        albedoColor *
        scene.ambientColor.rgb *
        scene.ambientColor.a;

    // Направленный источник света. Тень от солнца учитывается
    // только если карта теней была построена в этом кадре
    // (lightCounts.z != 0).
    vec3 directionalL =
        normalize(-scene.directionalDirection.xyz);

    vec3 directionalRadiance =
        scene.directionalColorIntensity.rgb *
        scene.directionalColorIntensity.a;

    if (scene.lightCounts.z != 0) {
        directionalRadiance *= shadowVisibility(
            sunShadowMap,
            scene.sunViewProjection,
            worldPosition
        );
    }

    result += blinnPhong(
        N, V, directionalL,
        directionalRadiance,
        albedoColor, specular, shininess
    );

    // Точечные источники: затухание обратно пропорционально квадрату расстояния.
    for (uint i = 0; i < scene.lightCounts.x; ++i) {
        PointLight light = pointBuffer.lights[i];

        vec3 delta = light.position.xyz - worldPosition;

        float distanceSquared = max(dot(delta, delta), 0.0001);

        vec3 L = delta * inversesqrt(distanceSquared);

        vec3 radiance =
            light.colorIntensity.rgb *
            light.colorIntensity.a /
            distanceSquared;

        result += blinnPhong(
            N, V, L, radiance,
            albedoColor, specular, shininess
        );
    }

    // Прожекторы с мягкой границей светового конуса.
    for (uint i = 0; i < scene.lightCounts.y; ++i) {
        SpotLight light = spotBuffer.lights[i];

        vec3 delta = light.position.xyz - worldPosition;

        float distanceSquared = max(dot(delta, delta), 0.0001);

        vec3 L = delta * inversesqrt(distanceSquared);

        float theta = dot(
            normalize(light.direction.xyz),
            -L
        );

        // C++ хранит углы в радианах.
        float innerCos = cos(light.cone.x);
        float outerCos = cos(light.cone.y);

        float coneWidth = max(innerCos - outerCos, 0.0001);

        float cone = clamp(
            (theta - outerCos) / coneWidth,
            0.0,
            1.0
        );

        // Сглаживание перехода от света к тени.
        cone = cone * cone * (3.0 - 2.0 * cone);

        // Тень от первого прожектора (дополнительное задание):
        // карта глубины с точки зрения прожектора строится
        // перспективной проекцией.
        float spotVisibility = 1.0;

        if (i == 0 && scene.lightCounts.w != 0) {
            spotVisibility = shadowVisibility(
                spotShadowMap,
                scene.spotViewProjection,
                worldPosition
            );
        }

        vec3 radiance =
            light.colorIntensity.rgb *
            light.colorIntensity.a *
            cone *
            spotVisibility /
            distanceSquared;

        result += blinnPhong(
            N, V, L, radiance,
            albedoColor, specular, shininess
        );
    }

    // Свечение не зависит от модели освещения Блинна — Фонга.
    result += emissive * emissiveStrength;

    // Простое тональное отображение.
    result = result / (vec3(1.0) + result);

    outColor = vec4(result, material.albedo.a);
}
