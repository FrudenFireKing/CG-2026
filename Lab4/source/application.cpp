
#include "application.hpp"
#include "graphics_internal.hpp"

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/constants.hpp>
#include <imgui.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>

namespace application {
namespace {

using graphics::internal::context;

constexpr uint32_t MAX_LIGHTS = 64;
constexpr uint32_t SPHERE_SEGMENTS = 32;
constexpr uint32_t SPHERE_RINGS = 20;
constexpr uint32_t SPHERE_COUNT = 3;
// Слоты динамического uniform-буфера моделей: по одному на каждую
// сферу, по одному на маркер каждого источника света и один на пол.
constexpr uint32_t MAX_MODEL_SLOTS = SPHERE_COUNT + 1 + MAX_LIGHTS * 2;

// Карта теней: разрешение и формат. Формат D32_SFLOAT даёт
// достаточную точность глубины для сравнивающего сэмплера.
constexpr uint32_t SHADOW_MAP_SIZE = 2048;
constexpr VkFormat SHADOW_FORMAT = VK_FORMAT_D32_SFLOAT;

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;
};

struct alignas(16) SceneUBO {
    glm::mat4 viewProjection;
    glm::vec4 cameraPosition;
    glm::vec4 ambientColor;
    glm::vec4 directionalDirection;
    glm::vec4 directionalColorIntensity;
    glm::uvec4 lightCounts;
    glm::mat4 sunViewProjection;
    glm::mat4 spotViewProjection;
    glm::vec4 shadowSettings;
};

struct alignas(16) ModelUBO {
    glm::mat4 model;
    glm::vec4 albedo;
    glm::vec4 specularColor;
    // x = shininess, y = сила зеркальной составляющей,
    // z = сила свечения (emissive).
    glm::vec4 materialParameters;
};

struct alignas(16) PointLightGPU {
    glm::vec4 position;
    glm::vec4 colorIntensity;
};

struct alignas(16) SpotLightGPU {
    glm::vec4 position;
    glm::vec4 direction;
    glm::vec4 colorIntensity;
    glm::vec4 cone;
};

struct DrawPush {
    glm::vec4 overrideColor;
    float time;
    float textureEffectStrength;
    uint32_t unlit;
    uint32_t textureMode;
};

struct Buffer {
    VkBuffer handle = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    void* mapped = nullptr;
};

struct Texture {
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
};

// Карта теней: изображение глубины с точки зрения источника света.
struct ShadowMap {
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
};

struct Material {
    const char* name;
    const char* albedoPath;
    const char* specularPath;
    const char* emissivePath;
    Texture albedo;
    Texture specular;
    Texture emissive;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
};

struct SceneObject {
    glm::vec3 position;
    glm::vec3 rotation;
    float scale;
    glm::vec3 albedo;
    glm::vec3 specularColor;
    float shininess;
    float specularStrength;
    float emissiveStrength;
    uint32_t materialIndex;
};

struct PointLight {
    glm::vec3 position{2.0f, 1.0f, 1.5f};
    glm::vec3 color{1.0f, 0.75f, 0.45f};
    float intensity = 15.0f;
};

struct SpotLight {
    glm::vec3 position{-2.0f, 2.0f, 2.0f};
    glm::vec3 direction{0.5f, -0.6f, -0.6f};
    glm::vec3 color{0.45f, 0.65f, 1.0f};
    float intensity = 25.0f;
    float innerAngle = 18.0f;
    float outerAngle = 28.0f;
};

Buffer vertexBuffer;
Buffer indexBuffer;
Buffer floorVertexBuffer;
Buffer floorIndexBuffer;
Buffer sceneBuffer;
Buffer modelBuffer;
Buffer pointLightsBuffer;
Buffer spotLightsBuffer;

uint32_t indexCount = 0;
uint32_t floorIndexCount = 0;
VkDeviceSize modelUniformStride = 0;

VkCommandPool uploadCommandPool = VK_NULL_HANDLE;
VkSampler textureSampler = VK_NULL_HANDLE;
Texture blackTexture;

// Ресурсы теней: две карты (солнце и первый прожектор),
// сэмплер сравнения и отдельный графический конвейер с
// единственным вершинным шейдером для динамического рендеринга.
std::array<ShadowMap, 2> shadowMaps;
VkSampler shadowSampler = VK_NULL_HANDLE;
VkPipelineLayout shadowPipelineLayout = VK_NULL_HANDLE;
VkPipeline shadowPipeline = VK_NULL_HANDLE;

// Матрицы "источник света -> клип-пространство" для текущего кадра.
glm::mat4 sunViewProjection{1.0f};
glm::mat4 spotViewProjection{1.0f};

// Материал Stone не содержит specular- и emissive-карт: для них
// подставляется чёрная служебная текстура 1x1.
std::array<Material, 3> materials = {{
    {
        .name = "Polished metal",
        .albedoPath = "textures/metal_albedo.png",
        .specularPath = "textures/metal_specular.png",
        .emissivePath = nullptr,
        .albedo = {},
        .specular = {},
        .emissive = {},
        .descriptorSet = VK_NULL_HANDLE
    },
    {
        .name = "Neon shell",
        .albedoPath = "textures/neon_albedo.png",
        .specularPath = nullptr,
        .emissivePath = "textures/neon_emissive.png",
        .albedo = {},
        .specular = {},
        .emissive = {},
        .descriptorSet = VK_NULL_HANDLE
    },
    {
        .name = "Stone",
        .albedoPath = "textures/stone_albedo.png",
        .specularPath = nullptr,
        .emissivePath = nullptr,
        .albedo = {},
        .specular = {},
        .emissive = {},
        .descriptorSet = VK_NULL_HANDLE
    }
}};

std::array<SceneObject, SPHERE_COUNT> spheres = {{
    {
        .position = glm::vec3(-1.35f, 0.0f, 0.0f),
        .rotation = glm::vec3(0.0f, 0.0f, 0.0f),
        .scale = 0.85f,
        .albedo = glm::vec3(0.92f, 0.92f, 0.96f),
        .specularColor = glm::vec3(1.0f, 0.96f, 0.88f),
        .shininess = 48.0f,
        .specularStrength = 1.0f,
        .emissiveStrength = 1.0f,
        .materialIndex = 0
    },
    {
        .position = glm::vec3(0.0f, 0.05f, -0.35f),
        .rotation = glm::vec3(0.0f, 0.0f, 0.0f),
        .scale = 0.95f,
        .albedo = glm::vec3(0.95f, 0.95f, 0.95f),
        .specularColor = glm::vec3(0.85f, 0.92f, 1.0f),
        .shininess = 64.0f,
        .specularStrength = 0.8f,
        .emissiveStrength = 1.2f,
        .materialIndex = 1
    },
    {
        .position = glm::vec3(1.35f, 0.0f, -0.15f),
        .rotation = glm::vec3(0.0f, 0.0f, 0.0f),
        .scale = 0.8f,
        .albedo = glm::vec3(1.0f, 1.0f, 1.0f),
        .specularColor = glm::vec3(0.6f, 0.6f, 0.6f),
        .shininess = 32.0f,
        .specularStrength = 1.0f,
        .emissiveStrength = 1.0f,
        .materialIndex = 2
    }
}};

VkDescriptorSetLayout sceneSetLayout = VK_NULL_HANDLE;
VkDescriptorSetLayout modelSetLayout = VK_NULL_HANDLE;
VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
VkDescriptorSet sceneSet = VK_NULL_HANDLE;

VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
VkPipeline pipeline = VK_NULL_HANDLE;

std::vector<PointLight> pointLights;
std::vector<SpotLight> spotLights;

glm::vec3 cameraPosition{0.0f, 0.0f, 5.0f};
glm::vec3 cameraTarget{0.0f, 0.0f, 0.0f};
float cameraFov = 55.0f;

glm::vec3 ambientColor{0.18f, 0.20f, 0.24f};
float ambientIntensity = 0.25f;

glm::vec3 directionalDirection{-0.5f, -1.0f, -0.4f};
glm::vec3 directionalColor{1.0f, 0.95f, 0.85f};
float directionalIntensity = 1.0f;

// Параметры теней, управляемые из ImGui.
bool sunShadowsEnabled = true;
bool spotShadowsEnabled = true;
float sunShadowHalfExtent = 4.0f;
float shadowDepthBiasConstant = 0.0015f;
float shadowDepthBiasSlope = 3.0f;

// Анимация траектории Лиссажу, сохранена из предыдущих работ.
bool animationPlaying = true;
float animationSpeed = 1.0f;
glm::vec3 trajectoryAmplitude{0.25f, 0.30f, 0.25f};
glm::vec3 trajectoryFrequency{1.0f, 2.0f, 1.5f};
glm::vec3 rotationSpeed{0.0f, 22.0f, 0.0f};
double previousFrameTime = 0.0;
float animationTime = 0.0f;

// Текстурный режим и параметры анимированного сэмплирования.
bool textureMode = true;
bool textureEffectEnabled = true;
float textureEffectStrength = 0.65f;

Buffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage)
{
    Buffer result{};

    VkBufferCreateInfo bufferInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage
    };

    VmaAllocationCreateInfo allocationInfo{};
    allocationInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocationInfo.flags =
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
        VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo allocationResult{};

    if (vmaCreateBuffer(
            context.allocator,
            &bufferInfo,
            &allocationInfo,
            &result.handle,
            &result.allocation,
            &allocationResult) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create GPU buffer");
    }

    result.mapped = allocationResult.pMappedData;
    return result;
}

void destroyBuffer(Buffer& buffer)
{
    if (buffer.handle != VK_NULL_HANDLE) {
        vmaDestroyBuffer(
            context.allocator,
            buffer.handle,
            buffer.allocation
        );
    }

    buffer = {};
}

void writeBuffer(const Buffer& buffer, const void* data, size_t size)
{
    if (!buffer.mapped) {
        throw std::runtime_error("GPU buffer is not mapped");
    }

    std::memcpy(buffer.mapped, data, size);

    if (vmaFlushAllocation(
            context.allocator,
            buffer.allocation,
            0,
            size) != VK_SUCCESS) {
        throw std::runtime_error("Failed to flush GPU buffer");
    }
}

void writeBufferAt(
    const Buffer& buffer,
    VkDeviceSize offset,
    const void* data,
    size_t size
)
{
    if (!buffer.mapped) {
        throw std::runtime_error("GPU buffer is not mapped");
    }

    auto* destination =
        static_cast<std::byte*>(buffer.mapped) + offset;

    std::memcpy(destination, data, size);

    if (vmaFlushAllocation(
            context.allocator,
            buffer.allocation,
            offset,
            size) != VK_SUCCESS) {
        throw std::runtime_error("Failed to flush GPU buffer");
    }
}

void createUploadCommandPool()
{
    VkCommandPoolCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = context.graphics_queue_index
    };

    if (vkCreateCommandPool(
            context.device, &info, nullptr,
            &uploadCommandPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create upload command pool");
    }
}

// Однократная отправка команды копирования в графовую очередь.
// Используется для загрузки геометрии и изображений в память GPU.
template<class Record>
void submitUpload(Record&& record)
{
    VkCommandBufferAllocateInfo allocateInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = uploadCommandPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
    };

    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;

    if (vkAllocateCommandBuffers(
            context.device, &allocateInfo,
            &commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate upload command buffer");
    }

    VkCommandBufferBeginInfo beginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };

    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("Failed to begin upload command buffer");
    }

    record(commandBuffer);

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to end upload command buffer");
    }

    VkSubmitInfo submitInfo{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &commandBuffer
    };

    if (vkQueueSubmit(
            context.graphics_queue, 1, &submitInfo,
            VK_NULL_HANDLE) != VK_SUCCESS ||
        vkQueueWaitIdle(context.graphics_queue) != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit upload command buffer");
    }

    vkFreeCommandBuffers(
        context.device, uploadCommandPool, 1, &commandBuffer);
}

// Буфер в памяти GPU: данные проходят через промежуточный staging
// buffer в хостовой памяти и копируются командой vkCmdCopyBuffer.
Buffer createDeviceBuffer(
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    const void* data
)
{
    Buffer staging = createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

    writeBuffer(staging, data, static_cast<size_t>(size));

    Buffer result{};

    VkBufferCreateInfo bufferInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT
    };

    VmaAllocationCreateInfo allocationInfo{};
    allocationInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocationInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    if (vmaCreateBuffer(
            context.allocator,
            &bufferInfo,
            &allocationInfo,
            &result.handle,
            &result.allocation,
            nullptr) != VK_SUCCESS) {
        destroyBuffer(staging);
        throw std::runtime_error("Failed to create device-local buffer");
    }

    submitUpload([&](VkCommandBuffer commandBuffer) {
        VkBufferCopy region{};
        region.size = size;

        vkCmdCopyBuffer(
            commandBuffer, staging.handle, result.handle, 1, &region);

        VkBufferMemoryBarrier barrier{
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = static_cast<VkAccessFlags>(
                (usage & VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
                    ? VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT : 0) |
                (usage & VK_BUFFER_USAGE_INDEX_BUFFER_BIT
                    ? VK_ACCESS_INDEX_READ_BIT : 0)),
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = result.handle,
            .offset = 0,
            .size = size
        };

        vkCmdPipelineBarrier(
            commandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
            0, 0, nullptr, 1, &barrier, 0, nullptr);
    });

    destroyBuffer(staging);
    return result;
}

void destroyTexture(Texture& texture)
{
    if (texture.view != VK_NULL_HANDLE) {
        vkDestroyImageView(context.device, texture.view, nullptr);
    }

    if (texture.image != VK_NULL_HANDLE) {
        vmaDestroyImage(context.allocator, texture.image, texture.allocation);
    }

    texture = {};
}

// Изображение в памяти GPU из пикселей в памяти хоста: staging buffer,
// переходы раскладок UNDEFINED -> TRANSFER_DST -> SHADER_READ_ONLY.
Texture createTextureFromPixels(
    const unsigned char* pixels,
    uint32_t width,
    uint32_t height,
    bool srgb
)
{
    const VkDeviceSize size =
        static_cast<VkDeviceSize>(width) * height * 4;

    Buffer staging = createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

    writeBuffer(staging, pixels, static_cast<size_t>(size));

    Texture result{};

    VkImageCreateInfo imageInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = srgb
            ? VK_FORMAT_R8G8B8A8_SRGB
            : VK_FORMAT_R8G8B8A8_UNORM,
        .extent = {width, height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage =
            VK_IMAGE_USAGE_TRANSFER_DST_BIT |
            VK_IMAGE_USAGE_SAMPLED_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
    };

    VmaAllocationCreateInfo allocationInfo{};
    allocationInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocationInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    if (vmaCreateImage(
            context.allocator,
            &imageInfo,
            &allocationInfo,
            &result.image,
            &result.allocation,
            nullptr) != VK_SUCCESS) {
        destroyBuffer(staging);
        throw std::runtime_error("Failed to create texture image");
    }

    submitUpload([&](VkCommandBuffer commandBuffer) {
        VkImageMemoryBarrier barrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = result.image,
            .subresourceRange = {
                VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1
            }
        };

        vkCmdPipelineBarrier(
            commandBuffer,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrier);

        VkBufferImageCopy copy{
            .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .imageExtent = {width, height, 1}
        };

        vkCmdCopyBufferToImage(
            commandBuffer,
            staging.handle,
            result.image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &copy);

        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        vkCmdPipelineBarrier(
            commandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrier);
    });

    destroyBuffer(staging);

    VkImageViewCreateInfo viewInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = result.image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = imageInfo.format,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}
    };

    if (vkCreateImageView(
            context.device, &viewInfo, nullptr,
            &result.view) != VK_SUCCESS) {
        destroyTexture(result);
        throw std::runtime_error("Failed to create texture image view");
    }

    return result;
}

Texture createSolidTexture(const std::array<unsigned char, 4>& pixel)
{
    return createTextureFromPixels(pixel.data(), 1, 1, true);
}

// Загрузка PNG-файла с диска через stb_image.
Texture loadTextureFile(const std::string& path, bool srgb)
{
    const std::array<std::string, 6> prefixes = {
        "",
        "Lab4/",
        "build-mingw-debug/Lab4/",
        "build-mingw-release/Lab4/",
        "build-vs-debug/Lab4/",
        "build-vs-release/Lab4/"
    };

    int width = 0;
    int height = 0;
    int components = 0;

    unsigned char* pixels = nullptr;

    for (const auto& prefix : prefixes) {
        pixels = stbi_load(
            (prefix + path).c_str(),
            &width, &height, &components, 4);

        if (pixels != nullptr) {
            break;
        }
    }

    if (pixels == nullptr || width <= 0 || height <= 0) {
        throw std::runtime_error(
            "Failed to load texture: " + path + " ("
            + stbi_failure_reason() + ")");
    }

    Texture texture = createTextureFromPixels(
        pixels,
        static_cast<uint32_t>(width),
        static_cast<uint32_t>(height),
        srgb);

    stbi_image_free(pixels);

    std::cout << "Loaded texture: " << path << '\n';
    return texture;
}

std::vector<char> readFile(const std::string& filename)
{
    const std::array<std::string, 8> paths = {
        "shaders/" + filename,
        "Lab4/shaders/" + filename,
        "build-mingw-debug/shaders/" + filename,
        "Lab4/build-mingw-debug/shaders/" + filename,
        "build-mingw-release/shaders/" + filename,
        "Lab4/build-mingw-release/shaders/" + filename,
        "build-vs-debug/shaders/" + filename,
        "Lab4/build-vs-debug/shaders/" + filename
    };

    for (const auto& path : paths) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);

        if (!file) {
            continue;
        }

        const auto size = file.tellg();

        if (size <= 0) {
            continue;
        }

        std::vector<char> data(static_cast<size_t>(size));
        file.seekg(0);
        file.read(data.data(), static_cast<std::streamsize>(size));

        if (file) {
            std::cout << "Loaded shader: " << path << '\n';
            return data;
        }
    }

    throw std::runtime_error(
        "Cannot find " + filename +
        ". Check shader compilation and working directory."
    );
}

VkShaderModule createShaderModule(const std::vector<char>& code)
{
    if (code.empty() || code.size() % sizeof(uint32_t) != 0) {
        throw std::runtime_error("Invalid SPIR-V shader file");
    }

    VkShaderModuleCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = code.size(),
        .pCode = reinterpret_cast<const uint32_t*>(code.data())
    };

    VkShaderModule module = VK_NULL_HANDLE;

    if (vkCreateShaderModule(
            context.device, &info, nullptr, &module) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shader module");
    }

    return module;
}

// Процедурная сфера: позиция, нормаль и текстурные координаты
// (сферическая развёртка). Геометрия загружается в память GPU
// через staging buffer.
void createSphere()
{
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    vertices.reserve(
        (SPHERE_RINGS + 1) * (SPHERE_SEGMENTS + 1));

    for (uint32_t ring = 0; ring <= SPHERE_RINGS; ++ring) {
        const float v =
            static_cast<float>(ring) / static_cast<float>(SPHERE_RINGS);
        const float theta = v * glm::pi<float>();

        for (uint32_t segment = 0;
             segment <= SPHERE_SEGMENTS;
             ++segment) {
            const float u =
                static_cast<float>(segment) /
                static_cast<float>(SPHERE_SEGMENTS);

            const float phi = u * glm::two_pi<float>();

            glm::vec3 normal{
                std::sin(theta) * std::cos(phi),
                std::cos(theta),
                std::sin(theta) * std::sin(phi)
            };

            vertices.push_back(
                {normal, glm::normalize(normal), glm::vec2(u, 1.0f - v)});
        }
    }

    for (uint32_t ring = 0; ring < SPHERE_RINGS; ++ring) {
        for (uint32_t segment = 0;
             segment < SPHERE_SEGMENTS;
             ++segment) {
            const uint32_t a =
                ring * (SPHERE_SEGMENTS + 1) + segment;
            const uint32_t b = a + SPHERE_SEGMENTS + 1;

            indices.push_back(a);
            indices.push_back(b);
            indices.push_back(a + 1);

            indices.push_back(a + 1);
            indices.push_back(b);
            indices.push_back(b + 1);
        }
    }

    vertexBuffer = createDeviceBuffer(
        vertices.size() * sizeof(Vertex),
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        vertices.data());

    indexBuffer = createDeviceBuffer(
        indices.size() * sizeof(uint32_t),
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        indices.data());

    indexCount = static_cast<uint32_t>(indices.size());
}

// Пол под сценой: единичный квадрт с повторением текстуры.
// Тень от сфер должна на что-то падать, поэтому пол обязателен
// по условию работы.
void createFloorMesh()
{
    constexpr float edge = 4.0f;
    constexpr float height = -1.0f;
    constexpr float uvRepeat = 3.0f;

    const std::array<Vertex, 4> vertices = {{
        {{-edge, height, -edge}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
        {{ edge, height, -edge}, {0.0f, 1.0f, 0.0f}, {uvRepeat, 0.0f}},
        {{ edge, height,  edge}, {0.0f, 1.0f, 0.0f}, {uvRepeat, uvRepeat}},
        {{-edge, height,  edge}, {0.0f, 1.0f, 0.0f}, {0.0f, uvRepeat}}
    }};

    const std::array<uint32_t, 6> indices = {0, 2, 1, 0, 3, 2};

    floorVertexBuffer = createDeviceBuffer(
        sizeof(Vertex) * vertices.size(),
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        vertices.data());

    floorIndexBuffer = createDeviceBuffer(
        sizeof(uint32_t) * indices.size(),
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        indices.data());

    floorIndexCount = static_cast<uint32_t>(indices.size());
}

// Создание карт теней и сэмплера сравнения глубины.
// Изображение используется одновременно как attachment глубины
// (запись в теневом проходе) и как текстура с возможностью
// сэмплирования с сравнением (основной проход).
void createShadowResources()
{
    VkFormatProperties formatProperties{};
    vkGetPhysicalDeviceFormatProperties(
        context.physical_device, SHADOW_FORMAT, &formatProperties);

    constexpr VkFormatFeatureFlags requiredFeatures =
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT |
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;

    if ((formatProperties.optimalTilingFeatures & requiredFeatures)
            != requiredFeatures) {
        throw std::runtime_error(
            "D32_SFLOAT cannot be used as a sampled depth attachment");
    }

    const VkImageCreateInfo imageInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = SHADOW_FORMAT,
        .extent = {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage =
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
            VK_IMAGE_USAGE_SAMPLED_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
    };

    VmaAllocationCreateInfo allocationInfo{};
    allocationInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocationInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    for (ShadowMap& map : shadowMaps) {
        if (vmaCreateImage(
                context.allocator,
                &imageInfo,
                &allocationInfo,
                &map.image,
                &map.allocation,
                nullptr) != VK_SUCCESS) {
            throw std::runtime_error(
                "Failed to create shadow map image");
        }

        const VkImageViewCreateInfo viewInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = map.image,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = SHADOW_FORMAT,
            .subresourceRange = {
                VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1
            }
        };

        if (vkCreateImageView(
                context.device, &viewInfo, nullptr,
                &map.view) != VK_SUCCESS) {
            throw std::runtime_error(
                "Failed to create shadow map view");
        }
    }

    // Сэмплер с включённым сравнением глубины: результат выборки
    // texture() в шейдере - 1.0, если референсная глубина меньше
    // глубины из карты (оператор "меньше чем"), иначе 0.0.
    const VkSamplerCreateInfo samplerInfo{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_NEAREST,
        .minFilter = VK_FILTER_NEAREST,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
        .compareEnable = VK_TRUE,
        .compareOp = VK_COMPARE_OP_LESS,
        .maxLod = 0.0f,
        .borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE
    };

    if (vkCreateSampler(
            context.device, &samplerInfo, nullptr,
            &shadowSampler) != VK_SUCCESS) {
        throw std::runtime_error(
            "Failed to create shadow comparison sampler");
    }
}

void createDescriptorLayouts()
{
    // Набор сцены: UBO с параметрами камеры и освещения, два SSBO с
    // источниками света и два сравнивающих сэмплера карт теней
    // (binding 3 - солнце, binding 4 - первый прожектор).
    const std::array<VkDescriptorSetLayoutBinding, 5> sceneBindings = {{
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .stageFlags =
                VK_SHADER_STAGE_VERTEX_BIT |
                VK_SHADER_STAGE_FRAGMENT_BIT
        },
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT
        },
        {
            .binding = 2,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT
        },
        {
            .binding = 3,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT
        },
        {
            .binding = 4,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT
        }
    }};

    VkDescriptorSetLayoutCreateInfo sceneInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = static_cast<uint32_t>(sceneBindings.size()),
        .pBindings = sceneBindings.data()
    };

    if (vkCreateDescriptorSetLayout(
            context.device, &sceneInfo, nullptr,
            &sceneSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create scene layout");
    }

    // Набор модели: динамический uniform-буфер и три
    // combined image sampler (albedo, specular, emissive).
    const std::array<VkDescriptorSetLayoutBinding, 4> modelBindings = {{
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
            .descriptorCount = 1,
            .stageFlags =
                VK_SHADER_STAGE_VERTEX_BIT |
                VK_SHADER_STAGE_FRAGMENT_BIT
        },
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT
        },
        {
            .binding = 2,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT
        },
        {
            .binding = 3,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT
        }
    }};

    VkDescriptorSetLayoutCreateInfo modelInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = static_cast<uint32_t>(modelBindings.size()),
        .pBindings = modelBindings.data()
    };

    if (vkCreateDescriptorSetLayout(
            context.device, &modelInfo, nullptr,
            &modelSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create model layout");
    }
}

void createMaterialTextures()
{
    // Чёрная служебная текстура 1x1 подставляется вместо
    // отсутствующих specular- и emissive-карт.
    blackTexture = createSolidTexture({0, 0, 0, 255});

    for (Material& material : materials) {
        material.albedo = loadTextureFile(material.albedoPath, true);

        if (material.specularPath != nullptr) {
            // Карты данных загружаются в линейном (UNORM) формате.
            material.specular =
                loadTextureFile(material.specularPath, false);
        }

        if (material.emissivePath != nullptr) {
            material.emissive =
                loadTextureFile(material.emissivePath, true);
        }
    }

    VkSamplerCreateInfo samplerInfo{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .maxAnisotropy = 1.0f,
        .minLod = 0.0f,
        .maxLod = 0.0f
    };

    if (vkCreateSampler(
            context.device, &samplerInfo, nullptr,
            &textureSampler) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create texture sampler");
    }
}

void createDescriptors()
{
    sceneBuffer = createBuffer(
        sizeof(SceneUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    // Один динамический uniform-буфер на все модели и маркеры:
    // размер слота выравнивается по minUniformBufferOffsetAlignment.
    VkPhysicalDeviceProperties deviceProperties{};
    vkGetPhysicalDeviceProperties(
        context.physical_device, &deviceProperties);

    const VkDeviceSize alignment = std::max<VkDeviceSize>(
        1, deviceProperties.limits.minUniformBufferOffsetAlignment);

    modelUniformStride =
        (sizeof(ModelUBO) + alignment - 1) / alignment * alignment;

    modelBuffer = createBuffer(
        modelUniformStride * MAX_MODEL_SLOTS,
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    pointLightsBuffer = createBuffer(
        sizeof(PointLightGPU) * MAX_LIGHTS,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    spotLightsBuffer = createBuffer(
        sizeof(SpotLightGPU) * MAX_LIGHTS,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    const std::array<VkDescriptorPoolSize, 4> poolSizes = {{
        {
            .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1
        },
        {
            .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
            .descriptorCount = static_cast<uint32_t>(materials.size())
        },
        {
            .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 2
        },
        {
            .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = static_cast<uint32_t>(
                materials.size() * 3 + shadowMaps.size())
        }
    }};

    VkDescriptorPoolCreateInfo poolInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = static_cast<uint32_t>(materials.size() + 1),
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data()
    };

    if (vkCreateDescriptorPool(
            context.device, &poolInfo, nullptr,
            &descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor pool");
    }

    const std::array<VkDescriptorSetLayout, materials.size() + 1> layouts =
        {sceneSetLayout, modelSetLayout, modelSetLayout, modelSetLayout};

    std::array<VkDescriptorSet, materials.size() + 1> sets{};

    VkDescriptorSetAllocateInfo allocateInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptorPool,
        .descriptorSetCount = static_cast<uint32_t>(sets.size()),
        .pSetLayouts = layouts.data()
    };

    if (vkAllocateDescriptorSets(
            context.device, &allocateInfo, sets.data()) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate descriptor sets");
    }

    sceneSet = sets[0];

    for (size_t i = 0; i < materials.size(); ++i) {
        materials[i].descriptorSet = sets[i + 1];
    }

    VkDescriptorBufferInfo sceneBufferInfo{
        .buffer = sceneBuffer.handle,
        .offset = 0,
        .range = sizeof(SceneUBO)
    };

    VkDescriptorBufferInfo pointBufferInfo{
        .buffer = pointLightsBuffer.handle,
        .offset = 0,
        .range = sizeof(PointLightGPU) * MAX_LIGHTS
    };

    VkDescriptorBufferInfo spotBufferInfo{
        .buffer = spotLightsBuffer.handle,
        .offset = 0,
        .range = sizeof(SpotLightGPU) * MAX_LIGHTS
    };

    // Карты теней привязываются вместе с сэмплером сравнения.
    // Раскладка DEPTH_READ_ONLY_OPTIMAL устанавливается барьером
    // после каждого теневого прохода.
    const std::array<VkDescriptorImageInfo, 2> shadowImageInfos = {{
        {
            .sampler = shadowSampler,
            .imageView = shadowMaps[0].view,
            .imageLayout = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL
        },
        {
            .sampler = shadowSampler,
            .imageView = shadowMaps[1].view,
            .imageLayout = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL
        }
    }};

    // Все наборы материалов ссылаются на один и тот же динамический
    // uniform-буфер: экземпляр модели выбирается динамическим смещением.
    VkDescriptorBufferInfo modelBufferInfo{
        .buffer = modelBuffer.handle,
        .offset = 0,
        .range = sizeof(ModelUBO)
    };

    const std::array<VkWriteDescriptorSet, 5> sceneWrites = {{
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = sceneSet,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .pBufferInfo = &sceneBufferInfo
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = sceneSet,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &pointBufferInfo
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = sceneSet,
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &spotBufferInfo
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = sceneSet,
            .dstBinding = 3,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &shadowImageInfos[0]
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = sceneSet,
            .dstBinding = 4,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &shadowImageInfos[1]
        }
    }};

    vkUpdateDescriptorSets(
        context.device,
        static_cast<uint32_t>(sceneWrites.size()),
        sceneWrites.data(), 0, nullptr);

    for (size_t i = 0; i < materials.size(); ++i) {
        Material& material = materials[i];

        VkWriteDescriptorSet modelWrite{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = material.descriptorSet,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
            .pBufferInfo = &modelBufferInfo
        };

        std::array<VkDescriptorImageInfo, 3> imageInfos = {{
            {
                .sampler = textureSampler,
                .imageView = material.albedo.view,
                .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
            },
            {
                .sampler = textureSampler,
                .imageView = material.specularPath != nullptr
                    ? material.specular.view
                    : blackTexture.view,
                .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
            },
            {
                .sampler = textureSampler,
                .imageView = material.emissivePath != nullptr
                    ? material.emissive.view
                    : blackTexture.view,
                .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
            }
        }};

        std::array<VkWriteDescriptorSet, 4> materialWrites = {{
            modelWrite,
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = material.descriptorSet,
                .dstBinding = 1,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = &imageInfos[0]
            },
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = material.descriptorSet,
                .dstBinding = 2,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = &imageInfos[1]
            },
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = material.descriptorSet,
                .dstBinding = 3,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = &imageInfos[2]
            }
        }};

        vkUpdateDescriptorSets(
            context.device,
            static_cast<uint32_t>(materialWrites.size()),
            materialWrites.data(), 0, nullptr);
    }
}

void createPipeline()
{
    const auto vertexCode = readFile("vert.spv");
    const auto fragmentCode = readFile("frag.spv");

    VkShaderModule vertexModule = createShaderModule(vertexCode);
    VkShaderModule fragmentModule = VK_NULL_HANDLE;

    try {
        fragmentModule = createShaderModule(fragmentCode);

        const std::array<VkPipelineShaderStageCreateInfo, 2> stages = {{
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_VERTEX_BIT,
                .module = vertexModule,
                .pName = "main"
            },
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                .module = fragmentModule,
                .pName = "main"
            }
        }};

        VkVertexInputBindingDescription vertexBinding{
            .binding = 0,
            .stride = sizeof(Vertex),
            .inputRate = VK_VERTEX_INPUT_RATE_VERTEX
        };

        const std::array<VkVertexInputAttributeDescription, 3> attributes = {{
            {
                .location = 0,
                .binding = 0,
                .format = VK_FORMAT_R32G32B32_SFLOAT,
                .offset = offsetof(Vertex, position)
            },
            {
                .location = 1,
                .binding = 0,
                .format = VK_FORMAT_R32G32B32_SFLOAT,
                .offset = offsetof(Vertex, normal)
            },
            {
                .location = 2,
                .binding = 0,
                .format = VK_FORMAT_R32G32_SFLOAT,
                .offset = offsetof(Vertex, uv)
            }
        }};

        VkPipelineVertexInputStateCreateInfo vertexInput{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &vertexBinding,
            .vertexAttributeDescriptionCount = 3,
            .pVertexAttributeDescriptions = attributes.data()
        };

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
        };

        VkPipelineViewportStateCreateInfo viewportState{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .viewportCount = 1,
            .scissorCount = 1
        };

        VkPipelineRasterizationStateCreateInfo rasterizer{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .polygonMode = VK_POLYGON_MODE_FILL,
            .cullMode = VK_CULL_MODE_NONE,
            .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
            .lineWidth = 1.0f
        };

        VkPipelineMultisampleStateCreateInfo multisampling{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT
        };

        VkPipelineDepthStencilStateCreateInfo depthStencil{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            .depthTestEnable = VK_TRUE,
            .depthWriteEnable = VK_TRUE,
            .depthCompareOp = VK_COMPARE_OP_LESS
        };

        VkPipelineColorBlendAttachmentState blendAttachment{};
        blendAttachment.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT |
            VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT |
            VK_COLOR_COMPONENT_A_BIT;

        VkPipelineColorBlendStateCreateInfo blending{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .attachmentCount = 1,
            .pAttachments = &blendAttachment
        };

        const std::array<VkDynamicState, 2> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };

        VkPipelineDynamicStateCreateInfo dynamicState{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            .dynamicStateCount = 2,
            .pDynamicStates = dynamicStates.data()
        };

        const std::array<VkDescriptorSetLayout, 2> setLayouts = {
            sceneSetLayout, modelSetLayout
        };

        VkPushConstantRange pushRange{
            .stageFlags =
                VK_SHADER_STAGE_VERTEX_BIT |
                VK_SHADER_STAGE_FRAGMENT_BIT,
            .offset = 0,
            .size = sizeof(DrawPush)
        };

        VkPipelineLayoutCreateInfo layoutInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 2,
            .pSetLayouts = setLayouts.data(),
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pushRange
        };

        if (vkCreatePipelineLayout(
                context.device, &layoutInfo, nullptr,
                &pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create pipeline layout");
        }

        VkGraphicsPipelineCreateInfo pipelineInfo{
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .stageCount = 2,
            .pStages = stages.data(),
            .pVertexInputState = &vertexInput,
            .pInputAssemblyState = &inputAssembly,
            .pViewportState = &viewportState,
            .pRasterizationState = &rasterizer,
            .pMultisampleState = &multisampling,
            .pDepthStencilState = &depthStencil,
            .pColorBlendState = &blending,
            .pDynamicState = &dynamicState,
            .layout = pipelineLayout,
            .renderPass = context.render_pass,
            .subpass = 0
        };

        if (vkCreateGraphicsPipelines(
                context.device, VK_NULL_HANDLE, 1,
                &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create graphics pipeline");
        }
    }
    catch (...) {
        if (fragmentModule != VK_NULL_HANDLE) {
            vkDestroyShaderModule(
                context.device, fragmentModule, nullptr);
        }
        vkDestroyShaderModule(context.device, vertexModule, nullptr);
        throw;
    }

    vkDestroyShaderModule(context.device, fragmentModule, nullptr);
    vkDestroyShaderModule(context.device, vertexModule, nullptr);
}

// Теневой конвейер для динамического рендеринга: render pass не
// используется (VK_NULL_HANDLE), формат attachment глубины передаётся
// через VkPipelineRenderingCreateInfo. Фрагментная стадия отсутствует -
// допустимо для конвейеров, рисующих только в карту глубины.
void createShadowPipeline()
{
    const VkPushConstantRange pushRange{
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        .offset = 0,
        .size = sizeof(glm::mat4)
    };

    VkPipelineLayoutCreateInfo layoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pushRange
    };

    if (vkCreatePipelineLayout(
            context.device, &layoutInfo, nullptr,
            &shadowPipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error(
            "Failed to create shadow pipeline layout");
    }

    const auto vertexCode = readFile("shadow.spv");

    VkShaderModule vertexModule = createShaderModule(vertexCode);

    const VkPipelineShaderStageCreateInfo stage{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT,
        .module = vertexModule,
        .pName = "main"
    };

    // Из всех вершинных атрибутов теневому шейдеру нужна только позиция.
    VkVertexInputBindingDescription vertexBinding{
        .binding = 0,
        .stride = sizeof(Vertex),
        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX
    };

    const std::array<VkVertexInputAttributeDescription, 1> attributes = {{
        {
            .location = 0,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32_SFLOAT,
            .offset = offsetof(Vertex, position)
        }
    }};

    VkPipelineVertexInputStateCreateInfo vertexInput{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &vertexBinding,
        .vertexAttributeDescriptionCount = 1,
        .pVertexAttributeDescriptions = attributes.data()
    };

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
    };

    VkPipelineViewportStateCreateInfo viewportState{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .scissorCount = 1
    };

    VkPipelineRasterizationStateCreateInfo rasterizer{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .lineWidth = 1.0f
    };

    VkPipelineMultisampleStateCreateInfo multisampling{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT
    };

    VkPipelineDepthStencilStateCreateInfo depthStencil{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = VK_TRUE,
        .depthWriteEnable = VK_TRUE,
        .depthCompareOp = VK_COMPARE_OP_LESS
    };

    // Формат attachment глубины вместо VkRenderPass.
    const VkPipelineRenderingCreateInfo renderingInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .colorAttachmentCount = 0,
        .depthAttachmentFormat = SHADOW_FORMAT
    };

    // Сдвиг глубины настраивается командой vkCmdSetDepthBias
    // из значений ImGui, поэтому состояние динамическое.
    const std::array<VkDynamicState, 3> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
        VK_DYNAMIC_STATE_DEPTH_BIAS
    };

    VkPipelineDynamicStateCreateInfo dynamicState{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
        .pDynamicStates = dynamicStates.data()
    };

    VkGraphicsPipelineCreateInfo pipelineInfo{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &renderingInfo,
        .stageCount = 1,
        .pStages = &stage,
        .pVertexInputState = &vertexInput,
        .pInputAssemblyState = &inputAssembly,
        .pViewportState = &viewportState,
        .pRasterizationState = &rasterizer,
        .pMultisampleState = &multisampling,
        .pDepthStencilState = &depthStencil,
        .pDynamicState = &dynamicState,
        .layout = shadowPipelineLayout,
        .renderPass = VK_NULL_HANDLE,
        .subpass = 0
    };

    if (vkCreateGraphicsPipelines(
            context.device, VK_NULL_HANDLE, 1,
            &pipelineInfo, nullptr, &shadowPipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(context.device, vertexModule, nullptr);
        throw std::runtime_error("Failed to create shadow pipeline");
    }

    vkDestroyShaderModule(context.device, vertexModule, nullptr);
}

glm::vec3 trajectoryOffset(uint32_t sphereIndex)
{
    const float phase =
        static_cast<float>(sphereIndex) * glm::two_pi<float>()
        / static_cast<float>(SPHERE_COUNT);

    // При паузе animationTime не накапливается, поэтому объекты
    // останавливаются в текущей точке траектории.
    return trajectoryAmplitude *
        glm::vec3(
            std::sin(trajectoryFrequency.x * animationTime + phase),
            std::sin(trajectoryFrequency.y * animationTime + phase * 2.0f),
            std::sin(trajectoryFrequency.z * animationTime + phase * 3.0f));
}

glm::vec3 trajectoryRotation()
{
    return rotationSpeed * animationTime;
}

glm::mat4 makeModelMatrix(const SceneObject& object, uint32_t index)
{
    const glm::vec3 position =
        object.position + trajectoryOffset(index);

    const glm::vec3 rotation =
        object.rotation + trajectoryRotation();

    glm::mat4 model = glm::mat4(1.0f);

    model = glm::translate(model, position);

    model = glm::rotate(
        model, glm::radians(rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));

    model = glm::rotate(
        model, glm::radians(rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));

    model = glm::rotate(
        model, glm::radians(rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));

    return glm::scale(model, glm::vec3(object.scale));
}

// Матрица сцены солнца: LookAt вдоль направления направленного
// источника света и ортографическая проекция с площадью,
// регулируемой из ImGui (sunShadowHalfExtent).
glm::mat4 makeSunViewProjection()
{
    glm::vec3 direction = directionalDirection;

    if (glm::length(direction) < 0.0001f) {
        direction = glm::vec3(0.0f, -1.0f, 0.0f);
    }

    direction = glm::normalize(direction);

    const glm::vec3 center{0.0f, -0.25f, 0.0f};
    const glm::vec3 eye = center - direction * 8.0f;

    glm::vec3 up{0.0f, 1.0f, 0.0f};

    if (std::abs(direction.y) > 0.95f) {
        up = glm::vec3(0.0f, 0.0f, 1.0f);
    }

    const glm::mat4 view = glm::lookAt(eye, center, up);

    // Квадратная ортографическая проекция: площадь покрываемой
    // области задается половиной стороны квадрата в метрах сцены.
    const glm::mat4 projection = glm::ortho(
        -sunShadowHalfExtent, sunShadowHalfExtent,
        -sunShadowHalfExtent, sunShadowHalfExtent,
        0.1f, 30.0f);

    return projection * view;
}

// Матрица сцены первого прожектора: перспективная проекция с углом,
// охватывающим внешний конус источника света (дополнительное задание).
glm::mat4 makeSpotViewProjection()
{
    if (spotLights.empty()) {
        return glm::mat4(1.0f);
    }

    const SpotLight& light = spotLights.front();

    glm::vec3 direction = light.direction;

    if (glm::length(direction) < 0.0001f) {
        direction = glm::vec3(0.0f, -1.0f, 0.0f);
    }

    direction = glm::normalize(direction);

    glm::vec3 up{0.0f, 1.0f, 0.0f};

    if (std::abs(direction.y) > 0.95f) {
        up = glm::vec3(0.0f, 0.0f, 1.0f);
    }

    const float fov = std::clamp(
        light.outerAngle * 2.0f, 1.0f, 160.0f);

    const glm::mat4 projection = glm::perspective(
        glm::radians(fov), 1.0f, 0.05f, 30.0f);

    const glm::mat4 view = glm::lookAt(
        light.position,
        light.position + direction,
        up);

    return projection * view;
}

void writeModelSlot(
    uint32_t slot,
    const glm::mat4& modelMatrix,
    const glm::vec4& albedo,
    const glm::vec4& specularColor,
    const glm::vec4& materialParameters
)
{
    ModelUBO model{};
    model.model = modelMatrix;
    model.albedo = albedo;
    model.specularColor = specularColor;
    model.materialParameters = materialParameters;

    writeBufferAt(
        modelBuffer,
        modelUniformStride * slot,
        &model,
        sizeof(model));
}

void updateImGui()
{
    ImGui::Begin("Lab 4 - Shadow mapping");

    if (ImGui::CollapsingHeader("Models and materials",
                                ImGuiTreeNodeFlags_DefaultOpen)) {
        for (uint32_t i = 0; i < SPHERE_COUNT; ++i) {
            SceneObject& sphere = spheres[i];

            ImGui::PushID(static_cast<int>(i));

            const std::string title =
                "Sphere " + std::to_string(i + 1);

            if (ImGui::TreeNode(title.c_str())) {
                ImGui::DragFloat3(
                    "Position", glm::value_ptr(sphere.position), 0.05f);

                ImGui::SliderFloat(
                    "Rotation X", &sphere.rotation.x, 0.0f, 360.0f);
                ImGui::SliderFloat(
                    "Rotation Y", &sphere.rotation.y, 0.0f, 360.0f);
                ImGui::SliderFloat(
                    "Rotation Z", &sphere.rotation.z, 0.0f, 360.0f);

                ImGui::SliderFloat("Scale", &sphere.scale, 0.1f, 2.0f);

                const char* currentMaterial =
                    materials[sphere.materialIndex].name;

                if (ImGui::BeginCombo("Material", currentMaterial)) {
                    for (size_t m = 0; m < materials.size(); ++m) {
                        const bool selected =
                            m == sphere.materialIndex;

                        if (ImGui::Selectable(
                                materials[m].name, selected)) {
                            sphere.materialIndex =
                                static_cast<uint32_t>(m);
                        }

                        if (selected) {
                            ImGui::SetItemDefaultFocus();
                        }
                    }

                    ImGui::EndCombo();
                }

                ImGui::ColorEdit3(
                    "Albedo tint", glm::value_ptr(sphere.albedo));
                ImGui::ColorEdit3(
                    "Specular color",
                    glm::value_ptr(sphere.specularColor));
                ImGui::SliderFloat(
                    "Shininess", &sphere.shininess, 1.0f, 256.0f);
                ImGui::SliderFloat(
                    "Specular strength",
                    &sphere.specularStrength, 0.0f, 2.0f);
                ImGui::SliderFloat(
                    "Emissive strength",
                    &sphere.emissiveStrength, 0.0f, 2.0f);

                ImGui::TreePop();
            }

            ImGui::PopID();
        }

        if (ImGui::CollapsingHeader("Animated sampling")) {
            ImGui::Checkbox(
                "Enable animated sampling", &textureEffectEnabled);
            ImGui::SliderFloat(
                "Effect strength",
                &textureEffectStrength, 0.0f, 1.0f);
            ImGui::TextWrapped(
                "The effect is applied to sphere 1: the same albedo map "
                "is sampled with a second, time-varying set of texture "
                "coordinates and blended with the base color.");
        }
    }

    if (ImGui::CollapsingHeader("Animation",
                                ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Play animation", &animationPlaying);
        ImGui::SliderFloat("Animation speed", &animationSpeed, 0.0f, 4.0f);
        ImGui::DragFloat3(
            "Trajectory amplitude",
            glm::value_ptr(trajectoryAmplitude), 0.02f, 0.0f, 2.0f);
        ImGui::DragFloat3(
            "Trajectory frequency",
            glm::value_ptr(trajectoryFrequency), 0.05f, 0.0f, 8.0f);
        ImGui::DragFloat3(
            "Rotation speed (deg/s)",
            glm::value_ptr(rotationSpeed), 0.5f, -180.0f, 180.0f);
    }

    if (ImGui::CollapsingHeader("Camera",
                                ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::DragFloat3(
            "Camera position", glm::value_ptr(cameraPosition), 0.05f);
        ImGui::DragFloat3(
            "Look at", glm::value_ptr(cameraTarget), 0.05f);
        ImGui::SliderFloat("FOV", &cameraFov, 25.0f, 100.0f);
    }

    if (ImGui::CollapsingHeader("Ambient and directional lighting",
                                ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::ColorEdit3(
            "Ambient color", glm::value_ptr(ambientColor));
        ImGui::SliderFloat(
            "Ambient intensity", &ambientIntensity, 0.0f, 2.0f);

        ImGui::DragFloat3(
            "Directional direction",
            glm::value_ptr(directionalDirection), 0.02f);
        ImGui::ColorEdit3(
            "Directional color", glm::value_ptr(directionalColor));
        ImGui::SliderFloat(
            "Directional intensity", &directionalIntensity, 0.0f, 5.0f);
    }

    if (ImGui::CollapsingHeader("Point lights",
                                ImGuiTreeNodeFlags_DefaultOpen)) {
        if (pointLights.size() < MAX_LIGHTS &&
            ImGui::Button("Add point light")) {
            pointLights.emplace_back();
        }

        for (size_t i = 0; i < pointLights.size();) {
            ImGui::PushID(static_cast<int>(i));
            auto& light = pointLights[i];

            const std::string title =
                "Point light " + std::to_string(i + 1);

            if (ImGui::TreeNode(title.c_str())) {
                ImGui::DragFloat3(
                    "Position", glm::value_ptr(light.position), 0.05f);
                ImGui::ColorEdit3(
                    "Color", glm::value_ptr(light.color));
                ImGui::SliderFloat(
                    "Intensity", &light.intensity, 0.0f, 100.0f);

                if (ImGui::Button("Remove")) {
                    pointLights.erase(pointLights.begin() + i);
                    ImGui::TreePop();
                    ImGui::PopID();
                    continue;
                }

                ImGui::TreePop();
            }

            ImGui::PopID();
            ++i;
        }
    }

    if (ImGui::CollapsingHeader("Spot lights",
                                ImGuiTreeNodeFlags_DefaultOpen)) {
        if (spotLights.size() < MAX_LIGHTS &&
            ImGui::Button("Add spot light")) {
            spotLights.emplace_back();
        }

        for (size_t i = 0; i < spotLights.size();) {
            ImGui::PushID(static_cast<int>(i + MAX_LIGHTS));
            auto& light = spotLights[i];

            const std::string title =
                "Spot light " + std::to_string(i + 1);

            if (ImGui::TreeNode(title.c_str())) {
                ImGui::DragFloat3(
                    "Position", glm::value_ptr(light.position), 0.05f);
                ImGui::DragFloat3(
                    "Direction", glm::value_ptr(light.direction), 0.02f);
                ImGui::ColorEdit3(
                    "Color", glm::value_ptr(light.color));
                ImGui::SliderFloat(
                    "Intensity", &light.intensity, 0.0f, 100.0f);
                ImGui::SliderFloat(
                    "Inner angle", &light.innerAngle, 1.0f, 85.0f);
                ImGui::SliderFloat(
                    "Outer angle", &light.outerAngle,
                    light.innerAngle, 89.0f);

                light.outerAngle =
                    std::max(light.outerAngle, light.innerAngle);

                if (ImGui::Button("Remove")) {
                    spotLights.erase(spotLights.begin() + i);
                    ImGui::TreePop();
                    ImGui::PopID();
                    continue;
                }

                ImGui::TreePop();
            }

            ImGui::PopID();
            ++i;
        }
    }

    if (ImGui::CollapsingHeader("Shadow mapping",
                                ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Sun shadow", &sunShadowsEnabled);
        ImGui::Checkbox("First spot light shadow", &spotShadowsEnabled);

        if (spotLights.empty()) {
            ImGui::TextWrapped(
                "Add a spot light to enable its shadow.");
        }

        // Площадь, покрываемая ортографической проекцией солнца:
        // половина стороны квадрата области теневой карты в метрах.
        ImGui::SliderFloat(
            "Sun projection half extent",
            &sunShadowHalfExtent, 1.0f, 12.0f);

        // Сдвиг глубины (depth bias): константная составляющая и
        // масштаб, пропорциональный наклону поверхности.
        ImGui::SliderFloat(
            "Depth bias (constant)",
            &shadowDepthBiasConstant, 0.0f, 0.01f, "%.4f");
        ImGui::SliderFloat(
            "Depth bias (slope)",
            &shadowDepthBiasSlope, 0.0f, 10.0f, "%.2f");

        ImGui::Text("Shadow maps: %u x %u (sun, spot)",
            SHADOW_MAP_SIZE, SHADOW_MAP_SIZE);
    }

    if (ImGui::CollapsingHeader("Texture options",
                                ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Texture mode (Lab 3)", &textureMode);

        if (!textureMode) {
            ImGui::TextWrapped(
                "Lab 2 mode: procedural material colors without textures.");
        }
    }

    ImGui::Text("Point lights: %u / %u",
        static_cast<uint32_t>(pointLights.size()), MAX_LIGHTS);
    ImGui::Text("Spot lights: %u / %u",
        static_cast<uint32_t>(spotLights.size()), MAX_LIGHTS);

    ImGui::End();
}

void updateGpuBuffers()
{
    const auto extent = context.swapchain_extent;

    if (extent.width == 0 || extent.height == 0) {
        return;
    }

    const float aspect =
        static_cast<float>(extent.width) /
        static_cast<float>(extent.height);

    glm::vec3 forward = cameraTarget - cameraPosition;

    if (glm::length(forward) < 0.0001f) {
        forward = glm::vec3(0.0f, 0.0f, -1.0f);
    }

    glm::vec3 up{0.0f, 1.0f, 0.0f};

    if (std::abs(glm::dot(glm::normalize(forward), up)) > 0.99f) {
        up = glm::vec3(0.0f, 0.0f, 1.0f);
    }

    const glm::mat4 view = glm::lookAt(
        cameraPosition, cameraPosition + forward, up);

    glm::mat4 projection = glm::perspective(
        glm::radians(cameraFov), aspect, 0.1f, 100.0f);

    projection[1][1] *= -1.0f;

    SceneUBO scene{};
    scene.viewProjection = projection * view;
    scene.cameraPosition = glm::vec4(cameraPosition, 1.0f);
    scene.ambientColor = glm::vec4(ambientColor, ambientIntensity);

    glm::vec3 dir = directionalDirection;
    if (glm::length(dir) < 0.0001f) {
        dir = glm::vec3(0.0f, -1.0f, 0.0f);
    }

    scene.directionalDirection =
        glm::vec4(glm::normalize(dir), 0.0f);

    scene.directionalColorIntensity =
        glm::vec4(directionalColor, directionalIntensity);

    scene.lightCounts = glm::uvec4(
        static_cast<uint32_t>(pointLights.size()),
        static_cast<uint32_t>(spotLights.size()),
        // Флаги включённости теней для фрагментного шейдера:
        // z - солнце, w - первый прожектор.
        sunShadowsEnabled ? 1u : 0u,
        (spotShadowsEnabled && !spotLights.empty()) ? 1u : 0u
    );

    // Матрицы источников света и параметры выборки теней.
    sunViewProjection = makeSunViewProjection();
    spotViewProjection = makeSpotViewProjection();

    scene.sunViewProjection = sunViewProjection;
    scene.spotViewProjection = spotViewProjection;
    scene.shadowSettings = glm::vec4(
        1.0f / static_cast<float>(SHADOW_MAP_SIZE), 0.0f, 0.0f, 0.0f);

    writeBuffer(sceneBuffer, &scene, sizeof(scene));

    // Свойства всех моделей записываются в один динамический
    // uniform-буфер по выровненным слотам.
    for (uint32_t i = 0; i < SPHERE_COUNT; ++i) {
        const SceneObject& sphere = spheres[i];

        writeModelSlot(
            i,
            makeModelMatrix(sphere, i),
            glm::vec4(sphere.albedo, 1.0f),
            glm::vec4(sphere.specularColor, 1.0f),
            glm::vec4(
                sphere.shininess,
                sphere.specularStrength,
                sphere.emissiveStrength,
                0.0f));
    }

    // Пол не двигается: матрица модели - идентичность.
    writeModelSlot(
        static_cast<uint32_t>(SPHERE_COUNT + MAX_LIGHTS * 2),
        glm::mat4(1.0f),
        glm::vec4(1.0f, 1.0f, 1.0f, 1.0f),
        glm::vec4(1.0f),
        glm::vec4(16.0f, 0.8f, 0.0f, 0.0f));

    std::array<PointLightGPU, MAX_LIGHTS> gpuPoints{};

    for (size_t i = 0; i < pointLights.size(); ++i) {
        gpuPoints[i].position =
            glm::vec4(pointLights[i].position, 1.0f);
        gpuPoints[i].colorIntensity =
            glm::vec4(pointLights[i].color, pointLights[i].intensity);

        writeModelSlot(
            static_cast<uint32_t>(SPHERE_COUNT + i),
            glm::translate(glm::mat4(1.0f), pointLights[i].position) *
                glm::scale(glm::mat4(1.0f), glm::vec3(0.09f)),
            glm::vec4(pointLights[i].color, 1.0f),
            glm::vec4(0.0f),
            glm::vec4(1.0f, 0.0f, 0.0f, 0.0f));
    }

    writeBuffer(
        pointLightsBuffer, gpuPoints.data(), sizeof(gpuPoints));

    std::array<SpotLightGPU, MAX_LIGHTS> gpuSpots{};

    for (size_t i = 0; i < spotLights.size(); ++i) {
        const auto& light = spotLights[i];

        gpuSpots[i].position = glm::vec4(light.position, 1.0f);

        glm::vec3 direction = light.direction;
        if (glm::length(direction) < 0.0001f) {
            direction = glm::vec3(0.0f, -1.0f, 0.0f);
        }

        gpuSpots[i].direction =
            glm::vec4(glm::normalize(direction), 0.0f);

        gpuSpots[i].colorIntensity =
            glm::vec4(light.color, light.intensity);

        gpuSpots[i].cone = glm::vec4(
            glm::radians(light.innerAngle),
            glm::radians(light.outerAngle),
            0.0f, 0.0f
        );

        writeModelSlot(
            static_cast<uint32_t>(SPHERE_COUNT + MAX_LIGHTS + i),
            glm::translate(glm::mat4(1.0f), light.position) *
                glm::scale(glm::mat4(1.0f), glm::vec3(0.11f)),
            glm::vec4(light.color, 1.0f),
            glm::vec4(0.0f),
            glm::vec4(1.0f, 0.0f, 0.0f, 0.0f));
    }

    writeBuffer(
        spotLightsBuffer, gpuSpots.data(), sizeof(gpuSpots));
}

void bindModelSet(
    VkCommandBuffer cmd,
    const Material& material,
    uint32_t slot
)
{
    const uint32_t dynamicOffset =
        static_cast<uint32_t>(modelUniformStride * slot);

    vkCmdBindDescriptorSets(
        cmd,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        pipelineLayout,
        1,
        1,
        &material.descriptorSet,
        1,
        &dynamicOffset
    );
}

// Запись одного теневого прохода (динамический рендеринг):
// барьер UNDEFINED -> DEPTH_ATTACHMENT_OPTIMAL, рендеринг в карту
// глубины с точки зрения источника света, барьер в
// DEPTH_READ_ONLY_OPTIMAL для выборки из основного шейдера.
void recordShadowMap(
    VkCommandBuffer cmd,
    const ShadowMap& map,
    const glm::mat4& lightMatrix
)
{
    VkImageMemoryBarrier barrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = map.image,
        .subresourceRange = {
            VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1
        }
    };

    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    const VkRenderingAttachmentInfo depthAttachment{
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = map.view,
        .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue = {.depthStencil = {1.0f, 0}}
    };

    // В отличие от классического render pass ареа, attachment и
    // load/store операции задаются структурой VkRenderingInfo,
    // без объектов VkRenderPass и VkFramebuffer.
    const VkRenderingInfo renderingInfo{
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea = {{0, 0}, {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE}},
        .layerCount = 1,
        .pDepthAttachment = &depthAttachment
    };

    vkCmdBeginRendering(cmd, &renderingInfo);

    const VkViewport viewport{
        .x = 0.0f,
        .y = 0.0f,
        .width = static_cast<float>(SHADOW_MAP_SIZE),
        .height = static_cast<float>(SHADOW_MAP_SIZE),
        .minDepth = 0.0f,
        .maxDepth = 1.0f
    };

    const VkRect2D scissor{{0, 0}, {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE}};

    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdBindPipeline(
        cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipeline);

    // Сдвиг глубины из ImGui: запись в карту глубины выполняется с
    // небольшим смещением, чтобы исключить самоосвещение поверхностей
    // (shadow acne). Константная составляющая и масштаб по наклону
    // задаются динамической командой vkCmdSetDepthBias.
    vkCmdSetDepthBias(
        cmd,
        shadowDepthBiasConstant,
        0.0f,
        shadowDepthBiasSlope);

    const VkDeviceSize offset = 0;

    vkCmdBindVertexBuffers(cmd, 0, 1, &vertexBuffer.handle, &offset);
    vkCmdBindIndexBuffer(
        cmd, indexBuffer.handle, 0, VK_INDEX_TYPE_UINT32);

    // Сферы: теневому конвейеру не нужны наборы дескрипторов,
    // полная матрица свет-пространства передается push-константой.
    for (uint32_t i = 0; i < SPHERE_COUNT; ++i) {
        const glm::mat4 mvp =
            lightMatrix * makeModelMatrix(spheres[i], i);

        vkCmdPushConstants(
            cmd,
            shadowPipelineLayout,
            VK_SHADER_STAGE_VERTEX_BIT,
            0,
            sizeof(mvp),
            &mvp
        );

        vkCmdDrawIndexed(cmd, indexCount, 1, 0, 0, 0);
    }

    vkCmdBindVertexBuffers(cmd, 0, 1, &floorVertexBuffer.handle, &offset);
    vkCmdBindIndexBuffer(
        cmd, floorIndexBuffer.handle, 0, VK_INDEX_TYPE_UINT32);

    vkCmdPushConstants(
        cmd,
        shadowPipelineLayout,
        VK_SHADER_STAGE_VERTEX_BIT,
        0,
        sizeof(lightMatrix),
        &lightMatrix
    );

    vkCmdDrawIndexed(cmd, floorIndexCount, 1, 0, 0, 0);

    vkCmdEndRendering(cmd);

    barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL;

    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);
}

} // namespace

bool initialize()
{
    static_assert(sizeof(SceneUBO) == 288);
    static_assert(sizeof(ModelUBO) == 112);
    static_assert(sizeof(PointLightGPU) == 32);
    static_assert(sizeof(SpotLightGPU) == 64);
    static_assert(sizeof(DrawPush) == 32);

    try {
        pointLights.clear();
        spotLights.clear();

        pointLights.emplace_back();
        spotLights.emplace_back();

        spotLights[0].position = {-2.0f, 2.0f, 2.0f};
        spotLights[0].direction = {0.5f, -0.6f, -0.6f};

        createUploadCommandPool();
        createSphere();
        createFloorMesh();
        createDescriptorLayouts();
        createMaterialTextures();
        createShadowResources();
        createDescriptors();
        createPipeline();
        createShadowPipeline();

        previousFrameTime = 0.0;

        return true;
    }
    catch (const std::exception& error) {
        std::cerr << "Lab4 initialization failed: "
                  << error.what() << '\n';
        shutdown();
        return false;
    }
}

void shutdown()
{
    if (context.device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(context.device);

        if (pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(context.device, pipeline, nullptr);
            pipeline = VK_NULL_HANDLE;
        }

        if (pipelineLayout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(
                context.device, pipelineLayout, nullptr);
            pipelineLayout = VK_NULL_HANDLE;
        }

        if (descriptorPool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(
                context.device, descriptorPool, nullptr);
            descriptorPool = VK_NULL_HANDLE;
        }

        if (textureSampler != VK_NULL_HANDLE) {
            vkDestroySampler(context.device, textureSampler, nullptr);
            textureSampler = VK_NULL_HANDLE;
        }

        if (shadowPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(
                context.device, shadowPipeline, nullptr);
            shadowPipeline = VK_NULL_HANDLE;
        }

        if (shadowPipelineLayout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(
                context.device, shadowPipelineLayout, nullptr);
            shadowPipelineLayout = VK_NULL_HANDLE;
        }

        if (shadowSampler != VK_NULL_HANDLE) {
            vkDestroySampler(context.device, shadowSampler, nullptr);
            shadowSampler = VK_NULL_HANDLE;
        }

        for (ShadowMap& map : shadowMaps) {
            if (map.view != VK_NULL_HANDLE) {
                vkDestroyImageView(context.device, map.view, nullptr);
                map.view = VK_NULL_HANDLE;
            }

            if (map.image != VK_NULL_HANDLE) {
                vmaDestroyImage(
                    context.allocator, map.image, map.allocation);
                map.image = VK_NULL_HANDLE;
                map.allocation = VK_NULL_HANDLE;
            }
        }

        for (Material& material : materials) {
            destroyTexture(material.albedo);
            destroyTexture(material.specular);
            destroyTexture(material.emissive);
            material.descriptorSet = VK_NULL_HANDLE;
        }

        destroyTexture(blackTexture);

        if (sceneSetLayout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(
                context.device, sceneSetLayout, nullptr);
            sceneSetLayout = VK_NULL_HANDLE;
        }

        if (modelSetLayout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(
                context.device, modelSetLayout, nullptr);
            modelSetLayout = VK_NULL_HANDLE;
        }

        if (uploadCommandPool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(
                context.device, uploadCommandPool, nullptr);
            uploadCommandPool = VK_NULL_HANDLE;
        }
    }

    destroyBuffer(vertexBuffer);
    destroyBuffer(indexBuffer);
    destroyBuffer(floorVertexBuffer);
    destroyBuffer(floorIndexBuffer);
    destroyBuffer(sceneBuffer);
    destroyBuffer(modelBuffer);
    destroyBuffer(pointLightsBuffer);
    destroyBuffer(spotLightsBuffer);

    sceneSet = VK_NULL_HANDLE;
    indexCount = 0;
    floorIndexCount = 0;
    modelUniformStride = 0;
}

void update(double time)
{
    if (previousFrameTime == 0.0) {
        previousFrameTime = time;
    }

    const double delta = time - previousFrameTime;
    previousFrameTime = time;

    if (animationPlaying) {
        animationTime +=
            static_cast<float>(delta) * animationSpeed;
    }

    updateImGui();
}

void render(const graphics::internal::FrameData& fd)
{
    const bool sceneReady =
        pipeline != VK_NULL_HANDLE &&
        indexCount != 0 &&
        context.swapchain_extent.width != 0 &&
        context.swapchain_extent.height != 0;

    VkCommandBuffer cmd = fd.command_buffer;

    if (sceneReady) {
        updateGpuBuffers();

        // Теневые проходы: карты глубины с точек зрения солнца и
        // первого прожектора строятся каждый кадр динамическим
        // рендерингом (вне обычного render pass).
        if (sunShadowsEnabled) {
            recordShadowMap(cmd, shadowMaps[0], sunViewProjection);
        }

        if (spotShadowsEnabled && !spotLights.empty()) {
            recordShadowMap(cmd, shadowMaps[1], spotViewProjection);
        }
    }

    // Основной render pass остался классическим, как в предыдущих
    // работах: он открывается здесь и закрывается в
    // submitAndPresent() после отрисовки интерфейса ImGui.
    VkClearValue clearValues[2] = {};
    clearValues[0].color = {{ 0.02f, 0.03f, 0.05f, 1.0f }};
    clearValues[1].depthStencil = { 1.0f, 0 };

    const VkRenderPassBeginInfo rpBegin{
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = context.render_pass,
        .framebuffer = fd.framebuffer,
        .renderArea = {.extent = context.swapchain_extent},
        .clearValueCount = 2,
        .pClearValues = clearValues
    };

    vkCmdBeginRenderPass(cmd, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);

    if (!sceneReady) {
        return;
    }

    VkViewport viewport{
        .x = 0.0f,
        .y = 0.0f,
        .width = static_cast<float>(context.swapchain_extent.width),
        .height = static_cast<float>(context.swapchain_extent.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f
    };

    VkRect2D scissor{
        .offset = {0, 0},
        .extent = context.swapchain_extent
    };

    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdBindPipeline(
        cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

    const VkDeviceSize offset = 0;

    vkCmdBindDescriptorSets(
        cmd,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        pipelineLayout,
        0,
        1,
        &sceneSet,
        0,
        nullptr
    );

    // Пол: та же текстура камня с повторением; тень от сфер
    // падает на него через сэмплер сравнения из binding 3.
    vkCmdBindVertexBuffers(
        cmd, 0, 1, &floorVertexBuffer.handle, &offset);

    vkCmdBindIndexBuffer(
        cmd, floorIndexBuffer.handle, 0, VK_INDEX_TYPE_UINT32);

    {
        DrawPush floorPush{};
        floorPush.overrideColor = glm::vec4(1.0f);
        floorPush.time = 0.0f;
        floorPush.textureEffectStrength = 0.0f;
        floorPush.unlit = 0;
        floorPush.textureMode = textureMode ? 1u : 0u;

        vkCmdPushConstants(
            cmd,
            pipelineLayout,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            0,
            sizeof(floorPush),
            &floorPush
        );

        bindModelSet(
            cmd,
            materials[2],
            static_cast<uint32_t>(SPHERE_COUNT + MAX_LIGHTS * 2));

        vkCmdDrawIndexed(cmd, floorIndexCount, 1, 0, 0, 0);
    }

    // Три сферы: материал выбирается набором дескрипторов,
    // экземпляр модели - динамическим смещением в общем UBO.
    vkCmdBindVertexBuffers(
        cmd, 0, 1, &vertexBuffer.handle, &offset);

    vkCmdBindIndexBuffer(
        cmd, indexBuffer.handle, 0, VK_INDEX_TYPE_UINT32);

    for (uint32_t i = 0; i < SPHERE_COUNT; ++i) {
        const SceneObject& sphere = spheres[i];
        const Material& material = materials[sphere.materialIndex];

        DrawPush push{};
        push.overrideColor = glm::vec4(1.0f);
        push.time = animationTime;
        push.textureEffectStrength =
            (i == 0 && textureMode && textureEffectEnabled)
                ? textureEffectStrength : 0.0f;
        push.unlit = 0;
        push.textureMode = textureMode ? 1u : 0u;

        vkCmdPushConstants(
            cmd,
            pipelineLayout,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            0,
            sizeof(push),
            &push
        );

        bindModelSet(cmd, material, i);

        vkCmdDrawIndexed(cmd, indexCount, 1, 0, 0, 0);
    }

    // Цветные маркеры источников света отрисовываются без освещения
    // в процедурной окраске уровня лабораторной №2.
    DrawPush push{};
    push.overrideColor = glm::vec4(1.0f);
    push.time = 0.0f;
    push.textureEffectStrength = 0.0f;
    push.unlit = 1;
    push.textureMode = 0;

    for (size_t i = 0; i < pointLights.size(); ++i) {
        push.overrideColor = glm::vec4(pointLights[i].color, 1.0f);

        vkCmdPushConstants(
            cmd,
            pipelineLayout,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            0,
            sizeof(push),
            &push
        );

        bindModelSet(
            cmd,
            materials[0],
            static_cast<uint32_t>(SPHERE_COUNT + i));

        vkCmdDrawIndexed(cmd, indexCount, 1, 0, 0, 0);
    }

    for (size_t i = 0; i < spotLights.size(); ++i) {
        push.overrideColor = glm::vec4(spotLights[i].color, 1.0f);

        vkCmdPushConstants(
            cmd,
            pipelineLayout,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            0,
            sizeof(push),
            &push
        );

        bindModelSet(
            cmd,
            materials[0],
            static_cast<uint32_t>(SPHERE_COUNT + MAX_LIGHTS + i));

        vkCmdDrawIndexed(cmd, indexCount, 1, 0, 0, 0);
    }
}

void rotateCamera(float deltaX, float deltaY) {
    glm::vec3 offset = cameraTarget - cameraPosition;
    float distance = glm::length(offset);

    if (distance < 0.001f) {
        return;
    }

    glm::vec3 direction = glm::normalize(offset);

    float yaw = std::atan2(direction.z, direction.x);
    float pitch = std::asin(
        std::clamp(direction.y, -1.0f, 1.0f)
    );

    constexpr float sensitivity = 0.005f;

    yaw += deltaX * sensitivity;
    pitch += deltaY * sensitivity;

    constexpr float maxPitch = 1.5f;
    pitch = std::clamp(pitch, -maxPitch, maxPitch);

    glm::vec3 newDirection{
        std::cos(yaw) * std::cos(pitch),
        std::sin(pitch),
        std::sin(yaw) * std::cos(pitch)
    };

    cameraTarget = cameraPosition + newDirection * distance;
}

void processInput(
    float deltaTime,
    bool forward,
    bool backward,
    bool left,
    bool right,
    bool up,
    bool down,
    bool fast
) {
    glm::vec3 viewDirection = cameraTarget - cameraPosition;

    if (glm::length(viewDirection) < 0.001f) {
        return;
    }

    viewDirection = glm::normalize(viewDirection);

    const glm::vec3 worldUp{0.0f, 1.0f, 0.0f};

    glm::vec3 horizontalForward{viewDirection.x, 0.0f, viewDirection.z};

    if (glm::length(horizontalForward) > 0.001f) {
        horizontalForward = glm::normalize(horizontalForward);
    } else {
        horizontalForward = {0.0f, 0.0f, -1.0f};
    }

    const glm::vec3 rightDirection =
        glm::normalize(glm::cross(horizontalForward, worldUp));

    glm::vec3 movement{0.0f};

    if (forward)  movement += horizontalForward;
    if (backward) movement -= horizontalForward;
    if (right)    movement += rightDirection;
    if (left)     movement -= rightDirection;
    if (up)       movement += worldUp;
    if (down)     movement -= worldUp;

    if (glm::length(movement) < 0.001f) {
        return;
    }

    movement = glm::normalize(movement);

    const float baseSpeed = 2.5f;
    const float speed = fast ? baseSpeed * 2.5f : baseSpeed;
    const glm::vec3 displacement = movement * speed * deltaTime;

    cameraPosition += displacement;
    cameraTarget += displacement;
}

} // namespace application
