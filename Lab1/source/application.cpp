
#include "application.hpp"
#include "graphics_internal.hpp"

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/constants.hpp>
#include <imgui.h>

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

namespace application {
namespace {

using graphics::internal::context;

constexpr uint32_t MAX_LIGHTS = 64;
constexpr uint32_t SPHERE_SEGMENTS = 32;
constexpr uint32_t SPHERE_RINGS = 20;

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
};

struct alignas(16) SceneUBO {
    glm::mat4 viewProjection;
    glm::vec4 cameraPosition;
    glm::vec4 ambientColor;
    glm::vec4 directionalDirection;
    glm::vec4 directionalColorIntensity;
    glm::uvec4 lightCounts;
};

struct alignas(16) ModelUBO {
    glm::mat4 model;
    glm::vec4 albedo;
    glm::vec4 specularColor;
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
    uint32_t unlit;
};

struct Buffer {
    VkBuffer handle = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    void* mapped = nullptr;
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
Buffer sceneBuffer;
Buffer modelBuffer;
Buffer pointLightsBuffer;
Buffer spotLightsBuffer;

std::vector<Buffer> markerModelBuffers;
std::vector<VkDescriptorSet> markerDescriptorSets;

uint32_t indexCount = 0;

VkDescriptorSetLayout sceneSetLayout = VK_NULL_HANDLE;
VkDescriptorSetLayout modelSetLayout = VK_NULL_HANDLE;
VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
VkDescriptorSet sceneSet = VK_NULL_HANDLE;
VkDescriptorSet modelSet = VK_NULL_HANDLE;

VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
VkPipeline pipeline = VK_NULL_HANDLE;

std::vector<PointLight> pointLights;
std::vector<SpotLight> spotLights;

glm::vec3 cameraPosition{0.0f, 0.0f, 5.0f};
glm::vec3 cameraTarget{0.0f, 0.0f, 0.0f};
float cameraFov = 55.0f;

glm::vec3 albedo{0.72f, 0.38f, 0.22f};
glm::vec3 specularColor{1.0f, 1.0f, 1.0f};
float shininess = 48.0f;

glm::vec3 ambientColor{0.18f, 0.20f, 0.24f};
float ambientIntensity = 0.25f;

glm::vec3 directionalDirection{-0.5f, -1.0f, -0.4f};
glm::vec3 directionalColor{1.0f, 0.95f, 0.85f};
float directionalIntensity = 1.0f;

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

std::vector<char> readFile(const std::string& filename)
{
    const std::array<std::string, 8> paths = {
        "shaders/" + filename,
        "Lab2/shaders/" + filename,
        "build-mingw-debug/shaders/" + filename,
        "Lab2/build-mingw-debug/shaders/" + filename,
        "build-mingw-release/shaders/" + filename,
        "Lab2/build-mingw-release/shaders/" + filename,
        "build-vs-debug/shaders/" + filename,
        "Lab2/build-vs-debug/shaders/" + filename
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

void createSphere()
{
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

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

            vertices.push_back({normal, glm::normalize(normal)});
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

    vertexBuffer = createBuffer(
        vertices.size() * sizeof(Vertex),
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
    );

    indexBuffer = createBuffer(
        indices.size() * sizeof(uint32_t),
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT
    );

    writeBuffer(
        vertexBuffer, vertices.data(),
        vertices.size() * sizeof(Vertex)
    );

    writeBuffer(
        indexBuffer, indices.data(),
        indices.size() * sizeof(uint32_t)
    );

    indexCount = static_cast<uint32_t>(indices.size());
}

void createDescriptorLayouts()
{
    const std::array<VkDescriptorSetLayoutBinding, 3> sceneBindings = {{
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

    VkDescriptorSetLayoutBinding modelBinding{
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1,
        .stageFlags =
            VK_SHADER_STAGE_VERTEX_BIT |
            VK_SHADER_STAGE_FRAGMENT_BIT
    };

    VkDescriptorSetLayoutCreateInfo modelInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1,
        .pBindings = &modelBinding
    };

    if (vkCreateDescriptorSetLayout(
            context.device, &modelInfo, nullptr,
            &modelSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create model layout");
    }
}

void createDescriptors()
{
    sceneBuffer = createBuffer(
        sizeof(SceneUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    modelBuffer = createBuffer(
        sizeof(ModelUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    pointLightsBuffer = createBuffer(
        sizeof(PointLightGPU) * MAX_LIGHTS,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    spotLightsBuffer = createBuffer(
        sizeof(SpotLightGPU) * MAX_LIGHTS,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    const std::array<VkDescriptorPoolSize, 2> poolSizes = {{
        {
            .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 2 + MAX_LIGHTS * 2
        },
        {
            .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 2
        }
    }};

    VkDescriptorPoolCreateInfo poolInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 2 + MAX_LIGHTS * 2,
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data()
    };

    if (vkCreateDescriptorPool(
            context.device, &poolInfo, nullptr,
            &descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor pool");
    }

    const std::array<VkDescriptorSetLayout, 2> layouts = {
        sceneSetLayout, modelSetLayout
    };

    std::array<VkDescriptorSet, 2> sets{};

    VkDescriptorSetAllocateInfo allocateInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptorPool,
        .descriptorSetCount = 2,
        .pSetLayouts = layouts.data()
    };

    if (vkAllocateDescriptorSets(
            context.device, &allocateInfo, sets.data()) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate descriptor sets");
    }

    sceneSet = sets[0];
    modelSet = sets[1];

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

    VkDescriptorBufferInfo modelBufferInfo{
        .buffer = modelBuffer.handle,
        .offset = 0,
        .range = sizeof(ModelUBO)
    };

    const std::array<VkWriteDescriptorSet, 4> writes = {{
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
            .dstSet = modelSet,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .pBufferInfo = &modelBufferInfo
        }
    }};

    vkUpdateDescriptorSets(
        context.device, static_cast<uint32_t>(writes.size()),
        writes.data(), 0, nullptr);

    markerModelBuffers.reserve(MAX_LIGHTS * 2);
    markerDescriptorSets.resize(MAX_LIGHTS * 2);

    std::vector<VkDescriptorSetLayout> markerLayouts(
        MAX_LIGHTS * 2, modelSetLayout);

    VkDescriptorSetAllocateInfo markerAllocateInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptorPool,
        .descriptorSetCount = static_cast<uint32_t>(markerLayouts.size()),
        .pSetLayouts = markerLayouts.data()
    };

    if (vkAllocateDescriptorSets(
            context.device, &markerAllocateInfo,
            markerDescriptorSets.data()) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate marker sets");
    }

    for (uint32_t i = 0; i < MAX_LIGHTS * 2; ++i) {
        markerModelBuffers.push_back(createBuffer(
            sizeof(ModelUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));

        VkDescriptorBufferInfo info{
            .buffer = markerModelBuffers[i].handle,
            .offset = 0,
            .range = sizeof(ModelUBO)
        };

        VkWriteDescriptorSet write{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = markerDescriptorSets[i],
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .pBufferInfo = &info
        };

        vkUpdateDescriptorSets(
            context.device, 1, &write, 0, nullptr);
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

        const std::array<VkVertexInputAttributeDescription, 2> attributes = {{
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
            }
        }};

        VkPipelineVertexInputStateCreateInfo vertexInput{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &vertexBinding,
            .vertexAttributeDescriptionCount = 2,
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

void updateImGui()
{
    ImGui::Begin("Lab 2 - Camera and Lighting");

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

    if (ImGui::CollapsingHeader("Material",
                                ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::ColorEdit3("Albedo", glm::value_ptr(albedo));
        ImGui::ColorEdit3(
            "Specular color", glm::value_ptr(specularColor));
        ImGui::SliderFloat("Shininess", &shininess, 1.0f, 256.0f);
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
        0, 0
    );

    writeBuffer(sceneBuffer, &scene, sizeof(scene));

    ModelUBO model{};
    model.model = glm::mat4(1.0f);
    model.albedo = glm::vec4(albedo, 1.0f);
    model.specularColor = glm::vec4(specularColor, 1.0f);
    model.materialParameters = glm::vec4(shininess, 0.0f, 0.0f, 0.0f);

    writeBuffer(modelBuffer, &model, sizeof(model));

    std::array<PointLightGPU, MAX_LIGHTS> gpuPoints{};

    for (size_t i = 0; i < pointLights.size(); ++i) {
        gpuPoints[i].position =
            glm::vec4(pointLights[i].position, 1.0f);
        gpuPoints[i].colorIntensity =
            glm::vec4(pointLights[i].color, pointLights[i].intensity);

        ModelUBO marker{};
        marker.model =
            glm::translate(glm::mat4(1.0f), pointLights[i].position) *
            glm::scale(glm::mat4(1.0f), glm::vec3(0.09f));
        marker.albedo = glm::vec4(pointLights[i].color, 1.0f);
        marker.specularColor = glm::vec4(0.0f);
        marker.materialParameters = glm::vec4(1.0f);

        writeBuffer(markerModelBuffers[i], &marker, sizeof(marker));
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

        const size_t markerIndex = MAX_LIGHTS + i;

        ModelUBO marker{};
        marker.model =
            glm::translate(glm::mat4(1.0f), light.position) *
            glm::scale(glm::mat4(1.0f), glm::vec3(0.11f));
        marker.albedo = glm::vec4(light.color, 1.0f);
        marker.specularColor = glm::vec4(0.0f);
        marker.materialParameters = glm::vec4(1.0f);

        writeBuffer(
            markerModelBuffers[markerIndex], &marker, sizeof(marker));
    }

    writeBuffer(
        spotLightsBuffer, gpuSpots.data(), sizeof(gpuSpots));
}

} // namespace

bool initialize()
{
    static_assert(sizeof(SceneUBO) == 144);
    static_assert(sizeof(ModelUBO) == 112);
    static_assert(sizeof(PointLightGPU) == 32);
    static_assert(sizeof(SpotLightGPU) == 64);

    try {
        pointLights.clear();
        spotLights.clear();

        pointLights.emplace_back();
        spotLights.emplace_back();

        spotLights[0].position = {-2.0f, 2.0f, 2.0f};
        spotLights[0].direction = {0.5f, -0.6f, -0.6f};

        createSphere();
        createDescriptorLayouts();
        createDescriptors();
        createPipeline();

        return true;
    }
    catch (const std::exception& error) {
        std::cerr << "Lab2 initialization failed: "
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
    }

    destroyBuffer(vertexBuffer);
    destroyBuffer(indexBuffer);
    destroyBuffer(sceneBuffer);
    destroyBuffer(modelBuffer);
    destroyBuffer(pointLightsBuffer);
    destroyBuffer(spotLightsBuffer);

    for (auto& buffer : markerModelBuffers) {
        destroyBuffer(buffer);
    }

    markerModelBuffers.clear();
    markerDescriptorSets.clear();

    sceneSet = VK_NULL_HANDLE;
    modelSet = VK_NULL_HANDLE;
    indexCount = 0;
}

void update(double /*time*/)
{
    updateImGui();
}

void render(const graphics::internal::FrameData& fd)
{
    if (pipeline == VK_NULL_HANDLE ||
        indexCount == 0 ||
        context.swapchain_extent.width == 0 ||
        context.swapchain_extent.height == 0) {
        return;
    }

    updateGpuBuffers();

    VkCommandBuffer cmd = fd.command_buffer;

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

    vkCmdBindVertexBuffers(
        cmd, 0, 1, &vertexBuffer.handle, &offset);

    vkCmdBindIndexBuffer(
        cmd, indexBuffer.handle, 0, VK_INDEX_TYPE_UINT32);

    const std::array<VkDescriptorSet, 2> mainSets = {
        sceneSet, modelSet
    };

    vkCmdBindDescriptorSets(
        cmd,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        pipelineLayout,
        0,
        static_cast<uint32_t>(mainSets.size()),
        mainSets.data(),
        0,
        nullptr
    );

    DrawPush push{};
    push.overrideColor = glm::vec4(1.0f);
    push.unlit = 0;

    vkCmdPushConstants(
        cmd,
        pipelineLayout,
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        0,
        sizeof(push),
        &push
    );

    // Основная сфера.
    vkCmdDrawIndexed(cmd, indexCount, 1, 0, 0, 0);

    // Цветные маркеры источников света.
    push.unlit = 1;

    for (size_t i = 0; i < pointLights.size(); ++i) {
        push.overrideColor = glm::vec4(pointLights[i].color, 1.0f);

        vkCmdBindDescriptorSets(
            cmd,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipelineLayout,
            1,
            1,
            &markerDescriptorSets[i],
            0,
            nullptr
        );

        vkCmdPushConstants(
            cmd,
            pipelineLayout,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            0,
            sizeof(push),
            &push
        );

        vkCmdDrawIndexed(cmd, indexCount, 1, 0, 0, 0);
    }

    for (size_t i = 0; i < spotLights.size(); ++i) {
        const size_t markerIndex = MAX_LIGHTS + i;
        push.overrideColor = glm::vec4(spotLights[i].color, 1.0f);

        vkCmdBindDescriptorSets(
            cmd,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipelineLayout,
            1,
            1,
            &markerDescriptorSets[markerIndex],
            0,
            nullptr
        );

        vkCmdPushConstants(
            cmd,
            pipelineLayout,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            0,
            sizeof(push),
            &push
        );

        vkCmdDrawIndexed(cmd, indexCount, 1, 0, 0, 0);
    }
}

} // namespace application
