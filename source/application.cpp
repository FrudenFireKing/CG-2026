#include "application.hpp"
#include "graphics_internal.hpp"
#include <imgui.h>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <vector>
#include <cmath>
#include <fstream>
#include <iostream>
#include <array>

using namespace graphics::internal;

struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;
};

struct UBO_MVP {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
};

struct UBO_Color {
    glm::vec4 color;
};

struct SceneObject {
    glm::vec3 pos = {0.0f, 0.0f, 0.0f};
    glm::vec3 rot = {0.0f, 0.0f, 0.0f};
    glm::vec3 scale = {1.0f, 1.0f, 1.0f};
    glm::vec4 color = {1.0f, 1.0f, 1.0f, 1.0f};

    VkBuffer uboMVP;
    VmaAllocation allocMVP;
    VkBuffer uboColor;
    VmaAllocation allocColor;
    VkDescriptorSet descriptorSet;
};

std::vector<Vertex> vertices;
std::vector<uint32_t> indices;

VkBuffer vertexBuffer;
VmaAllocation vertexBufferAlloc;
VkBuffer indexBuffer;
VmaAllocation indexBufferAlloc;

VkDescriptorSetLayout descSetLayout;
VkPipelineLayout pipelineLayout;
VkPipeline graphicsPipeline;
VkDescriptorPool descriptorPool;

SceneObject objMain;
SceneObject objStatic; 
bool usePerspective = true;
bool isPlaying = true;
float animSpeed = 1.0f;
float animRadius = 3.0f;
float animTime = 0.0f;


static void generateSphere(float radius, int stacks, int sectors) {
    vertices.clear();
    indices.clear();

    for (int i = 0; i <= stacks; ++i) {
        float V = static_cast<float>(i) / static_cast<float>(stacks);
        float phi = V * glm::pi<float>();
        for (int j = 0; j <= sectors; ++j) {
            float U = static_cast<float>(j) / static_cast<float>(sectors);
            float theta = U * 2.0f * glm::pi<float>();

            float x = radius * std::sin(phi) * std::cos(theta);
            float y = radius * std::cos(phi);
            float z = radius * std::sin(phi) * std::sin(theta);

            glm::vec3 color = glm::vec3(x, y, z) * 0.5f + 0.5f; 
            vertices.push_back({{x, y, z}, color});
        }
    }

    for (int i = 0; i < stacks; ++i) {
        for (int j = 0; j < sectors; ++j) {
            uint32_t first = (i * (sectors + 1)) + j;
            uint32_t second = first + sectors + 1;

            indices.push_back(first);
            indices.push_back(second);
            indices.push_back(first + 1);

            indices.push_back(second);
            indices.push_back(second + 1);
            indices.push_back(first + 1);
        }
    }
}

static std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open shader file: " << filename << std::endl;
        return {};
    }
    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    return buffer;
}

static VkShaderModule createShaderModule(const std::vector<char>& code) {
    if (code.empty()) return VK_NULL_HANDLE;

    VkShaderModuleCreateInfo createInfo = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = code.size(),
        .pCode = reinterpret_cast<const uint32_t*>(code.data())
    };
    VkShaderModule shaderModule;
    if (vkCreateShaderModule(context.device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        std::cerr << "Failed to create shader module\n";
        return VK_NULL_HANDLE;
    }
    return shaderModule;
}

static void createVmaBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer& buffer, VmaAllocation& alloc) {
    VkBufferCreateInfo bufInfo = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE
    };
    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    
    vmaCreateBuffer(context.allocator, &bufInfo, &allocInfo, &buffer, &alloc, nullptr);
}

static void initSceneObject(SceneObject& obj) {
    createVmaBuffer(sizeof(UBO_MVP), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, obj.uboMVP, obj.allocMVP);
    createVmaBuffer(sizeof(UBO_Color), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, obj.uboColor, obj.allocColor);

    VkDescriptorSetAllocateInfo allocInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptorPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &descSetLayout
    };
    vkAllocateDescriptorSets(context.device, &allocInfo, &obj.descriptorSet);

    VkDescriptorBufferInfo mvpInfo = { .buffer = obj.uboMVP, .offset = 0, .range = sizeof(UBO_MVP) };
    VkDescriptorBufferInfo colorInfo = { .buffer = obj.uboColor, .offset = 0, .range = sizeof(UBO_Color) };

    std::array<VkWriteDescriptorSet, 2> descWrites{};
    descWrites[0] = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = obj.descriptorSet, .dstBinding = 0, .dstArrayElement = 0,
        .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .pBufferInfo = &mvpInfo
    };
    descWrites[1] = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = obj.descriptorSet, .dstBinding = 1, .dstArrayElement = 0,
        .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .pBufferInfo = &colorInfo
    };
    vkUpdateDescriptorSets(context.device, static_cast<uint32_t>(descWrites.size()), descWrites.data(), 0, nullptr);
}


bool application::initialize() {
    generateSphere(1.0f, 9, 9);

    if (vertices.empty() || indices.empty()) {
        std::cerr << "Failed to generate sphere vertices or indices!\n";
        return false;
    }

    createVmaBuffer(sizeof(Vertex) * vertices.size(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertexBuffer, vertexBufferAlloc);
    createVmaBuffer(sizeof(uint32_t) * indices.size(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, indexBuffer, indexBufferAlloc);

    void* data = nullptr;
    vmaMapMemory(context.allocator, vertexBufferAlloc, &data);
    memcpy(data, vertices.data(), sizeof(Vertex) * vertices.size());
    vmaUnmapMemory(context.allocator, vertexBufferAlloc);

    vmaMapMemory(context.allocator, indexBufferAlloc, &data);
    memcpy(data, indices.data(), sizeof(uint32_t) * indices.size());
    vmaUnmapMemory(context.allocator, indexBufferAlloc);

    std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
    bindings[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr};
    bindings[1] = {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};

    VkDescriptorSetLayoutCreateInfo layoutInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = static_cast<uint32_t>(bindings.size()),
        .pBindings = bindings.data()
    };
    if (vkCreateDescriptorSetLayout(context.device, &layoutInfo, nullptr, &descSetLayout) != VK_SUCCESS) {
        return false;
    }

    std::array<VkDescriptorPoolSize, 1> poolSizes{};
    poolSizes[0] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4};
    VkDescriptorPoolCreateInfo poolInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 2,
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data()
    };
    if (vkCreateDescriptorPool(context.device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        return false;
    }

    auto vertCode = readFile("shaders/vert.spv");
    auto fragCode = readFile("shaders/frag.spv");
    VkShaderModule vertModule = createShaderModule(vertCode);
    VkShaderModule fragModule = createShaderModule(fragCode);

    if (vertModule == VK_NULL_HANDLE || fragModule == VK_NULL_HANDLE) {
        std::cerr << "Shader modules loading failed!\n";
        return false;
    }

    VkPipelineShaderStageCreateInfo shaderStages[] = {
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vertModule, .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fragModule, .pName = "main"}
    };

    VkVertexInputBindingDescription bindingDesc = {0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
    std::array<VkVertexInputAttributeDescription, 2> attrDescs{};
    attrDescs[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos)};
    attrDescs[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color)};

    VkPipelineVertexInputStateCreateInfo vertexInputInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &bindingDesc,
        .vertexAttributeDescriptionCount = 2, .pVertexAttributeDescriptions = attrDescs.data()
    };

    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, .primitiveRestartEnable = VK_FALSE
    };

    VkPipelineViewportStateCreateInfo viewportState = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1, .scissorCount = 1
    };

    VkPipelineRasterizationStateCreateInfo rasterizer = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_FRONT_BIT,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .lineWidth = 1.0f
    };

    VkPipelineMultisampleStateCreateInfo multisampling = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT
    };

    VkPipelineDepthStencilStateCreateInfo depthStencil = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = VK_TRUE,
        .depthWriteEnable = VK_TRUE,
        .depthCompareOp = VK_COMPARE_OP_LESS
    };

    VkPipelineColorBlendAttachmentState colorBlendAttachment = {
        .blendEnable = VK_FALSE,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT
    };
    VkPipelineColorBlendStateCreateInfo colorBlending = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1, .pAttachments = &colorBlendAttachment
    };

    std::vector<VkDynamicState> dynamicStates = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamicState = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()), .pDynamicStates = dynamicStates.data()
    };

    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &descSetLayout
    };
    vkCreatePipelineLayout(context.device, &pipelineLayoutInfo, nullptr, &pipelineLayout);

    VkGraphicsPipelineCreateInfo pipelineInfo = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = 2, .pStages = shaderStages,
        .pVertexInputState = &vertexInputInfo,
        .pInputAssemblyState = &inputAssembly,
        .pViewportState = &viewportState,
        .pRasterizationState = &rasterizer,
        .pMultisampleState = &multisampling,
        .pDepthStencilState = &depthStencil,
        .pColorBlendState = &colorBlending,
        .pDynamicState = &dynamicState,
        .layout = pipelineLayout,
        .renderPass = context.render_pass,
        .subpass = 0
    };
    vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline);

    vkDestroyShaderModule(context.device, fragModule, nullptr);
    vkDestroyShaderModule(context.device, vertModule, nullptr);

    initSceneObject(objMain);
    objMain.pos = {0.0f, 0.0f, 0.0f};
    objMain.color = {1.0f, 1.0f, 0.1f, 1.0f};

    initSceneObject(objStatic);
    objStatic.pos = {-2.5f, 0.0f, 0.0f};
    objStatic.color = {0.2f, 1.0f, 0.2f, 1.0f};

    return true;
}


void application::update(double time_or_delta) {
    static double last_frame_timestamp = 0.0;
    float delta_time = 0.0f;

    if (time_or_delta > 1.0) {
        if (last_frame_timestamp == 0.0) last_frame_timestamp = time_or_delta;
        delta_time = static_cast<float>(time_or_delta - last_frame_timestamp);
        last_frame_timestamp = time_or_delta;
    } else {
        delta_time = static_cast<float>(time_or_delta);
    }
    if (delta_time > 0.1f) delta_time = 0.1f;

    static float moonRadius = 2.0f;
    static float moonSpeed = 2.0f;

    if (isPlaying) {
        animTime += delta_time * animSpeed;
        
        objMain.rot.y += 30.0f * delta_time;
        if (objMain.rot.y >= 360.0f) objMain.rot.y -= 360.0f;

        objStatic.rot.y += 60.0f * delta_time;
        if (objStatic.rot.y >= 360.0f) objStatic.rot.y -= 360.0f;
    }

    ImGui::Begin("Lab #1 Settings");

    ImGui::Text("Task 1: Projection Switch");
    int projType = usePerspective ? 0 : 1;
    ImGui::RadioButton("Perspective", &projType, 0); ImGui::SameLine();
    ImGui::RadioButton("Orthographic", &projType, 1);
    usePerspective = (projType == 0);

    ImGui::Separator();
    ImGui::Text("Task 2: Main Sphere (Center)");
    ImGui::DragFloat3("Center Position", glm::value_ptr(objMain.pos), 0.05f);
    ImGui::DragFloat3("Rotation", glm::value_ptr(objMain.rot), 0.5f);
    ImGui::DragFloat3("Scale", glm::value_ptr(objMain.scale), 0.05f, 0.1f, 10.0f);

    ImGui::Separator();
    ImGui::Text("Task 3: Orbit Animation");
    ImGui::Checkbox("Play Animation", &isPlaying);
    ImGui::SliderFloat("Orbit Speed", &animSpeed, 0.1f, 3.0f);
    ImGui::SliderFloat("Orbit Radius", &moonRadius, 0.5f, 5.0f);

    ImGui::Separator();
    ImGui::Text("Tasks 4 & 5: Colors");
    ImGui::ColorEdit4("Central Sphere", glm::value_ptr(objMain.color));
    ImGui::ColorEdit4("Orbiting Sphere", glm::value_ptr(objStatic.color));

    ImGui::End();

    float aspect = static_cast<float>(context.swapchain_extent.width) / 
                   static_cast<float>(context.swapchain_extent.height);

    glm::mat4 view = glm::lookAt(
        glm::vec3(0.0f, 2.5f, 7.0f),
        glm::vec3(0.0f, 0.0f, 0.0f),
        glm::vec3(0.0f, 1.0f, 0.0f)
    );
    
    glm::mat4 proj;
    if (usePerspective) {
        proj = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 100.0f);
    } else {
        proj = glm::ortho(-aspect * 3.5f, aspect * 3.5f, -3.5f, 3.5f, 0.1f, 100.0f);
    }
    proj[1][1] *= -1.0f;

    glm::vec3 mainPos = objMain.pos;

    glm::vec3 moonPos = mainPos;

    if (isPlaying) {
        moonPos += glm::vec3(
            std::cos(animTime * moonSpeed) * moonRadius, 
            0.0f, 
            std::sin(animTime * moonSpeed) * moonRadius
        );
    } else {
        moonPos += glm::vec3(moonRadius, 0.0f, 0.0f);
    }

    auto updateObject = [&](SceneObject& obj, const glm::vec3& worldPos) {
        UBO_MVP mvp{};
        mvp.view = view;
        mvp.proj = proj;

        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, worldPos);
        model = glm::rotate(model, glm::radians(obj.rot.x), glm::vec3(1.0f, 0.0f, 0.0f));
        model = glm::rotate(model, glm::radians(obj.rot.y), glm::vec3(0.0f, 1.0f, 0.0f));
        model = glm::rotate(model, glm::radians(obj.rot.z), glm::vec3(0.0f, 0.0f, 1.0f));
        model = glm::scale(model, obj.scale);
        mvp.model = model;

        void* data = nullptr;
        if (vmaMapMemory(context.allocator, obj.allocMVP, &data) == VK_SUCCESS) {
            std::memcpy(data, &mvp, sizeof(UBO_MVP));
            vmaUnmapMemory(context.allocator, obj.allocMVP);
        }

        UBO_Color uboColor = { obj.color };
        if (vmaMapMemory(context.allocator, obj.allocColor, &data) == VK_SUCCESS) {
            std::memcpy(data, &uboColor, sizeof(UBO_Color));
            vmaUnmapMemory(context.allocator, obj.allocColor);
        }
    };

    updateObject(objMain, mainPos);
    updateObject(objStatic, moonPos);
}

void application::render(const graphics::internal::FrameData& fd) {
    vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

	VkViewport viewport = { 0.0f, 0.0f, (float)context.swapchain_extent.width, (float)context.swapchain_extent.height, 0.0f, 1.0f };
	vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
	VkRect2D scissor = { {0, 0}, context.swapchain_extent };
	vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vertexBuffer, offsets);
    vkCmdBindIndexBuffer(fd.command_buffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);

    vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &objMain.descriptorSet, 0, nullptr);
    vkCmdDrawIndexed(fd.command_buffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

    vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &objStatic.descriptorSet, 0, nullptr);
    vkCmdDrawIndexed(fd.command_buffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
}


void application::shutdown() {
    auto cleanupObj = [](SceneObject& obj) {
        vmaDestroyBuffer(context.allocator, obj.uboMVP, obj.allocMVP);
        vmaDestroyBuffer(context.allocator, obj.uboColor, obj.allocColor);
    };
    cleanupObj(objMain);
    cleanupObj(objStatic);

    vmaDestroyBuffer(context.allocator, vertexBuffer, vertexBufferAlloc);
    vmaDestroyBuffer(context.allocator, indexBuffer, indexBufferAlloc);

    vkDestroyDescriptorPool(context.device, descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(context.device, descSetLayout, nullptr);
    vkDestroyPipeline(context.device, graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(context.device, pipelineLayout, nullptr);
}