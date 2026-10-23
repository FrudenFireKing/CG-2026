#include <cstdint>
#include <climits>
#include <iostream>
#include <algorithm>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <backends/imgui_impl_vulkan.h>

#include "graphics_internal.hpp"
#include "application.hpp"

namespace {

constexpr int32_t default_window_width = 1280;
constexpr int32_t default_window_height = 720;

constexpr char default_window_title[] = "lab 2 - Camera and Lightin";

GLFWwindow* glfw_window = nullptr;
double lastFrameTime = 0.0;
double lastMouseX = 0.0;
double lastMouseY = 0.0;

bool firstMouseEvent = true;
bool cameraMouseControl = false;

} // namespace

int main() {
    int status = EXIT_SUCCESS;

    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return EXIT_FAILURE;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    glfw_window = glfwCreateWindow(default_window_width, default_window_height,
                                   default_window_title, nullptr, nullptr);

    uint32_t extensionCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
    std::cout << extensionCount << " extensions supported\n";
    
    if (glfw_window == nullptr) {
        status = EXIT_FAILURE;
        goto err_null_window;
    }

    glfwSetFramebufferSizeCallback(glfw_window, [](GLFWwindow*, int width, int height){
        if (width == 0 || height == 0) {
            return;
        }

        graphics::internal::resize(width, height);
    });

    if (ImGui::CreateContext() == nullptr) {
        std::cerr << "Failed to create ImGUI context\n";
        status = EXIT_FAILURE;
        goto err_imgui_init;
    }

    if (!ImGui_ImplGlfw_InitForVulkan(glfw_window, true)) {
        std::cerr << "Failed to initialize ImGUI GLFW backend for Vulkan renderer\n";
        status = EXIT_FAILURE;
        goto err_imgui_glfw_init;
    }

    if (!graphics::internal::initialize(glfw_window)) {
        std::cerr << "Failed to initialize graphics\n";
        status = EXIT_FAILURE;
        goto err_graphics_init;
    }

    if (!application::initialize()) {
        std::cerr << "Failed to initialize application\n";
        status = EXIT_FAILURE;
        goto err_application_init;
    }

    std::cout << "ImGui mouse input enabled: "
          << !ImGui::GetIO().WantCaptureMouse
          << std::endl;

    lastFrameTime = glfwGetTime();

    glfwSetMouseButtonCallback(
        glfw_window,
        [](GLFWwindow* window, int button, int action, int) {
            if (action == GLFW_PRESS) {
                std::cout << "Mouse pressed: "
                        << button << std::endl;
            }

            if (button == GLFW_MOUSE_BUTTON_RIGHT) {
                cameraMouseControl = (action == GLFW_PRESS);
                firstMouseEvent = true;
            }
        }
    );

    glfwSetCursorPosCallback(
        glfw_window,
        [](GLFWwindow* window, double xpos, double ypos) {
            if (!cameraMouseControl ||
                ImGui::GetIO().WantCaptureMouse) {
                firstMouseEvent = true;
                return;
            }

            if (firstMouseEvent) {
                lastMouseX = xpos;
                lastMouseY = ypos;
                firstMouseEvent = false;
                return;
            }

            const float dx = static_cast<float>(xpos - lastMouseX);
            const float dy = static_cast<float>(lastMouseY - ypos);

            lastMouseX = xpos;
            lastMouseY = ypos;

            application::rotateCamera(dx, dy);
        }
    );

    while (!glfwWindowShouldClose(glfw_window)) {
        int width = 0, height = 0;
        glfwGetFramebufferSize(glfw_window, &width, &height);
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(glfw_window, &width, &height);
            glfwWaitEvents();
        }

        
        glfwPollEvents();

        const double currentTime = glfwGetTime();
        const double time = currentTime;
        float deltaTime = static_cast<float>(currentTime - lastFrameTime);
        lastFrameTime = currentTime;

        // Не допускаем слишком большого скачка движения,
        // если приложение было приостановлено.
        deltaTime = std::min(deltaTime, 0.1f);

        if (!ImGui::GetIO().WantCaptureKeyboard) {
            application::processInput(
                deltaTime,
                glfwGetKey(glfw_window, GLFW_KEY_W) == GLFW_PRESS,
                glfwGetKey(glfw_window, GLFW_KEY_S) == GLFW_PRESS,
                glfwGetKey(glfw_window, GLFW_KEY_A) == GLFW_PRESS,
                glfwGetKey(glfw_window, GLFW_KEY_D) == GLFW_PRESS,
                glfwGetKey(glfw_window, GLFW_KEY_SPACE) == GLFW_PRESS,
                glfwGetKey(glfw_window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS,
                glfwGetKey(glfw_window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                glfwGetKey(glfw_window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS
            );
        }

        ImGui_ImplVulkan_NewFrame();

        ImGui_ImplGlfw_NewFrame();

        ImGui::NewFrame();
        application::update(time);
        ImGui::Render();

        graphics::internal::FrameData fd = graphics::internal::prepare();
        application::render(fd);
        graphics::internal::submitAndPresent();
    }

    application::shutdown();
err_application_init:
    graphics::internal::shutdown();
err_graphics_init:
    ImGui_ImplGlfw_Shutdown();
err_imgui_glfw_init:
    ImGui::DestroyContext();
err_imgui_init:
    glfwDestroyWindow(glfw_window);
err_null_window:
    glfwTerminate();

    return status;
}