
cmake_minimum_required(VERSION 3.20)
include(FetchContent)

project(vulkan-starter-app LANGUAGES C CXX)

add_executable(${PROJECT_NAME}
    source/main.cpp
    source/graphics_internal.cpp
    source/application.cpp
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED TRUE
    CXX_EXTENSIONS FALSE
)

if(WIN32)
    if(MSVC)
        target_compile_options(${PROJECT_NAME} PRIVATE /W4 /wd4201)
        target_link_options(${PROJECT_NAME} PRIVATE
            /subsystem:windows /entry:mainCRTStartup
        )
    else()
        target_compile_options(${PROJECT_NAME} PRIVATE
            -Wall -Wextra -Wno-missing-field-initializers
        )
        target_link_options(${PROJECT_NAME} PRIVATE -mwindows)
    endif()
    target_compile_definitions(${PROJECT_NAME} PRIVATE _USE_MATH_DEFINES)
elseif(APPLE OR LINUX)
    target_compile_options(${PROJECT_NAME} PRIVATE
        -Wall -Wextra -Wno-missing-field-initializers
    )
else()
    message(FATAL_ERROR "${CMAKE_SYSTEM_NAME} is not supported")
endif()

FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.5.1
)

FetchContent_Declare(
    vk-bootstrap
    GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap.git
    GIT_TAG v1.4.321
)

FetchContent_Declare(
    VulkanMemoryAllocator
    GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
    GIT_TAG v3.4.0
)

FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.92.9
)

find_package(Vulkan REQUIRED)

set(GLFW_LIBRARY_TYPE STATIC)
set(GLFW_BUILD_EXAMPLES OFF)
set(GLFW_BUILD_TESTS OFF)
set(GLFW_BUILD_DOCS OFF)

FetchContent_MakeAvailable(
    glfw
    vk-bootstrap
    VulkanMemoryAllocator
    imgui
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    glfw
    Vulkan::Vulkan
    vk-bootstrap::vk-bootstrap
    GPUOpen::VulkanMemoryAllocator
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
)

target_sources(${PROJECT_NAME} PRIVATE
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
)

find_program(GLSLC_EXECUTABLE glslc
    HINTS "$ENV{VULKAN_SDK}/Bin"
)

if(NOT GLSLC_EXECUTABLE)
    message(FATAL_ERROR
        "glslc not found. Check Vulkan SDK installation."
    )
endif()

set(SHADER_OUTPUTS)

function(compile_shader SOURCE_FILE OUTPUT_FILE)
    set(SOURCE_PATH
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SOURCE_FILE}"
    )
    set(OUTPUT_PATH
        "${CMAKE_CURRENT_BINARY_DIR}/shaders/${OUTPUT_FILE}"
    )

    get_filename_component(OUTPUT_DIR "${OUTPUT_PATH}" DIRECTORY)
    file(MAKE_DIRECTORY "${OUTPUT_DIR}")

    add_custom_command(
        OUTPUT "${OUTPUT_PATH}"
        COMMAND "${GLSLC_EXECUTABLE}" "${SOURCE_PATH}" -o "${OUTPUT_PATH}"
        DEPENDS "${SOURCE_PATH}"
        COMMENT "Compiling ${SOURCE_FILE}"
        VERBATIM
    )

    set(SHADER_OUTPUTS ${SHADER_OUTPUTS} "${OUTPUT_PATH}" PARENT_SCOPE)
endfunction()

compile_shader(vert.vert vert.spv)
compile_shader(frag.frag frag.spv)

add_custom_target(shaders DEPENDS ${SHADER_OUTPUTS})
add_dependencies(${PROJECT_NAME} shaders)
