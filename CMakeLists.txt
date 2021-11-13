cmake_minimum_required(VERSION 3.0)

project(MandelbrotShader)

option(ENABLE_STATIC_RUNTIME "Use static runtime library" OFF)

find_package(glm CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(GLEW REQUIRED)
find_package(Vulkan REQUIRED)
find_package(stb REQUIRED)

if(MSVC AND ENABLE_STATIC_RUNTIME)
    foreach(flag_var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
    endforeach(flag_var)
endif()

set(SHADERS ${CMAKE_CURRENT_LIST_DIR}/src/vk_shader.vert
            ${CMAKE_CURRENT_LIST_DIR}/src/vk_shader.frag
            ${CMAKE_CURRENT_LIST_DIR}/src/gl_shader.vert
            ${CMAKE_CURRENT_LIST_DIR}/src/gl_shader.frag)

set(COMPILED_SHADERS ${CMAKE_BINARY_DIR}/shaders/vk_shader.vert.spv
                     ${CMAKE_BINARY_DIR}/shaders/vk_shader.frag.spv)

add_executable(MandelbrotShader src/gl_tools.h
                                src/IMandelbrotRenderer.h
                                src/main.cpp
                                src/MandelbrotOpenGLRenderer.cpp
                                src/MandelbrotVulkanRenderer.cpp
                                src/util.h
                                ${SHADERS}
                                ${COMPILED_SHADERS})
target_link_libraries(MandelbrotShader PRIVATE glm::glm glfw GLEW::GLEW Vulkan::Vulkan)
target_compile_definitions(MandelbrotShader PRIVATE VK_USE_PLATFORM_WIN32_KHR)
target_include_directories(MandelbrotShader PRIVATE Vulkan::Vulkan)
set_property(TARGET MandelbrotShader PROPERTY CXX_STANDARD 17)

add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/shaders/vk_shader.vert.spv
                   COMMAND glslc ${CMAKE_CURRENT_LIST_DIR}/src/vk_shader.vert -o ${CMAKE_BINARY_DIR}/shaders/vk_shader.vert.spv
                   DEPENDS ${CMAKE_CURRENT_LIST_DIR}/src/vk_shader.vert)

add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/shaders/vk_shader.frag.spv
                   COMMAND glslc ${CMAKE_CURRENT_LIST_DIR}/src/vk_shader.frag -o ${CMAKE_BINARY_DIR}/shaders/vk_shader.frag.spv
                   DEPENDS ${CMAKE_CURRENT_LIST_DIR}/src/vk_shader.frag)

configure_file(${CMAKE_CURRENT_LIST_DIR}/src/gl_shader.vert ${CMAKE_BINARY_DIR}/shaders/gl_shader.vert COPYONLY)
configure_file(${CMAKE_CURRENT_LIST_DIR}/src/gl_shader.frag ${CMAKE_BINARY_DIR}/shaders/gl_shader.frag COPYONLY)

configure_file(${CMAKE_CURRENT_LIST_DIR}/colormaps/green.bmp ${CMAKE_BINARY_DIR}/colormaps/green.bmp COPYONLY)
configure_file(${CMAKE_CURRENT_LIST_DIR}/colormaps/headache.bmp ${CMAKE_BINARY_DIR}/colormaps/headache.bmp COPYONLY)
configure_file(${CMAKE_CURRENT_LIST_DIR}/colormaps/violet.bmp ${CMAKE_BINARY_DIR}/colormaps/violet.bmp COPYONLY)
configure_file(${CMAKE_CURRENT_LIST_DIR}/colormaps/white.bmp ${CMAKE_BINARY_DIR}/colormaps/white.bmp COPYONLY)