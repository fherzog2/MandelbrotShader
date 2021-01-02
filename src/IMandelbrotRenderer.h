// SPDX-License-Identifier: GPL-2.0-only
#pragma once

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>

#include <GL/glew.h>

#define NOMINMAX
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vector>
#include <memory>

struct ImageData
{
    ImageData() = default;
    ImageData(const ImageData& other) = default;
    ImageData& operator=(const ImageData& other) = default;
    ImageData(ImageData&& other) = default;
    ImageData& operator=(ImageData&& other) = default;

    operator bool() const
    {
        return width > 0 && height > 0 && !bytes.empty();
    }

    int width;
    int height;
    std::vector<uint8_t> bytes;
};

class IMandelbrotRenderer
{
public:
    virtual ~IMandelbrotRenderer() {}

    virtual void setCenter(glm::dvec2 center) = 0;
    virtual glm::dvec2 getCenter() const = 0;

    virtual void setScale(glm::float64 scale) = 0;
    virtual glm::float64 getScale() const = 0;

    virtual void setIterations(glm::int32 iterations) = 0;
    virtual glm::int32 getIterations() const = 0;

    virtual void setColormapIndex(int colormap_index) = 0;
    virtual int getColormapIndex() const = 0;

    virtual void notifyFramebufferResized() = 0;

    virtual void drawFrame() = 0;
    virtual void waitIdle() const = 0;
};

std::unique_ptr<IMandelbrotRenderer> createVulkanRenderer(GLFWwindow* window, const std::vector<ImageData>& colormaps);
std::unique_ptr<IMandelbrotRenderer> createOpenGLRenderer(GLFWwindow* window, const std::vector<ImageData>& colormaps);