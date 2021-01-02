// SPDX-License-Identifier: GPL-2.0-only
#include "IMandelbrotRenderer.h"
#include "util.h"

#include <vulkan/vulkan.hpp>
#include <optional>
#include <set>
#include <fstream>
#include <iostream>

const int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<const char*> validation_layers = {
    "VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> device_extensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

#ifdef NDEBUG
const bool enable_validation_layers = false;
#else
const bool enable_validation_layers = true;
#endif

PFN_vkCreateDebugUtilsMessengerEXT pfnVkCreateDebugUtilsMessengerEXT = nullptr;
PFN_vkDestroyDebugUtilsMessengerEXT pfnVkDestroyDebugUtilsMessengerEXT = nullptr;

VKAPI_ATTR VkResult VKAPI_CALL vkCreateDebugUtilsMessengerEXT(VkInstance instance,
    const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkDebugUtilsMessengerEXT* pMessenger)
{
    if (pfnVkCreateDebugUtilsMessengerEXT)
        return pfnVkCreateDebugUtilsMessengerEXT(instance, pCreateInfo, pAllocator, pMessenger);

    return VK_ERROR_EXTENSION_NOT_PRESENT;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyDebugUtilsMessengerEXT(VkInstance instance,
    VkDebugUtilsMessengerEXT messenger,
    VkAllocationCallbacks const* pAllocator)
{
    if (pfnVkDestroyDebugUtilsMessengerEXT)
        pfnVkDestroyDebugUtilsMessengerEXT(instance, messenger, pAllocator);
}

struct QueueFamilyIndices {
    std::optional<uint32_t> graphics_family;
    std::optional<uint32_t> present_family;

    bool isComplete() {
        return graphics_family.has_value() && present_family.has_value();
    }
};

struct SwapChainSupportDetails {
    vk::SurfaceCapabilitiesKHR capabilities;
    std::vector<vk::SurfaceFormatKHR> formats;
    std::vector<vk::PresentModeKHR> present_modes;
};

struct Vertex {
    glm::vec2 pos;
    glm::vec2 texcoord;

    static vk::VertexInputBindingDescription getBindingDescription() {
        return vk::VertexInputBindingDescription(0, sizeof(Vertex), vk::VertexInputRate::eVertex);
    }

    static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescriptions() {
        std::array<vk::VertexInputAttributeDescription, 2> attribute_descriptions;

        auto set_attribute = [&attribute_descriptions](uint32_t i, vk::Format format, uint32_t offset) {
            attribute_descriptions[i].binding = 0;
            attribute_descriptions[i].location = i;
            attribute_descriptions[i].format = format;
            attribute_descriptions[i].offset = offset;
        };

        set_attribute(0, vk::Format::eR32G32Sfloat, offsetof(Vertex, pos));
        set_attribute(1, vk::Format::eR32G32Sfloat, offsetof(Vertex, texcoord));

        return attribute_descriptions;
    }
};

struct UniformBufferObject {
    alignas(8) glm::dvec2 center;
    alignas(8) glm::float64 scale = 1;
    alignas(4) glm::int32 iterations = 5;
    alignas(8) glm::float64 aspect = 1;
    alignas(4) glm::float32 colormap_index = 0;
};

const std::vector<Vertex> vertices = {
    {{-1, -1}, {0, 0}},
    {{ 1, -1}, {1, 0}},
    {{ 1,  1}, {1, 1}},
    {{-1,  1}, {0, 1}}
};

const std::vector<uint16_t> indices = {
    1, 0, 3, 1, 3, 2
};

class MandelbrotVulkanRenderer : public IMandelbrotRenderer {
public:
    MandelbrotVulkanRenderer(GLFWwindow* window, const std::vector<ImageData>& colormaps);

    void setCenter(glm::dvec2 center) override;
    glm::dvec2 getCenter() const override;

    void setScale(glm::float64 scale) override;
    glm::float64 getScale() const override;

    void setIterations(glm::int32 iterations) override;
    glm::int32 getIterations() const override;

    void setColormapIndex(int colormap_index) override;
    int getColormapIndex() const override;

    void notifyFramebufferResized() override;

    void drawFrame() override;
    void waitIdle() const override;

private:
    GLFWwindow* _window;

    vk::UniqueInstance _instance;
    vk::UniqueDebugUtilsMessengerEXT _debug_messenger;
    vk::UniqueSurfaceKHR _surface;

    vk::PhysicalDevice _physical_device;
    vk::UniqueDevice _device;

    vk::Queue _graphics_queue;
    vk::Queue _present_queue;

    vk::UniqueSwapchainKHR _swap_chain;
    std::vector<vk::Image> _swap_chain_images;
    vk::Format _swap_chain_image_format;
    vk::Extent2D _swap_chain_extent;
    std::vector<vk::UniqueImageView> _swap_chain_image_views;

    vk::UniqueRenderPass _render_pass;
    std::vector<vk::UniqueFramebuffer> _swap_chain_framebuffers;

    vk::UniqueDescriptorSetLayout _descriptor_set_layout;
    vk::UniquePipelineLayout _pipeline_layout;
    vk::UniquePipeline _graphics_pipeline;

    vk::UniqueCommandPool _command_pool;

    vk::UniqueBuffer _vertex_buffer;
    vk::UniqueDeviceMemory _vertex_buffer_memory;
    vk::UniqueBuffer _index_buffer;
    vk::UniqueDeviceMemory _index_buffer_memory;

    vk::UniqueImage _colormap_array;
    vk::UniqueDeviceMemory _colormap_array_memory;
    vk::UniqueImageView _colormap_array_view;
    vk::UniqueSampler _colormap_array_sampler;

    std::vector<vk::UniqueBuffer> _uniform_buffers;
    std::vector<vk::UniqueDeviceMemory> _uniform_buffers_memory;

    vk::UniqueDescriptorPool _descriptor_pool;
    std::vector<vk::DescriptorSet> _descriptor_sets;

    std::vector<vk::UniqueCommandBuffer> _command_buffers;

    std::vector<vk::UniqueSemaphore> _image_available_semaphores;
    std::vector<vk::UniqueSemaphore> _render_finished_semaphores;
    std::vector<vk::UniqueFence> _in_flight_fences;
    std::vector<vk::Fence> _images_in_flight;
    size_t _current_frame = 0;

    UniformBufferObject ubo;

    bool _framebuffer_resized = false;

    void initVulkan(const std::vector<ImageData>& colormaps);
    void cleanupSwapChain();
    void recreateSwapChain();

    void createInstance();
    void populateDebugMessengerCreateInfo(vk::DebugUtilsMessengerCreateInfoEXT& create_info);
    void setupDebugMessenger();
    void createSurface();
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createSwapChain();
    void createImageViews();
    void createRenderPass();
    void createDescriptorSetLayout();
    void createGraphicsPipeline();
    void createFramebuffers();
    void createCommandPool();
    void createVertexBuffer();
    void createIndexBuffer();

    void transitionImageLayout(const vk::CommandBuffer command_buffer, const vk::Image image, const vk::ImageLayout old_layout, const vk::ImageLayout new_layout, uint32_t layer_count);

    void create2dTextureArray(const std::vector<ImageData>& texture_data,
        vk::UniqueImage& texture, vk::UniqueDeviceMemory& texture_memory, vk::UniqueImageView& texture_view);

    void createTextureSampler();
    void createUniformBuffers();
    void createDescriptorPool();
    void createDescriptorSets();

    void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::UniqueBuffer& buffer, vk::UniqueDeviceMemory& buffer_memory);
    void copyBuffer(vk::Buffer src_buffer, vk::Buffer dst_buffer, vk::DeviceSize size);

    vk::UniqueCommandBuffer createInitCommandBuffer() const;
    void submitInitCommandBuffer(vk::CommandBuffer command_buffer) const;

    uint32_t findMemoryType(uint32_t type_filter, vk::MemoryPropertyFlags properties);

    void createCommandBuffers();
    void createSyncObjects();

    void updateUniformBuffer(uint32_t current_image);

    vk::UniqueShaderModule createShaderModule(const std::vector<char>& code);
    vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& available_formats);
    vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& available_present_modes);
    vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities);
    SwapChainSupportDetails querySwapChainSupport(vk::PhysicalDevice device);
    bool isDeviceSuitable(vk::PhysicalDevice device);
    bool checkDeviceExtensionSupport(vk::PhysicalDevice device);
    QueueFamilyIndices findQueueFamilies(vk::PhysicalDevice device);
    std::vector<const char*> getRequiredExtensions();
    bool checkValidationLayerSupport();

    static std::vector<char> readFile(const std::string& filename);

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT message_severity, VkDebugUtilsMessageTypeFlagsEXT message_type, const VkDebugUtilsMessengerCallbackDataEXT* callback_data, void* user_data);
};

MandelbrotVulkanRenderer::MandelbrotVulkanRenderer(GLFWwindow* window, const std::vector<ImageData>& colormaps)
    : _window(window)
{
    initVulkan(colormaps);
}

void MandelbrotVulkanRenderer::setCenter(glm::dvec2 center)
{
    ubo.center = center;
}

glm::dvec2 MandelbrotVulkanRenderer::getCenter() const
{
    return ubo.center;
}

void MandelbrotVulkanRenderer::setScale(glm::float64 scale)
{
    ubo.scale = scale;
}

glm::float64 MandelbrotVulkanRenderer::getScale() const
{
    return ubo.scale;
}

void MandelbrotVulkanRenderer::setIterations(glm::int32 iterations)
{
    ubo.iterations = iterations;
}

glm::int32 MandelbrotVulkanRenderer::getIterations() const
{
    return ubo.iterations;
}

void MandelbrotVulkanRenderer::setColormapIndex(int colormap_index)
{
    ubo.colormap_index = static_cast<glm::float32>(colormap_index);
}

int MandelbrotVulkanRenderer::getColormapIndex() const
{
    return static_cast<int>(ubo.colormap_index);
}

void MandelbrotVulkanRenderer::notifyFramebufferResized()
{
    _framebuffer_resized = true;
}

void MandelbrotVulkanRenderer::waitIdle() const
{
    _device->waitIdle();
}

void MandelbrotVulkanRenderer::initVulkan(const std::vector<ImageData>& colormaps) {
    createInstance();
    setupDebugMessenger();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapChain();
    createImageViews();
    createRenderPass();
    createDescriptorSetLayout();
    createGraphicsPipeline();
    createFramebuffers();
    createCommandPool();
    createVertexBuffer();
    createIndexBuffer();
    create2dTextureArray(colormaps, _colormap_array, _colormap_array_memory, _colormap_array_view);
    createTextureSampler();
    createUniformBuffers();
    createDescriptorPool();
    createDescriptorSets();
    createCommandBuffers();
    createSyncObjects();
}

void MandelbrotVulkanRenderer::cleanupSwapChain() {
    _descriptor_pool.reset();

    _command_buffers.clear();

    _uniform_buffers.clear();
    _uniform_buffers_memory.clear();

    _graphics_pipeline.reset();
    _pipeline_layout.reset();
    _swap_chain_framebuffers.clear();

    _render_pass.reset();
    _swap_chain_image_views.clear();

    _swap_chain.reset();
}

void MandelbrotVulkanRenderer::recreateSwapChain() {
    int width = 0, height = 0;
    glfwGetFramebufferSize(_window, &width, &height);
    while (width == 0 || height == 0) {
        glfwGetFramebufferSize(_window, &width, &height);
        glfwWaitEvents();
    }

    _device->waitIdle();

    cleanupSwapChain();

    createSwapChain();
    createImageViews();
    createRenderPass();
    createGraphicsPipeline();
    createFramebuffers();
    createUniformBuffers();
    createDescriptorPool();
    createDescriptorSets();
    createCommandBuffers();
}

void MandelbrotVulkanRenderer::createInstance() {
    if (enable_validation_layers && !checkValidationLayerSupport()) {
        throw std::runtime_error("validation layers requested, but not available!");
    }

    vk::ApplicationInfo app_info("MandelbrotShader",
        VK_MAKE_VERSION(1, 0, 0),
        "No Engine",
        VK_MAKE_VERSION(1, 0, 0),
        VK_API_VERSION_1_0);

    auto extensions = getRequiredExtensions();

    vk::InstanceCreateInfo instance_create_info({}, &app_info, {}, {},
        static_cast<uint32_t>(extensions.size()), extensions.data());

    vk::DebugUtilsMessengerCreateInfoEXT debug_create_info;
    populateDebugMessengerCreateInfo(debug_create_info);
    if (enable_validation_layers) {
        instance_create_info.enabledLayerCount = static_cast<uint32_t>(validation_layers.size());
        instance_create_info.ppEnabledLayerNames = validation_layers.data();
        instance_create_info.pNext = &debug_create_info;
    }

    _instance = vk::createInstanceUnique(instance_create_info);
    if (!_instance) {
        throw std::runtime_error("failed to create instance!");
    }
}

void MandelbrotVulkanRenderer::populateDebugMessengerCreateInfo(vk::DebugUtilsMessengerCreateInfoEXT& createInfo) {
    createInfo.messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
    createInfo.messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance;
    createInfo.pfnUserCallback = debugCallback;
}

void MandelbrotVulkanRenderer::setupDebugMessenger() {
    if (!enable_validation_layers) return;

    pfnVkCreateDebugUtilsMessengerEXT = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT> (_instance->getProcAddr("vkCreateDebugUtilsMessengerEXT"));
    pfnVkDestroyDebugUtilsMessengerEXT = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(_instance->getProcAddr("vkDestroyDebugUtilsMessengerEXT"));

    vk::DebugUtilsMessengerCreateInfoEXT create_info;
    populateDebugMessengerCreateInfo(create_info);

    _debug_messenger = _instance->createDebugUtilsMessengerEXTUnique(create_info);
    if (!_debug_messenger) {
        throw std::runtime_error("failed to set up debug messenger!");
    }
}

void MandelbrotVulkanRenderer::createSurface() {
    VkSurfaceKHR surface;
    if (glfwCreateWindowSurface(*_instance, _window, nullptr, &surface) != VK_SUCCESS) {
        throw std::runtime_error("failed to create window surface!");
    }

    _surface = vk::UniqueSurfaceKHR(surface, vk::ObjectDestroy(*_instance, nullptr, VULKAN_HPP_DEFAULT_DISPATCHER));
}

void MandelbrotVulkanRenderer::pickPhysicalDevice() {
    std::vector<vk::PhysicalDevice> devices = _instance->enumeratePhysicalDevices();

    for (const auto& device : devices) {
        if (isDeviceSuitable(device)) {
            _physical_device = device;
            return;
        }
    }

    throw std::runtime_error("failed to find a suitable GPU!");
}

void MandelbrotVulkanRenderer::createLogicalDevice() {
    QueueFamilyIndices indices = findQueueFamilies(_physical_device);

    std::vector<vk::DeviceQueueCreateInfo> queue_create_infos;
    std::set<uint32_t> unique_queue_families = { indices.graphics_family.value(), indices.present_family.value() };

    float queuePriority = 1.0f;
    for (uint32_t queueFamily : unique_queue_families) {
        queue_create_infos.push_back(vk::DeviceQueueCreateInfo({}, queueFamily, 1, &queuePriority));
    }

    vk::PhysicalDeviceFeatures device_features;
    device_features.shaderFloat64 = VK_TRUE;
    device_features.samplerAnisotropy = VK_TRUE;

    vk::DeviceCreateInfo create_info({},
        static_cast<uint32_t>(queue_create_infos.size()), queue_create_infos.data(),
        {}, {},
        static_cast<uint32_t>(device_extensions.size()), device_extensions.data(),
        & device_features);

    if (enable_validation_layers) {
        create_info.enabledLayerCount = static_cast<uint32_t>(validation_layers.size());
        create_info.ppEnabledLayerNames = validation_layers.data();
    }

    _device = _physical_device.createDeviceUnique(create_info);
    if (!_device) {
        throw std::runtime_error("failed to create logical device!");
    }

    _graphics_queue = _device->getQueue(indices.graphics_family.value(), 0);
    _present_queue = _device->getQueue(indices.present_family.value(), 0);
}

void MandelbrotVulkanRenderer::createSwapChain() {
    SwapChainSupportDetails swap_chain_support = querySwapChainSupport(_physical_device);

    vk::SurfaceFormatKHR surface_format = chooseSwapSurfaceFormat(swap_chain_support.formats);
    vk::PresentModeKHR present_mode = chooseSwapPresentMode(swap_chain_support.present_modes);
    vk::Extent2D extent = chooseSwapExtent(swap_chain_support.capabilities);

    uint32_t image_count = swap_chain_support.capabilities.minImageCount + 1;
    if (swap_chain_support.capabilities.maxImageCount > 0 && image_count > swap_chain_support.capabilities.maxImageCount) {
        image_count = swap_chain_support.capabilities.maxImageCount;
    }

    vk::SwapchainCreateInfoKHR create_info;
    create_info.surface = *_surface;

    create_info.minImageCount = image_count;
    create_info.imageFormat = surface_format.format;
    create_info.imageColorSpace = surface_format.colorSpace;
    create_info.imageExtent = extent;
    create_info.imageArrayLayers = 1;
    create_info.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;

    QueueFamilyIndices indices = findQueueFamilies(_physical_device);
    std::array<uint32_t, 2> queue_family_indices = { indices.graphics_family.value(), indices.present_family.value() };

    if (indices.graphics_family != indices.present_family) {
        create_info.imageSharingMode = vk::SharingMode::eConcurrent;
        create_info.queueFamilyIndexCount = static_cast<uint32_t>(queue_family_indices.size());
        create_info.pQueueFamilyIndices = queue_family_indices.data();
    }
    else {
        create_info.imageSharingMode = vk::SharingMode::eExclusive;
    }

    create_info.preTransform = vk::SurfaceTransformFlagBitsKHR(swap_chain_support.capabilities.currentTransform);
    create_info.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
    create_info.presentMode = present_mode;
    create_info.clipped = VK_TRUE;

    _swap_chain = _device->createSwapchainKHRUnique(create_info);
    if (!_swap_chain) {
        throw std::runtime_error("failed to create swap chain!");
    }

    _swap_chain_images = _device->getSwapchainImagesKHR(*_swap_chain);

    _swap_chain_image_format = surface_format.format;
    _swap_chain_extent = extent;
}

void MandelbrotVulkanRenderer::createImageViews() {
    for (const auto& image : _swap_chain_images) {
        vk::ImageViewCreateInfo create_info({}, image,
            vk::ImageViewType::e2D,
            _swap_chain_image_format,
            vk::ComponentMapping(),
            vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));

        _swap_chain_image_views.push_back(_device->createImageViewUnique(create_info));

        if (!_swap_chain_image_views.back()) {
            throw std::runtime_error("failed to create image views!");
        }
    }
}

void MandelbrotVulkanRenderer::createRenderPass() {
    vk::AttachmentDescription color_attachment;
    color_attachment.format = _swap_chain_image_format;
    color_attachment.samples = vk::SampleCountFlagBits::e1;
    color_attachment.loadOp = vk::AttachmentLoadOp::eClear;
    color_attachment.storeOp = vk::AttachmentStoreOp::eStore;
    color_attachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
    color_attachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
    color_attachment.initialLayout = vk::ImageLayout::eUndefined;
    color_attachment.finalLayout = vk::ImageLayout::ePresentSrcKHR;

    vk::AttachmentReference color_attachment_ref;
    color_attachment_ref.attachment = 0;
    color_attachment_ref.layout = vk::ImageLayout::eColorAttachmentOptimal;

    vk::SubpassDescription subpass;
    subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_attachment_ref;

    vk::SubpassDependency dependency;
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    dependency.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    dependency.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;

    vk::RenderPassCreateInfo render_pass_info({},
        1, &color_attachment,
        1, &subpass,
        1, &dependency);

    _render_pass = _device->createRenderPassUnique(render_pass_info);
    if (!_render_pass) {
        throw std::runtime_error("failed to create render pass!");
    }
}

void MandelbrotVulkanRenderer::createDescriptorSetLayout() {
    const std::array< vk::DescriptorSetLayoutBinding, 2> bindings{
        vk::DescriptorSetLayoutBinding()
            .setBinding(0)
            .setDescriptorCount(1)
            .setDescriptorType(vk::DescriptorType::eUniformBuffer)
            .setStageFlags(vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment),
        vk::DescriptorSetLayoutBinding()
            .setBinding(1)
            .setDescriptorCount(1)
            .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
            .setStageFlags(vk::ShaderStageFlagBits::eFragment)
    };

    vk::DescriptorSetLayoutCreateInfo layout_info;
    layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
    layout_info.pBindings = bindings.data();

    _descriptor_set_layout = _device->createDescriptorSetLayoutUnique(layout_info);
    if (!_descriptor_set_layout) {
        throw std::runtime_error("failed to create descriptor set layout!");
    }
}

void MandelbrotVulkanRenderer::createGraphicsPipeline() {
    auto vert_shader_code = readFile("shaders/vk_shader.vert.spv");
    auto frag_shader_code = readFile("shaders/vk_shader.frag.spv");

    vk::UniqueShaderModule vert_shader_module = createShaderModule(vert_shader_code);
    vk::UniqueShaderModule frag_shader_module = createShaderModule(frag_shader_code);

    vk::PipelineShaderStageCreateInfo vert_shader_stage_info;
    vert_shader_stage_info.stage = vk::ShaderStageFlagBits::eVertex;
    vert_shader_stage_info.module = *vert_shader_module;
    vert_shader_stage_info.pName = "main";

    vk::PipelineShaderStageCreateInfo frag_shader_stage_info;
    frag_shader_stage_info.stage = vk::ShaderStageFlagBits::eFragment;
    frag_shader_stage_info.module = *frag_shader_module;
    frag_shader_stage_info.pName = "main";

    vk::PipelineShaderStageCreateInfo shader_stages[] = { vert_shader_stage_info, frag_shader_stage_info };

    vk::PipelineVertexInputStateCreateInfo vertex_input_info;

    auto binding_description = Vertex::getBindingDescription();
    auto attribute_descriptions = Vertex::getAttributeDescriptions();

    vertex_input_info.vertexBindingDescriptionCount = 1;
    vertex_input_info.pVertexBindingDescriptions = &binding_description;
    vertex_input_info.vertexAttributeDescriptionCount = static_cast<uint32_t>(attribute_descriptions.size());
    vertex_input_info.pVertexAttributeDescriptions = attribute_descriptions.data();

    vk::PipelineInputAssemblyStateCreateInfo input_assembly;
    input_assembly.topology = vk::PrimitiveTopology::eTriangleList;
    input_assembly.primitiveRestartEnable = VK_FALSE;

    vk::Viewport viewport;
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float)_swap_chain_extent.width;
    viewport.height = (float)_swap_chain_extent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    vk::Rect2D scissor;
    scissor.offset = { 0, 0 };
    scissor.extent = _swap_chain_extent;

    vk::PipelineViewportStateCreateInfo viewport_state;
    viewport_state.viewportCount = 1;
    viewport_state.pViewports = &viewport;
    viewport_state.scissorCount = 1;
    viewport_state.pScissors = &scissor;

    vk::PipelineRasterizationStateCreateInfo rasterizer;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = vk::PolygonMode::eFill;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = vk::CullModeFlagBits::eBack;
    rasterizer.frontFace = vk::FrontFace::eCounterClockwise;
    rasterizer.depthBiasEnable = VK_FALSE;

    vk::PipelineMultisampleStateCreateInfo multisampling;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = vk::SampleCountFlagBits::e1;

    vk::PipelineColorBlendAttachmentState color_blend_attachment{};
    color_blend_attachment.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
    color_blend_attachment.blendEnable = VK_FALSE;

    vk::PipelineColorBlendStateCreateInfo color_blending;
    color_blending.logicOpEnable = VK_FALSE;
    color_blending.logicOp = vk::LogicOp::eCopy;
    color_blending.attachmentCount = 1;
    color_blending.pAttachments = &color_blend_attachment;
    color_blending.blendConstants[0] = 0.0f;
    color_blending.blendConstants[1] = 0.0f;
    color_blending.blendConstants[2] = 0.0f;
    color_blending.blendConstants[3] = 0.0f;

    vk::PipelineLayoutCreateInfo pipeline_layout_info;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &*_descriptor_set_layout;

    _pipeline_layout = _device->createPipelineLayoutUnique(pipeline_layout_info);
    if (!_pipeline_layout) {
        throw std::runtime_error("failed to create pipeline layout!");
    }

    vk::GraphicsPipelineCreateInfo pipeline_info;
    pipeline_info.stageCount = 2;
    pipeline_info.pStages = shader_stages;
    pipeline_info.pVertexInputState = &vertex_input_info;
    pipeline_info.pInputAssemblyState = &input_assembly;
    pipeline_info.pViewportState = &viewport_state;
    pipeline_info.pRasterizationState = &rasterizer;
    pipeline_info.pMultisampleState = &multisampling;
    pipeline_info.pColorBlendState = &color_blending;
    pipeline_info.layout = *_pipeline_layout;
    pipeline_info.renderPass = *_render_pass;
    pipeline_info.subpass = 0;

    vk::PipelineCache cache;

    _graphics_pipeline = _device->createGraphicsPipelineUnique(cache, pipeline_info);
    if (!_graphics_pipeline) {
        throw std::runtime_error("failed to create graphics pipeline!");
    }
}

void MandelbrotVulkanRenderer::createFramebuffers() {
    for (const auto& image_view : _swap_chain_image_views) {
        vk::ImageView attachments[] = {
            *image_view
        };

        vk::FramebufferCreateInfo framebuffer_info;
        framebuffer_info.renderPass = *_render_pass;
        framebuffer_info.attachmentCount = 1;
        framebuffer_info.pAttachments = attachments;
        framebuffer_info.width = _swap_chain_extent.width;
        framebuffer_info.height = _swap_chain_extent.height;
        framebuffer_info.layers = 1;

        _swap_chain_framebuffers.push_back(_device->createFramebufferUnique(framebuffer_info));

        if (!_swap_chain_framebuffers.back()) {
            throw std::runtime_error("failed to create framebuffer!");
        }
    }
}

void MandelbrotVulkanRenderer::createCommandPool() {
    QueueFamilyIndices queue_family_indices = findQueueFamilies(_physical_device);

    vk::CommandPoolCreateInfo pool_info;
    pool_info.queueFamilyIndex = queue_family_indices.graphics_family.value();

    _command_pool = _device->createCommandPoolUnique(pool_info);
    if (!_command_pool) {
        throw std::runtime_error("failed to create graphics command pool!");
    }
}

void MandelbrotVulkanRenderer::createVertexBuffer() {
    vk::DeviceSize buffer_size = sizeof(vertices[0]) * vertices.size();

    vk::UniqueBuffer staging_buffer;
    vk::UniqueDeviceMemory staging_buffer_memory;
    createBuffer(buffer_size,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
        staging_buffer, staging_buffer_memory);

    {
        void* data = _device->mapMemory(*staging_buffer_memory, 0, buffer_size);
        FinalAction scoped_unmap([this, &staging_buffer_memory]() { _device->unmapMemory(*staging_buffer_memory); });

        memcpy(data, vertices.data(), (size_t)buffer_size);
    }

    createBuffer(buffer_size,
        vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        _vertex_buffer, _vertex_buffer_memory);

    copyBuffer(*staging_buffer, *_vertex_buffer, buffer_size);
}

void MandelbrotVulkanRenderer::createIndexBuffer() {
    vk::DeviceSize buffer_size = sizeof(indices[0]) * indices.size();

    vk::UniqueBuffer staging_buffer;
    vk::UniqueDeviceMemory staging_buffer_memory;
    createBuffer(buffer_size,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
        staging_buffer, staging_buffer_memory);

    {
        void* data = _device->mapMemory(*staging_buffer_memory, 0, buffer_size);
        FinalAction scoped_unmap([this, &staging_buffer_memory]() { _device->unmapMemory(*staging_buffer_memory); });

        memcpy(data, indices.data(), (size_t)buffer_size);
    }

    createBuffer(buffer_size,
        vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        _index_buffer, _index_buffer_memory);

    copyBuffer(*staging_buffer, *_index_buffer, buffer_size);
}

void MandelbrotVulkanRenderer::transitionImageLayout(const vk::CommandBuffer command_buffer, const vk::Image image, const vk::ImageLayout old_layout, const vk::ImageLayout new_layout, uint32_t layer_count)
{
    vk::ImageMemoryBarrier barrier;
    barrier.oldLayout = old_layout;
    barrier.newLayout = new_layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = layer_count;

    vk::PipelineStageFlags src_stage;
    vk::PipelineStageFlags dst_stage;

    if (old_layout == vk::ImageLayout::eUndefined && new_layout == vk::ImageLayout::eTransferDstOptimal)
    {
        barrier.srcAccessMask = vk::AccessFlags();
        barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
        src_stage = vk::PipelineStageFlagBits::eTopOfPipe;
        dst_stage = vk::PipelineStageFlagBits::eTransfer;
    }
    else if (old_layout == vk::ImageLayout::eTransferDstOptimal && new_layout == vk::ImageLayout::eShaderReadOnlyOptimal)
    {
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
        src_stage = vk::PipelineStageFlagBits::eTransfer;
        dst_stage = vk::PipelineStageFlagBits::eFragmentShader;
    }

    command_buffer.pipelineBarrier(src_stage, dst_stage, {}, {}, {}, barrier);
}

void MandelbrotVulkanRenderer::create2dTextureArray(const std::vector<ImageData>& texture_data,
    vk::UniqueImage& texture, vk::UniqueDeviceMemory& texture_memory, vk::UniqueImageView& texture_view)
{
    const vk::Extent3D extent = { uint32_t(texture_data.front().width), uint32_t(texture_data.front().height), 1 };
    const uint32_t array_layers = uint32_t(texture_data.size());

    // create texture

    vk::ImageCreateInfo image_info;
    image_info.imageType = vk::ImageType::e2D;
    image_info.extent = extent;
    image_info.mipLevels = 1;
    image_info.arrayLayers = array_layers;
    image_info.format = vk::Format::eR8G8B8A8Srgb;
    image_info.tiling = vk::ImageTiling::eOptimal;
    image_info.initialLayout = vk::ImageLayout::eUndefined;
    image_info.usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled;
    image_info.samples = vk::SampleCountFlagBits::e1;
    image_info.sharingMode = vk::SharingMode::eExclusive;
    texture = _device->createImageUnique(image_info);

    vk::MemoryRequirements mem_requirements = _device->getImageMemoryRequirements(*texture);

    vk::MemoryAllocateInfo alloc_info;
    alloc_info.allocationSize = mem_requirements.size;
    alloc_info.memoryTypeIndex = findMemoryType(mem_requirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
    texture_memory = _device->allocateMemoryUnique(alloc_info);

    _device->bindImageMemory(*texture, *texture_memory, 0);

    // fill texture using a staging buffer

    {
        size_t total_buffer_size = 0;
        for (const auto& layer : texture_data)
        {
            total_buffer_size += layer.bytes.size();
        }

        std::vector<vk::BufferImageCopy> buffer_copy_regions;

        vk::UniqueBuffer staging_buffer;
        vk::UniqueDeviceMemory staging_buffer_memory;
        createBuffer(total_buffer_size,
            vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            staging_buffer, staging_buffer_memory);

        {
            void* data = _device->mapMemory(*staging_buffer_memory, 0, total_buffer_size);
            FinalAction scoped_unmap([this, &staging_buffer_memory]() { _device->unmapMemory(*staging_buffer_memory); });

            uint8_t* data_it = static_cast<uint8_t*>(data);
            uint32_t layer_index = 0;
            vk::DeviceSize buffer_offset = 0;
            for (const auto& layer : texture_data)
            {
                memcpy(data_it, layer.bytes.data(), layer.bytes.size());
                data_it += layer.bytes.size();

                vk::BufferImageCopy region;
                region.bufferOffset = buffer_offset;
                region.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
                region.imageSubresource.baseArrayLayer = layer_index;
                region.imageSubresource.layerCount = 1;
                region.imageExtent = image_info.extent;

                buffer_copy_regions.push_back(region);

                ++layer_index;
                buffer_offset += layer.bytes.size();
            }
        }

        auto command_buffer = createInitCommandBuffer();

        transitionImageLayout(*command_buffer, *texture, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, array_layers);

        command_buffer->copyBufferToImage(*staging_buffer, *texture, vk::ImageLayout::eTransferDstOptimal, buffer_copy_regions);

        transitionImageLayout(*command_buffer, *texture, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, array_layers);

        submitInitCommandBuffer(*command_buffer);
    }

    // create image view

    vk::ImageViewCreateInfo view_info;
    view_info.image = *texture;
    view_info.viewType = vk::ImageViewType::e2DArray;
    view_info.format = image_info.format;
    view_info.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.layerCount = array_layers;
    texture_view = _device->createImageViewUnique(view_info);
}

void MandelbrotVulkanRenderer::createTextureSampler() {
    vk::SamplerCreateInfo sampler_info;
    sampler_info.magFilter = vk::Filter::eLinear;
    sampler_info.minFilter = vk::Filter::eLinear;
    sampler_info.addressModeU = vk::SamplerAddressMode::eRepeat;
    sampler_info.addressModeV = vk::SamplerAddressMode::eRepeat;
    sampler_info.addressModeW = vk::SamplerAddressMode::eRepeat;
    sampler_info.anisotropyEnable = VK_TRUE;
    sampler_info.maxAnisotropy = 16;
    sampler_info.borderColor = vk::BorderColor::eIntOpaqueBlack;
    sampler_info.unnormalizedCoordinates = VK_FALSE;
    sampler_info.compareEnable = VK_FALSE;
    sampler_info.compareOp = vk::CompareOp::eAlways;
    sampler_info.mipmapMode = vk::SamplerMipmapMode::eLinear;
    _colormap_array_sampler = _device->createSamplerUnique(sampler_info);
}

void MandelbrotVulkanRenderer::createUniformBuffers() {
    vk::DeviceSize bufferSize = sizeof(UniformBufferObject);

    _uniform_buffers.resize(_swap_chain_images.size());
    _uniform_buffers_memory.resize(_swap_chain_images.size());

    for (size_t i = 0; i < _swap_chain_images.size(); i++) {
        createBuffer(bufferSize,
            vk::BufferUsageFlagBits::eUniformBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            _uniform_buffers[i], _uniform_buffers_memory[i]);
    }
}

void MandelbrotVulkanRenderer::createDescriptorPool() {
    const uint32_t num_swap_chain_images = static_cast<uint32_t>(_swap_chain_images.size());

    const std::array<vk::DescriptorPoolSize, 2> pool_sizes{
        vk::DescriptorPoolSize().setType(vk::DescriptorType::eUniformBuffer).setDescriptorCount(num_swap_chain_images),
        vk::DescriptorPoolSize().setType(vk::DescriptorType::eCombinedImageSampler).setDescriptorCount(num_swap_chain_images),
    };

    vk::DescriptorPoolCreateInfo pool_info;
    pool_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
    pool_info.pPoolSizes = pool_sizes.data();
    pool_info.maxSets = num_swap_chain_images;

    _descriptor_pool = _device->createDescriptorPoolUnique(pool_info);
    if (!_descriptor_pool) {
        throw std::runtime_error("failed to create descriptor pool!");
    }
}

void MandelbrotVulkanRenderer::createDescriptorSets() {
    std::vector<vk::DescriptorSetLayout> layouts(_swap_chain_images.size(), *_descriptor_set_layout);
    vk::DescriptorSetAllocateInfo alloc_info;
    alloc_info.descriptorPool = *_descriptor_pool;
    alloc_info.descriptorSetCount = static_cast<uint32_t>(_swap_chain_images.size());
    alloc_info.pSetLayouts = layouts.data();

    _descriptor_sets = _device->allocateDescriptorSets(alloc_info);

    for (size_t i = 0; i < _swap_chain_images.size(); i++) {
        vk::DescriptorBufferInfo buffer_info;
        buffer_info.buffer = *_uniform_buffers[i];
        buffer_info.offset = 0;
        buffer_info.range = sizeof(UniformBufferObject);

        vk::DescriptorImageInfo colormap_array_info;
        colormap_array_info.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        colormap_array_info.imageView = *_colormap_array_view;
        colormap_array_info.sampler = *_colormap_array_sampler;

        const std::array<vk::WriteDescriptorSet, 2> descriptor_writes{
            vk::WriteDescriptorSet()
                .setDstSet(_descriptor_sets[i])
                .setDstBinding(0)
                .setDstArrayElement(0)
                .setDescriptorType(vk::DescriptorType::eUniformBuffer)
                .setDescriptorCount(1)
                .setPBufferInfo(&buffer_info),
            vk::WriteDescriptorSet()
                .setDstSet(_descriptor_sets[i])
                .setDstBinding(1)
                .setDstArrayElement(0)
                .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
                .setDescriptorCount(1)
                .setPImageInfo(&colormap_array_info)
        };

        _device->updateDescriptorSets(descriptor_writes, {});
    }
}

void MandelbrotVulkanRenderer::createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::UniqueBuffer& buffer, vk::UniqueDeviceMemory& buffer_memory) {
    vk::BufferCreateInfo buffer_info;
    buffer_info.size = size;
    buffer_info.usage = usage;
    buffer_info.sharingMode = vk::SharingMode::eExclusive;

    buffer = _device->createBufferUnique(buffer_info);
    if (!buffer) {
        throw std::runtime_error("failed to create buffer!");
    }

    vk::MemoryRequirements mem_requirements = _device->getBufferMemoryRequirements(*buffer);

    vk::MemoryAllocateInfo alloc_info;
    alloc_info.allocationSize = mem_requirements.size;
    alloc_info.memoryTypeIndex = findMemoryType(mem_requirements.memoryTypeBits, properties);

    buffer_memory = _device->allocateMemoryUnique(alloc_info);
    if (!buffer_memory) {
        throw std::runtime_error("failed to allocate buffer memory!");
    }

    _device->bindBufferMemory(*buffer, *buffer_memory, 0);
}

void MandelbrotVulkanRenderer::copyBuffer(vk::Buffer src_buffer, vk::Buffer dst_buffer, vk::DeviceSize size) {
    auto command_buffer = createInitCommandBuffer();

    vk::BufferCopy copy_region;
    copy_region.size = size;
    command_buffer->copyBuffer(src_buffer, dst_buffer, copy_region);

    submitInitCommandBuffer(*command_buffer);
}

vk::UniqueCommandBuffer MandelbrotVulkanRenderer::createInitCommandBuffer() const
{
    vk::CommandBufferAllocateInfo alloc_info;
    alloc_info.level = vk::CommandBufferLevel::ePrimary;
    alloc_info.commandPool = *_command_pool;
    alloc_info.commandBufferCount = 1;

    std::vector<vk::UniqueCommandBuffer> command_buffers = _device->allocateCommandBuffersUnique(alloc_info);

    vk::CommandBufferBeginInfo begin_info;
    begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

    command_buffers.front()->begin(begin_info);

    return std::move(command_buffers.front());
}

void MandelbrotVulkanRenderer::submitInitCommandBuffer(vk::CommandBuffer command_buffer) const
{
    command_buffer.end();

    vk::SubmitInfo submit_info;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;

    _graphics_queue.submit(submit_info, vk::Fence());
    _graphics_queue.waitIdle();
}

uint32_t MandelbrotVulkanRenderer::findMemoryType(uint32_t type_filter, vk::MemoryPropertyFlags properties) {
    vk::PhysicalDeviceMemoryProperties mem_properties = _physical_device.getMemoryProperties();

    for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) && (mem_properties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("failed to find suitable memory type!");
}

void MandelbrotVulkanRenderer::createCommandBuffers() {
    _command_buffers.resize(_swap_chain_framebuffers.size());

    vk::CommandBufferAllocateInfo alloc_info;
    alloc_info.commandPool = *_command_pool;
    alloc_info.level = vk::CommandBufferLevel::ePrimary;
    alloc_info.commandBufferCount = (uint32_t)_command_buffers.size();

    _command_buffers = _device->allocateCommandBuffersUnique(alloc_info);
    if (_command_buffers.empty()) {
        throw std::runtime_error("failed to allocate command buffers!");
    }

    for (size_t i = 0; i < _command_buffers.size(); i++) {
        vk::CommandBuffer command_buffer = *_command_buffers[i];

        vk::CommandBufferBeginInfo begin_info;
        command_buffer.begin(begin_info);

        vk::RenderPassBeginInfo render_pass_info;
        render_pass_info.renderPass = *_render_pass;
        render_pass_info.framebuffer = *_swap_chain_framebuffers[i];
        render_pass_info.renderArea.offset = { 0, 0 };
        render_pass_info.renderArea.extent = _swap_chain_extent;

        vk::ClearValue clear_color = std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 1.0f };
        render_pass_info.clearValueCount = 1;
        render_pass_info.pClearValues = &clear_color;

        command_buffer.beginRenderPass(render_pass_info, vk::SubpassContents::eInline);

        command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *_graphics_pipeline);

        command_buffer.bindVertexBuffers(0, *_vertex_buffer, vk::DeviceSize(0));

        command_buffer.bindIndexBuffer(*_index_buffer, 0, vk::IndexType::eUint16);

        command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *_pipeline_layout, 0, _descriptor_sets[i], {});

        command_buffer.drawIndexed(static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

        command_buffer.endRenderPass();

        command_buffer.end();
    }
}

void MandelbrotVulkanRenderer::createSyncObjects() {
    _image_available_semaphores.resize(MAX_FRAMES_IN_FLIGHT);
    _render_finished_semaphores.resize(MAX_FRAMES_IN_FLIGHT);
    _in_flight_fences.resize(MAX_FRAMES_IN_FLIGHT);
    _images_in_flight.resize(_swap_chain_images.size());

    vk::SemaphoreCreateInfo semaphore_info;

    vk::FenceCreateInfo fence_info;
    fence_info.flags = vk::FenceCreateFlagBits::eSignaled;

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        _image_available_semaphores[i] = _device->createSemaphoreUnique(semaphore_info);
        _render_finished_semaphores[i] = _device->createSemaphoreUnique(semaphore_info);
        _in_flight_fences[i] = _device->createFenceUnique(fence_info);

        if (!_image_available_semaphores[i] || !_render_finished_semaphores[i] || !_in_flight_fences[i]) {
            throw std::runtime_error("failed to create synchronization objects for a frame!");
        }
    }
}

void MandelbrotVulkanRenderer::updateUniformBuffer(uint32_t current_image) {
    UniformBufferObject ubo = this->ubo;
    ubo.aspect = float(_swap_chain_extent.width) / float(_swap_chain_extent.height);

    void* data = _device->mapMemory(*_uniform_buffers_memory[current_image], 0, sizeof(ubo));
    FinalAction scoped_unmap([this, current_image]() { _device->unmapMemory(*_uniform_buffers_memory[current_image]); });

    memcpy(data, &ubo, sizeof(ubo));
}

void MandelbrotVulkanRenderer::drawFrame() {
    _device->waitForFences(*_in_flight_fences[_current_frame], VK_TRUE, UINT64_MAX);

    uint32_t image_index;
    try
    {
        vk::ResultValue<uint32_t> acquire_next_image_result = _device->acquireNextImageKHR(*_swap_chain, UINT64_MAX, *_image_available_semaphores[_current_frame], vk::Fence());
        image_index = acquire_next_image_result.value;
    }
    catch (const vk::OutOfDateKHRError&)
    {
        recreateSwapChain();
        return;
    }

    updateUniformBuffer(image_index);

    if (_images_in_flight[image_index] != vk::Fence()) {
        _device->waitForFences(_images_in_flight[image_index], VK_TRUE, UINT64_MAX);
    }
    _images_in_flight[image_index] = *_in_flight_fences[_current_frame];

    vk::SubmitInfo submit_info;

    vk::Semaphore wait_semaphores[] = { *_image_available_semaphores[_current_frame] };
    vk::PipelineStageFlags wait_stages[] = { vk::PipelineStageFlagBits::eColorAttachmentOutput };
    submit_info.waitSemaphoreCount = 1;
    submit_info.pWaitSemaphores = wait_semaphores;
    submit_info.pWaitDstStageMask = wait_stages;

    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &*_command_buffers[image_index];

    vk::Semaphore signal_semaphores[] = { *_render_finished_semaphores[_current_frame] };
    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores = signal_semaphores;

    _device->resetFences(*_in_flight_fences[_current_frame]);

    _graphics_queue.submit(submit_info, *_in_flight_fences[_current_frame]);

    vk::PresentInfoKHR present_info;

    present_info.waitSemaphoreCount = 1;
    present_info.pWaitSemaphores = signal_semaphores;

    present_info.swapchainCount = 1;
    present_info.pSwapchains = &*_swap_chain;

    present_info.pImageIndices = &image_index;

    bool out_of_date = false;
    bool suboptimal = false;

    try
    {
        vk::Result result = _present_queue.presentKHR(present_info);
        suboptimal = result == vk::Result::eSuboptimalKHR;
    }
    catch (const vk::OutOfDateKHRError&)
    {
        out_of_date = true;
    }

    if (out_of_date || suboptimal || _framebuffer_resized)
    {
        _framebuffer_resized = false;
        recreateSwapChain();
    }

    _current_frame = (_current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
}

vk::UniqueShaderModule MandelbrotVulkanRenderer::createShaderModule(const std::vector<char>& code) {
    vk::ShaderModuleCreateInfo create_info;
    create_info.codeSize = code.size();
    create_info.pCode = reinterpret_cast<const uint32_t*>(code.data());

    vk::UniqueShaderModule shader_module = _device->createShaderModuleUnique(create_info);
    if (!shader_module) {
        throw std::runtime_error("failed to create shader module!");
    }

    return shader_module;
}

vk::SurfaceFormatKHR MandelbrotVulkanRenderer::chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& available_formats) {
    for (const auto& availableFormat : available_formats) {
        if (availableFormat.format == vk::Format::eB8G8R8A8Srgb && availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
            return availableFormat;
        }
    }

    return available_formats[0];
}

vk::PresentModeKHR MandelbrotVulkanRenderer::chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& available_present_modes) {
    for (const auto& available_present_mode : available_present_modes) {
        if (available_present_mode == vk::PresentModeKHR::eMailbox) {
            return available_present_mode;
        }
    }

    return vk::PresentModeKHR::eFifo;
}

vk::Extent2D MandelbrotVulkanRenderer::chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities) {
    if (capabilities.currentExtent.width != UINT32_MAX) {
        return capabilities.currentExtent;
    }
    else {
        int width, height;
        glfwGetFramebufferSize(_window, &width, &height);

        vk::Extent2D actual_extent(width, height);

        actual_extent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actual_extent.width));
        actual_extent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actual_extent.height));

        return actual_extent;
    }
}

SwapChainSupportDetails MandelbrotVulkanRenderer::querySwapChainSupport(vk::PhysicalDevice device) {
    SwapChainSupportDetails details;

    details.capabilities = device.getSurfaceCapabilitiesKHR(*_surface);
    details.formats = device.getSurfaceFormatsKHR(*_surface);
    details.present_modes = device.getSurfacePresentModesKHR(*_surface);

    return details;
}

bool MandelbrotVulkanRenderer::isDeviceSuitable(vk::PhysicalDevice device) {
    QueueFamilyIndices indices = findQueueFamilies(device);

    bool extensionsSupported = checkDeviceExtensionSupport(device);

    vk::PhysicalDeviceFeatures supported_features = device.getFeatures();

    bool swapChainAdequate = false;
    if (extensionsSupported) {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
        swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.present_modes.empty();
    }

    return indices.isComplete() && extensionsSupported && swapChainAdequate && supported_features.samplerAnisotropy;
}

bool MandelbrotVulkanRenderer::checkDeviceExtensionSupport(vk::PhysicalDevice device) {
    std::vector<vk::ExtensionProperties> available_extensions = device.enumerateDeviceExtensionProperties();

    std::set<std::string> required_extensions(device_extensions.begin(), device_extensions.end());

    for (const auto& extension : available_extensions) {
        required_extensions.erase(extension.extensionName);
    }

    return required_extensions.empty();
}

QueueFamilyIndices MandelbrotVulkanRenderer::findQueueFamilies(vk::PhysicalDevice device) {
    QueueFamilyIndices indices;

    std::vector<vk::QueueFamilyProperties> queue_families = device.getQueueFamilyProperties();

    int i = 0;
    for (const auto& queue_family : queue_families) {
        if (queue_family.queueFlags & vk::QueueFlagBits::eGraphics) {
            indices.graphics_family = i;
        }

        vk::Bool32 present_support = device.getSurfaceSupportKHR(i, *_surface);

        if (present_support) {
            indices.present_family = i;
        }

        if (indices.isComplete()) {
            break;
        }

        i++;
    }

    return indices;
}

std::vector<const char*> MandelbrotVulkanRenderer::getRequiredExtensions() {
    uint32_t glfw_extension_count = 0;
    const char** glfw_extensions;
    glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);

    std::vector<const char*> extensions(glfw_extensions, glfw_extensions + glfw_extension_count);

    if (enable_validation_layers) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    return extensions;
}

bool MandelbrotVulkanRenderer::checkValidationLayerSupport() {
    const std::vector<vk::LayerProperties> available_layers = vk::enumerateInstanceLayerProperties();

    for (const char* layer_name : validation_layers) {
        auto compare_layer_name = [layer_name](const vk::LayerProperties& layerProperties) {
            return strcmp(layer_name, layerProperties.layerName) == 0;
        };

        if (std::none_of(available_layers.begin(), available_layers.end(), compare_layer_name))
            return false;
    }

    return true;
}

std::vector<char> MandelbrotVulkanRenderer::readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("failed to open file!");
    }

    size_t file_size = (size_t)file.tellg();
    std::vector<char> buffer(file_size);

    file.seekg(0);
    file.read(buffer.data(), file_size);

    file.close();

    return buffer;
}

VKAPI_ATTR VkBool32 VKAPI_CALL MandelbrotVulkanRenderer::debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT message_severity, VkDebugUtilsMessageTypeFlagsEXT message_type, const VkDebugUtilsMessengerCallbackDataEXT* callback_data, void* user_data) {
    std::cerr << "validation layer: " << callback_data->pMessage << std::endl;

    return VK_FALSE;
}

std::unique_ptr<IMandelbrotRenderer> createVulkanRenderer(GLFWwindow* window, const std::vector<ImageData>& colormaps)
{
    return std::make_unique<MandelbrotVulkanRenderer>(window, colormaps);
}