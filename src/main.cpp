// SPDX-License-Identifier: GPL-2.0-only
#define NOMINMAX

#include "IMandelbrotRenderer.h"
#include "util.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <filesystem>
#include <iostream>
#include <array>
#include <string>

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

enum class Api
{
    VULKAN,
    OPENGL
};

class MandelbrotShaderApp
{
public:
    MandelbrotShaderApp();

    void run();

private:
    void initWindow(Api api);
    void mainLoop();

    void switchToNextRenderer();

    void updateWindowTitle();

    static ImageData loadColorMap(const std::filesystem::path& filename);
    void loadColormaps();

    static MandelbrotShaderApp* getAppFromWindow(GLFWwindow* window);
    static void framebufferResizeCallback(GLFWwindow* window, int width, int height);
    static void cursor_enter_callback(GLFWwindow* window, int entered);
    static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
    static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);
    static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
    static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);

    std::vector<ImageData> _colormaps;

    GLFWwindow* window;
    FinalAction _window_deleter;
    std::unique_ptr<IMandelbrotRenderer> _renderer;
    Api _api;

    bool cursor_entered = false;
    glm::dvec2 cursor_pos;

    bool left_mouse_button_down = false;
    glm::dvec2 click_cursor_pos;
    glm::dvec2 click_object_center;

    int _current_fps = 0;
};

MandelbrotShaderApp::MandelbrotShaderApp()
{}

void MandelbrotShaderApp::run()
{
    loadColormaps();

    initWindow(Api::VULKAN);
    glfwShowWindow(window);
    mainLoop();
}

void MandelbrotShaderApp::initWindow(Api api)
{
    if (_renderer)
    {
        // cleanup previous window

        _renderer->waitIdle();
        _renderer.reset();
        _window_deleter = FinalAction();

    }

    _api = api;

    glfwWindowHint(GLFW_CLIENT_API, _api == Api::VULKAN ? GLFW_NO_API : GLFW_OPENGL_API);

#ifdef NDEBUG
#else
    if (_api == Api::OPENGL)
        glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);
#endif

    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

    window = glfwCreateWindow(WIDTH, HEIGHT, "", nullptr, nullptr);
    _window_deleter = FinalAction([this]() { if (window) glfwDestroyWindow(window); window = nullptr; });

    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    glfwSetCursorEnterCallback(window, cursor_enter_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetKeyCallback(window, key_callback);

    switch (_api)
    {
    case Api::VULKAN:
        _renderer = createVulkanRenderer(window, _colormaps);
        break;
    case Api::OPENGL:
        _renderer = createOpenGLRenderer(window, _colormaps);
        break;
    }

    _renderer->setCenter(glm::dvec2(0.5, 0));
    _renderer->setScale(3);
    _renderer->setIterations(200);

    updateWindowTitle();
}

void MandelbrotShaderApp::mainLoop()
{
    auto last_title_update = std::chrono::steady_clock::now();
    int frame_counter = 0;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        _renderer->drawFrame();

        ++frame_counter;

        const auto now = std::chrono::steady_clock::now();
        const auto elapsed_millis = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_title_update).count();

        if (elapsed_millis > 500)
        {
            _current_fps = frame_counter * 1000 / static_cast<int>(elapsed_millis);
            frame_counter = 0;
            last_title_update = now;

            updateWindowTitle();
        }
    }

    _renderer->waitIdle();
}

void MandelbrotShaderApp::switchToNextRenderer()
{
    std::array<Api, 2> types{ Api::VULKAN, Api::OPENGL };

    auto current_renderer = std::find(types.begin(), types.end(), _api);
    auto next_renderer = current_renderer + 1;
    if (next_renderer == types.end())
        next_renderer = types.begin();

    // save settings

    int x, y;
    glfwGetWindowPos(window, &x, &y);

    int w, h;
    glfwGetWindowSize(window, &w, &h);

    int maximized = glfwGetWindowAttrib(window, GLFW_MAXIMIZED);

    auto center = _renderer->getCenter();
    auto scale = _renderer->getScale();
    auto iterations = _renderer->getIterations();
    auto colormap_index = _renderer->getColormapIndex();

    // create new renderer

    initWindow(*next_renderer);

    // restore settings

    glfwSetWindowPos(window, x, y);
    glfwSetWindowSize(window, w, h);

    if (maximized)
        glfwMaximizeWindow(window);

    _renderer->setCenter(center);
    _renderer->setScale(scale);
    _renderer->setIterations(iterations);
    _renderer->setColormapIndex(colormap_index);

    glfwShowWindow(window);
}

void MandelbrotShaderApp::updateWindowTitle()
{
    std::string title;
    title.reserve(200);

    title = "Mandelbrot";

    switch (_api)
    {
    case Api::VULKAN:
        title += " (Vulkan)";
        break;
    case Api::OPENGL:
        title += " (OpenGL)";
        break;
    }

    title += ", ";
    title += std::to_string(_renderer->getIterations());
    title += " iterations";

    title += ", ";
    title += std::to_string(_current_fps);
    title += " FPS";

    glfwSetWindowTitle(window, title.data());
}

ImageData MandelbrotShaderApp::loadColorMap(const std::filesystem::path& filename)
{
#ifdef _MSC_VER
    FILE* file = _wfopen(filename.c_str(), L"rb");
#else
    FILE* file = fopen(filename.c_str(), "rb");
#endif
    FinalAction file_close([file]() { fclose(file); });

    int w;
    int h;
    int channels;
    stbi_uc* pixels = stbi_load_from_file(file, &w, &h, &channels, STBI_rgb_alpha);
    FinalAction pixels_free([pixels]() { stbi_image_free(pixels); });

    ImageData result;

    if (pixels)
    {
        result.width = w;
        result.height = h;

        result.bytes.resize(w * 4);
        memcpy(result.bytes.data(), pixels, result.bytes.size());
    }

    return result;
}

void MandelbrotShaderApp::loadColormaps()
{
    for (const auto& p : std::filesystem::directory_iterator("colormaps"))
    {
        ImageData colormap = loadColorMap(p);

        if (!colormap)
            throw std::runtime_error("cannot load colormap file");

        _colormaps.push_back(std::move(colormap));
    }

    if (_colormaps.empty())
        throw std::runtime_error("no colormaps loaded");

    auto first = _colormaps.begin();

    auto equal_size = [first](const ImageData& colormap) {
        return colormap.width == first->width && colormap.height == first->height;
    };

    if (!std::all_of(first, _colormaps.end(), equal_size))
        throw std::runtime_error("all colormap must have the same size");
}

MandelbrotShaderApp* MandelbrotShaderApp::getAppFromWindow(GLFWwindow* window)
{
    return reinterpret_cast<MandelbrotShaderApp*>(glfwGetWindowUserPointer(window));
}

void MandelbrotShaderApp::framebufferResizeCallback(GLFWwindow* window, int width, int height) {
    auto app = getAppFromWindow(window);
    app->_renderer->notifyFramebufferResized();
}

void MandelbrotShaderApp::cursor_enter_callback(GLFWwindow* window, int entered) {
    auto app = getAppFromWindow(window);

    app->cursor_entered = bool(entered);
}

void MandelbrotShaderApp::mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    auto app = getAppFromWindow(window);

    if (app->cursor_entered)
    {
        if (button == GLFW_MOUSE_BUTTON_LEFT)
        {
            if (action == GLFW_PRESS)
            {
                app->left_mouse_button_down = true;
                app->click_cursor_pos = app->cursor_pos;
                app->click_object_center = app->_renderer->getCenter();
            }
            else if (action == GLFW_RELEASE)
            {
                app->left_mouse_button_down = false;
            }
        }
    }
}

void MandelbrotShaderApp::cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    auto app = getAppFromWindow(window);

    if (app->cursor_entered)
    {
        app->cursor_pos.x = xpos;
        app->cursor_pos.y = ypos;

        if (app->left_mouse_button_down)
        {
            int width;
            int height;
            glfwGetWindowSize(window, &width, &height);
            const glm::dvec2 window_size{ width, height };

            const glm::float64 aspect = window_size.x / window_size.y;

            const glm::dvec2 obj_size{ app->_renderer->getScale() * aspect, app->_renderer->getScale() };

            const glm::dvec2 window_drag_offset = app->cursor_pos - app->click_cursor_pos;

            glm::dvec2 obj_drag_offset = window_drag_offset;
            obj_drag_offset.x = obj_drag_offset.x / window_size.x * obj_size.x;
            obj_drag_offset.y = obj_drag_offset.y / window_size.y * obj_size.y;

            app->_renderer->setCenter(app->click_object_center + obj_drag_offset);
        }
    }
}

void MandelbrotShaderApp::scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    auto app = getAppFromWindow(window);

    if (app->cursor_entered && !app->left_mouse_button_down)
    {
        const glm::float64 zoom_factor = 1.2;

        // calculate the cursor position in object space

        int width;
        int height;
        glfwGetWindowSize(window, &width, &height);
        const glm::dvec2 window_size{ width, height };

        const glm::float64 aspect = window_size.x / window_size.y;

        glm::float64 obj_width = app->_renderer->getScale() * aspect;
        glm::float64 obj_height = app->_renderer->getScale();

        glm::float64 obj_left = app->_renderer->getCenter().x - obj_width * 0.5;
        glm::float64 obj_top = app->_renderer->getCenter().y - obj_height * 0.5;

        const glm::float64 cursor_object_x = (app->cursor_pos.x / window_size.x) * obj_width + obj_left;
        const glm::float64 cursor_object_y = (app->cursor_pos.y / window_size.y) * obj_height + obj_top;

        // apply zoom

        if (yoffset < 0)
        {
            app->_renderer->setScale(app->_renderer->getScale() * zoom_factor);
        }
        else
        {
            app->_renderer->setScale(app->_renderer->getScale() / zoom_factor);
        }

        // calculate the cursor position in object space again

        obj_width = app->_renderer->getScale() * aspect;
        obj_height = app->_renderer->getScale();

        obj_left = app->_renderer->getCenter().x - obj_width * 0.5;
        obj_top = app->_renderer->getCenter().y - obj_height * 0.5;

        const glm::float64 cursor_object_x2 = (app->cursor_pos.x / window_size.x) * obj_width + obj_left;
        const glm::float64 cursor_object_y2 = (app->cursor_pos.y / window_size.y) * obj_height + obj_top;

        // move the center so the object pos under the cursor stays constant

        app->_renderer->setCenter(app->_renderer->getCenter() + glm::dvec2(cursor_object_x2 - cursor_object_x, cursor_object_y2 - cursor_object_y));
    }
}

void MandelbrotShaderApp::key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    auto app = getAppFromWindow(window);

    if (action == GLFW_PRESS || action == GLFW_REPEAT)
    {
        if (key == GLFW_KEY_PAGE_DOWN)
        {
            app->_renderer->setIterations(std::max(1, app->_renderer->getIterations() - 1));
            app->updateWindowTitle();
        }
        else if (key == GLFW_KEY_PAGE_UP)
        {
            app->_renderer->setIterations(app->_renderer->getIterations() + 1);
            app->updateWindowTitle();
        }
    }
    if (action == GLFW_PRESS)
    {
        if (key == GLFW_KEY_F5)
        {
            app->switchToNextRenderer();
        }
        else if (key == GLFW_KEY_F7)
        {
            // previous colormap

            int new_index = app->_renderer->getColormapIndex() - 1;

            if (new_index < 0)
            {
                new_index = static_cast<int>(app->_colormaps.size()) - 1;
            }

            app->_renderer->setColormapIndex(new_index);
        }
        else if (key == GLFW_KEY_F8)
        {
            // next colormap

            int new_index = app->_renderer->getColormapIndex() + 1;

            if (new_index >= static_cast<int>(app->_colormaps.size()))
            {
                new_index = 0;
            }

            app->_renderer->setColormapIndex(new_index);
        }
    }
}

int main() {

    try {
        glfwInit();
        auto glfw_terminate = FinalAction([]() { glfwTerminate(); });

        MandelbrotShaderApp app;
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}