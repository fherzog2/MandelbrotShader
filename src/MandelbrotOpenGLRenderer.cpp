// SPDX-License-Identifier: GPL-2.0-only
#include "IMandelbrotRenderer.h"
#include "util.h"
#include "gl_tools.h"

#include <iostream>
#include <fstream>

class MandelbrotOpenGLRenderer : public IMandelbrotRenderer
{
public:
    MandelbrotOpenGLRenderer(GLFWwindow* window, const std::vector<ImageData>& colormaps);

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
    static void APIENTRY debugCallback(GLenum source,
        GLenum type,
        GLuint id,
        GLenum severity,
        GLsizei length,
        const GLchar* message,
        const void* userParam);

    void createVertexBuffer();
    void createColormapArray(const std::vector<ImageData>& colormaps);
    void createShader();

    static std::string loadTextFile(const char* filename);
    static void compileShader(GLuint shader, const char* source);
    static void linkProgram(GLuint program, const std::vector<GLuint>& shaders);

    GLFWwindow* _window;

    glm::dvec2 _center;
    glm::float64 _scale;
    glm::int32 _iterations;
    int _colormap_index;

    gl::Buffer _vertex_buffer;
    gl::Buffer _index_buffer;
    gl::VertexArray _vertex_array;

    gl::Program _program;
    gl::Buffer _uniform_buffer;

    gl::Texture _colormap_array;

    struct Vertex
    {
        glm::vec2 pos;
        glm::vec2 texcoord;
    };

    struct ShaderData {
        alignas(8) glm::dvec2 center;
        alignas(8) glm::float64 scale;
        alignas(4) glm::int32 iterations;
        alignas(8) glm::float64 aspect;
        alignas(4) glm::float32 colormap_index;
    };
};

MandelbrotOpenGLRenderer::MandelbrotOpenGLRenderer(GLFWwindow* window, const std::vector<ImageData>& colormaps)
    : _window(window)
{
    glfwMakeContextCurrent(window);

    glewInit();

    glDebugMessageCallback(debugCallback, nullptr);

    createVertexBuffer();
    createColormapArray(colormaps);
    createShader();
}

void MandelbrotOpenGLRenderer::setCenter(glm::dvec2 center)
{
    _center = center;
}

glm::dvec2 MandelbrotOpenGLRenderer::getCenter() const
{
    return _center;
}

void MandelbrotOpenGLRenderer::setScale(glm::float64 scale)
{
    _scale = scale;
}

glm::float64 MandelbrotOpenGLRenderer::getScale() const
{
    return _scale;
}

void MandelbrotOpenGLRenderer::setIterations(glm::int32 iterations)
{
    _iterations = iterations;
}

glm::int32 MandelbrotOpenGLRenderer::getIterations() const
{
    return _iterations;
}

void MandelbrotOpenGLRenderer::setColormapIndex(int colormap_index)
{
    _colormap_index = colormap_index;
}

int MandelbrotOpenGLRenderer::getColormapIndex() const
{
    return _colormap_index;
}

void MandelbrotOpenGLRenderer::notifyFramebufferResized()
{
    int w, h;
    glfwGetWindowSize(_window, &w, &h);

    glViewport(0, 0, w, h);
}

void MandelbrotOpenGLRenderer::drawFrame()
{
    int w, h;
    glfwGetWindowSize(_window, &w, &h);

    ShaderData shader_data;
    shader_data.center = _center;
    shader_data.scale = _scale;
    shader_data.iterations = _iterations;
    shader_data.aspect = double(w) / double(h);
    shader_data.colormap_index = static_cast<glm::float32>(_colormap_index);

    glBindBuffer(GL_UNIFORM_BUFFER, _uniform_buffer);
    GLvoid* p = glMapBuffer(GL_UNIFORM_BUFFER, GL_WRITE_ONLY);
    memcpy(p, &shader_data, sizeof(shader_data));
    glUnmapBuffer(GL_UNIFORM_BUFFER);

    glClear(GL_COLOR_BUFFER_BIT);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _index_buffer);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, nullptr);

    glfwSwapBuffers(_window);
}

void MandelbrotOpenGLRenderer::waitIdle() const
{
}

void APIENTRY MandelbrotOpenGLRenderer::debugCallback(GLenum source, GLenum type, GLuint id, GLenum severity,
    GLsizei length, const GLchar* message, const void* userParam)
{
    if (severity == GL_DEBUG_SEVERITY_NOTIFICATION)
        return;

    auto format_source = [](GLenum source) {
        switch (source)
        {
        case GL_DEBUG_SOURCE_API: return "API";
        case GL_DEBUG_SOURCE_WINDOW_SYSTEM: return "WINDOW SYSTEM";
        case GL_DEBUG_SOURCE_SHADER_COMPILER: return "SHADER COMPILER";
        case GL_DEBUG_SOURCE_THIRD_PARTY: return "THIRD PARTY";
        case GL_DEBUG_SOURCE_APPLICATION: return "APPLICATION";
        case GL_DEBUG_SOURCE_OTHER: return "OTHER";
        case GL_DONT_CARE: return "DONT CARE";
        }

        return "?";
    };

    auto format_type = [](GLenum type) {
        switch (type)
        {
        case GL_DEBUG_TYPE_ERROR: return "ERROR";
        case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: return "DEPRECATED BEHAVIOUR";
        case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR: return "UNDEFINED BEHAVIOUR";
        case GL_DEBUG_TYPE_PORTABILITY: return "PORTABILITY";
        case GL_DEBUG_TYPE_PERFORMANCE: return "PERFORMANCE";
        case GL_DEBUG_TYPE_OTHER: return "OTHER";
        case GL_DEBUG_TYPE_MARKER: return "MARKER";
        case GL_DEBUG_TYPE_PUSH_GROUP: return "PUSH GROUP";
        case GL_DEBUG_TYPE_POP_GROUP: return "POP GROUP";
        case GL_DONT_CARE: return "DONT CARE";
        }

        return "?";
    };

    auto format_severity = [](GLenum severity) {
        switch (severity)
        {
        case GL_DEBUG_SEVERITY_LOW: return "LOW";
        case GL_DEBUG_SEVERITY_MEDIUM: return "MEDIUM";
        case GL_DEBUG_SEVERITY_HIGH: return "HIGH";
        case GL_DEBUG_SEVERITY_NOTIFICATION: return "NOTIFICATION";
        case GL_DONT_CARE: return "DONT CARE";
        }

        return "?";
    };

    std::cerr
        << "source " << format_source(source)
        << " type " << format_type(type)
        << " id " << id
        << " severity " << format_severity(severity)
        << ": " << std::string(message, message + length)
        << std::endl;
}

void MandelbrotOpenGLRenderer::createVertexBuffer()
{
    const std::vector<Vertex> vertices = {
        {{-1, -1}, {0, 0}},
        {{ 1, -1}, {1, 0}},
        {{ 1,  1}, {1, 1}},
        {{-1,  1}, {0, 1}}
    };

    const std::vector<uint16_t> indices = {
        1, 0, 3, 1, 3, 2
    };

    _vertex_buffer = gl::makeObject<gl::BufferTrait>();
    glBindBuffer(GL_ARRAY_BUFFER, _vertex_buffer);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), vertices.data(), GL_STATIC_DRAW);

    _index_buffer = gl::makeObject<gl::BufferTrait>();
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _index_buffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(uint16_t), indices.data(), GL_STATIC_DRAW);

    _vertex_array = gl::makeObject<gl::VertexArrayTrait>();
    glBindVertexArray(_vertex_array);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), 0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)sizeof(glm::vec2));
}

void MandelbrotOpenGLRenderer::createColormapArray(const std::vector<ImageData>& colormaps)
{
    int w = colormaps.front().width;
    int h = colormaps.front().height;
    GLsizei layer_count = static_cast<GLsizei>(colormaps.size());

    size_t total_texel_count = 0;
    for (const auto& layer : colormaps)
        total_texel_count += layer.bytes.size();

    std::vector<uint8_t> texels;
    texels.reserve(total_texel_count);

    for (const auto& layer : colormaps)
    {
        texels.insert(texels.end(), layer.bytes.begin(), layer.bytes.end());
    }

    _colormap_array = gl::makeObject<gl::TextureTrait>();
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D_ARRAY, _colormap_array);
    glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, GL_RGBA8, w, h, layer_count);
    glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0, w, h, layer_count, GL_RGBA, GL_UNSIGNED_BYTE, texels.data());
}

void MandelbrotOpenGLRenderer::createShader()
{
    // compile shader

    const auto vs_source = loadTextFile("shaders/gl_shader.vert");
    const auto fs_source = loadTextFile("shaders/gl_shader.frag");

    const auto vertex_shader = gl::makeObject<gl::VertexShaderTrait>();
    compileShader(vertex_shader, vs_source.data());

    const auto fragment_shader = gl::makeObject<gl::FragmentShaderTrait>();
    compileShader(fragment_shader, fs_source.data());

    _program = gl::makeObject<gl::ProgramTrait>();
    linkProgram(_program, { vertex_shader, fragment_shader });
    glUseProgram(_program);

    // setup uniform buffer

    const GLuint ubo_index = glGetUniformBlockIndex(_program, "shader_data");

    _uniform_buffer = gl::makeObject<gl::BufferTrait>();
    glBindBuffer(GL_UNIFORM_BUFFER, _uniform_buffer);

    ShaderData shader_data;
    shader_data.center = _center;
    shader_data.scale = _scale;
    shader_data.iterations = _iterations;
    shader_data.aspect = 1;
    shader_data.colormap_index = static_cast<glm::float32>(_colormap_index);
    glBufferData(GL_UNIFORM_BUFFER, sizeof(shader_data), &shader_data, GL_DYNAMIC_DRAW);

    glBindBufferBase(GL_UNIFORM_BUFFER, ubo_index, _uniform_buffer);

    // setup texture sampler

    GLuint colormap_array_sampler = glGetUniformLocation(_program, "colormap_array");
    glUniform1i(colormap_array_sampler, 0);
}

std::string MandelbrotOpenGLRenderer::loadTextFile(const char* filename)
{
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open())
        throw std::runtime_error("failed to open file!");

    const size_t fileSize = (size_t)file.tellg();

    std::string buffer;
    buffer.resize(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);

    file.close();

    return buffer;
}

void MandelbrotOpenGLRenderer::compileShader(GLuint shader, const char* source)
{
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status == GL_FALSE)
    {
        GLint info_log_length;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &info_log_length);

        std::string info_log;
        info_log.resize(info_log_length);
        glGetShaderInfoLog(shader, info_log_length, 0, &info_log[0]);

        throw std::runtime_error("shader compilation failed:\n" + info_log);
    }
}

void MandelbrotOpenGLRenderer::linkProgram(GLuint program, const std::vector<GLuint>& shaders)
{
    for (GLuint shader : shaders)
        glAttachShader(program, shader);

    glLinkProgram(program);

    GLint status;
    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if (status == GL_FALSE)
    {
        GLint info_log_length;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &info_log_length);

        std::string info_log;
        info_log.resize(info_log_length);
        glGetProgramInfoLog(program, info_log_length, NULL, &info_log[0]);

        throw std::runtime_error("shader linking failed:\n" + info_log);
    }
}

std::unique_ptr<IMandelbrotRenderer> createOpenGLRenderer(GLFWwindow* window, const std::vector<ImageData>& colormaps)
{
    return std::make_unique<MandelbrotOpenGLRenderer>(window, colormaps);
}