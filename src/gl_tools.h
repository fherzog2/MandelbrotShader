// SPDX-License-Identifier: GPL-2.0-only
#pragma once

namespace gl {

    /**
    * Generic wrapper for OpenGL resources that need cleanup.
    * The interface is similar to std::unique_ptr, but with GL handles instead of pointers as internal type.
    */
    template<typename T>
    class Object
    {
    public:
        Object() : _obj(nullValue())
        {
        }

        explicit Object(typename T::value_type obj)
            : _obj(obj)
        {}

        ~Object()
        {
            reset();
        }

        Object(const Object& other) = delete;
        Object& operator=(const Object& other) = delete;

        Object(Object&& other)
        {
            std::swap(_obj, other._obj);
        }

        Object& operator=(Object&& other)
        {
            std::swap(_obj, other._obj);
            return *this;
        }

        operator typename T::value_type() const { return _obj; }

        void reset(typename T::value_type obj = nullValue())
        {
            if (_obj != nullValue())
                T::destroy(_obj);
            _obj = obj;
        }

    private:
        static constexpr typename T::value_type nullValue() { return 0; }

        typename T::value_type _obj;
    };

    template<typename T>
    typename Object<T> makeObject()
    {
        return Object<T>(T::create());
    }

    //=============================================================================

    struct BufferTrait
    {
        typedef GLuint value_type;
        static value_type create()
        {
            value_type obj;
            glGenBuffers(1, &obj);
            return obj;
        }
        static void destroy(value_type obj)
        {
            glDeleteBuffers(1, &obj);
        }
    };

    struct VertexArrayTrait
    {
        typedef GLuint value_type;
        static value_type create()
        {
            value_type obj;
            glGenVertexArrays(1, &obj);
            return obj;
        }
        static void destroy(value_type obj)
        {
            glDeleteVertexArrays(1, &obj);
        }
    };

    struct VertexShaderTrait
    {
        typedef GLuint value_type;
        static value_type create()
        {
            return glCreateShader(GL_VERTEX_SHADER);
        }
        static void destroy(value_type obj)
        {
            glDeleteShader(obj);
        }
    };

    struct FragmentShaderTrait
    {
        typedef GLuint value_type;
        static value_type create()
        {
            return glCreateShader(GL_FRAGMENT_SHADER);
        }
        static void destroy(value_type obj)
        {
            glDeleteShader(obj);
        }
    };

    struct ProgramTrait
    {
        typedef GLuint value_type;
        static value_type create()
        {
            return glCreateProgram();
        }
        static void destroy(value_type obj)
        {
            glDeleteProgram(obj);
        }
    };

    struct TextureTrait
    {
        typedef GLuint value_type;
        static value_type create()
        {
            GLuint obj;
            glGenTextures(1, &obj);
            return obj;
        }
        static void destroy(value_type obj)
        {
            glDeleteTextures(1, &obj);
        }
    };

    using Buffer = Object<BufferTrait>;
    using VertexArray = Object<VertexArrayTrait>;
    using VertexShader = Object<VertexShaderTrait>;
    using FragmentShader = Object<FragmentShaderTrait>;
    using Program = Object<ProgramTrait>;
    using Texture = Object<TextureTrait>;

}