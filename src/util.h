// SPDX-License-Identifier: GPL-2.0-only
#pragma once

#include <functional>

class FinalAction
{
public:
    FinalAction()
    {}

    template<class FUNC>
    explicit FinalAction(FUNC func)
        : _func(func)
    {}

    FinalAction(const FinalAction& other) = delete;
    FinalAction& operator=(const FinalAction& other) = delete;

    FinalAction(FinalAction&& other)
    {
        std::swap(_func, other._func);
    }

    FinalAction& operator=(FinalAction&& other)
    {
        std::swap(_func, other._func);
        return *this;
    }

    ~FinalAction()
    {
        if (_func)
            _func();
    }

private:
    std::function<void()> _func;
};