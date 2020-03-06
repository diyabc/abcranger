#pragma once
#include <vector>
#include <memory>
#include <iostream>
#include <iomanip>
#include <queue>
#include <deque>
#include <chrono>
#include "utility.h"

static constexpr std::size_t operator"" _z(unsigned long long n)
{
    return n;
}

static auto sz = 5_z;
static_assert(std::is_same<decltype(sz), std::size_t>::value, "");

static std::vector<double> DEFAULT_SAMPLE_FRACTION = std::vector<double>({1});

template <class T_SRC, class T_DEST>
std::unique_ptr<T_DEST> unique_cast(std::unique_ptr<T_SRC> &&src)
{
    if (!src)
        return std::unique_ptr<T_DEST>();

    // Throws a std::bad_cast() if this doesn't work out
    T_DEST *dest_ptr = &dynamic_cast<T_DEST &>(*src.get());

    src.release();
    std::unique_ptr<T_DEST> ret(dest_ptr);
    return ret;
}
