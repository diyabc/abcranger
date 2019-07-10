#pragma once
#include <vector>
#include <memory>
#include <iostream>
#include <iomanip>

constexpr std::size_t operator"" _z(unsigned long long n)
{
    return n;
}

auto sz = 5_z;
static_assert(std::is_same<decltype(sz), std::size_t>::value, "");

std::vector<double> DEFAULT_SAMPLE_FRACTION = std::vector<double>({1});

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

static inline void loadbar(size_t x, size_t n, size_t w = 50)
{
    if ((x != n) && (x % (n / 100 + 1) != 0))
        return;
    float ratio = x / (float)n;
    int c = ratio * w;
    std::cout << " [";
    for (auto x = 0; x < c; x++)
        std::cout << "=";
    for (auto x = c; x < w; x++)
        std::cout << " ";
    std::cout << "] " << std::setw(5) << x << "/" << std::setw(5) << n << "\r" << std::flush;
}
