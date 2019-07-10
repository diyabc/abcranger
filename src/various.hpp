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

using std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::chrono::seconds;
using std::chrono::microseconds;

static steady_clock::time_point last_time;;

static std::deque<microseconds> intervals;
static size_t etamaxwidth;

static inline void initbar() {
    etamaxwidth = 0;
    intervals = std::deque<microseconds>();
    last_time = steady_clock::now();
}


static inline void loadbar(size_t x, size_t n, size_t w = 50)
{
    if ((x != n) && (x % (n / 100 + 1) != 0))
        return;
    size_t width = (std::to_string(n)).length();
    steady_clock::time_point temp_time = steady_clock::now();
    microseconds elapsed_time = duration_cast<microseconds>(temp_time - last_time);
    if (intervals.size() == 10) {
        intervals.pop_front();
    }
    intervals.push_back(elapsed_time);

    float ratio = x / (float)n;
    int c = ratio * w;

    size_t total = 0;
    for(auto& e : intervals) total += e.count();
    float mean = static_cast<float>(total)/static_cast<float>(intervals.size());
    
    size_t remaining_time = std::floor(mean * static_cast<float>(n-x)/1e6);
    std::string eta = ranger::beautifyTime(remaining_time);
    size_t etalength = eta.length();
    if ((intervals.size() == 10) && (etalength > etamaxwidth)) etamaxwidth = etalength;

    std::string itsec = std::to_string(1e6/mean);
    std::cout << " [";
    for (auto x = 0; x < c; x++)
        std::cout << "=";
    for (auto x = c; x < w; x++)
        std::cout << " ";
    std::cout << "] (" << std::setw(width) << x << "/" << std::setw(width) << n << ") ";
    std::cout << " " << std::setw(5) << std::fixed << std::setprecision(1) << 1e6/mean << " it/sec";
    if (intervals.size() == 10) std::cout << ", ETA : " << std::setw(etamaxwidth) << eta;
    std::cout << "\r" << std::flush;
    last_time = temp_time;
}
