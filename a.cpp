#include "a.hpp"
thread_local std::mt19937 Test::mt{ std::random_device{}() };
thread_local size_t Test::a;
std::mutex Test::m;
