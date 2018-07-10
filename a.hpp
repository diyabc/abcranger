#pragma once
#include <mutex>
#include <random>

class Test{
public:
	static thread_local std::mt19937 mt;
	static thread_local size_t a;
	static std::mutex m;
};