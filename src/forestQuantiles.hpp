#pragma once
#include <vector>

std::vector<double> forestQuantiles(const std::vector<double>& obs, 
    const std::vector<double>& weights,
    const std::vector<double>& asked);
