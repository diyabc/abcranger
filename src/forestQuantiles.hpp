#pragma once
#include <vector>

std::vector<double> forestQuantiles(const std::vector<double>& obs, 
    const std::vector<double>& weights,
    const std::vector<double>& asked);

std::vector<std::vector<double>> forestQuantiles_b(const std::vector<double>& obs, 
    const std::vector<std::vector<double>>& weights,
    const std::vector<double>& asked);


double median(std::vector<double> v);
