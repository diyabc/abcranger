#pragma once
#include <vector>
#include <string>
#include <map>
#include "Reftable.hpp"
#include "cxxopts.hpp"

struct EstimParamResults {
    std::vector<double> plsvar;
    std::vector<std::pair<std::string,double>> plsweights;
    std::vector<std::pair<std::string,double>> variable_importance;
    std::vector<double> ntree_oob_error;
    std::vector<std::pair<double,double>> values_weights; 
    std::map<std::string,double> point_estimates;
    std::map<std::string, // Global/Local
        std::map<std::string, // Mean/Median/CI
            std::map<std::string,double>>> errors;
};

EstimParamResults EstimParam_fun(Reftable &reftable,
                                   std::vector<double> statobs,
                                   const cxxopts::ParseResult opts,
                                   bool quiet = false);