#pragma once
#include <vector>
#include <string>
#include "Reftable.hpp"
#include "cxxopts.hpp"

struct EstimParamResults {
    std::vector<double> plsvar;
    std::vector<std::pair<std::string,double>> plsweights;
    double oob_error;
    std::vector<std::pair<std::string,double>> variable_importance;
    std::vector<double> ntree_oob_error;
    std::vector<std::pair<double,double>> values_weights; 
    double expectation;
    double variance;
    std::vector<double> quantiles;
    double MSE,NMSE,NMAE,meanCI,meanrelativeCI,medianCI,medianrelativeCI;
};

EstimParamResults EstimParam_fun(Reftable &reftable,
                                   std::vector<double> statobs,
                                   const cxxopts::ParseResult& opts,
                                   bool quiet = false);