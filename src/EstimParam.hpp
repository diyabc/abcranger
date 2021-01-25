#pragma once
#include <vector>
#include <string>
#include <map>
#include "Reftable.hpp"
#include "cxxopts.hpp"

struct EstimParamResults
{
    std::vector<double> plsvar;
    std::vector<std::pair<std::string, double>> plsweights;
    std::vector<std::pair<std::string, double>> variable_importance;
    std::vector<double> ntree_oob_error;
    std::vector<std::pair<double, double>> values_weights;
    std::map<size_t, size_t> oob_map;
    Eigen::MatrixXd oob_weights;
    std::vector<std::map<std::string, double>> point_estimates;
    std::map<std::string,          // Global/Local
                         std::vector<std::map<std::string, // Mean/Median/CI
                                  std::map<std::string, double>>>>
        errors;
};

template <class MatrixType>
EstimParamResults EstimParam_fun(Reftable<MatrixType> &reftable,
                                 MatrixXd statobs,
                                 const cxxopts::ParseResult opts,
                                 bool quiet = false,
                                 bool weights = false);

template <class MatrixType>
EstimParamResults EstimParam_fun(Reftable<MatrixType> &reftable,
                                 std::vector<double> statobs,
                                 const cxxopts::ParseResult opts,
                                 bool quiet = false,
                                 bool weights = false);