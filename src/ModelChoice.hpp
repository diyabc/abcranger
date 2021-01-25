#pragma once
#include <vector>
#include <string>
#include "Reftable.hpp"
#include "cxxopts.hpp"

struct ModelChoiceResults
{
    double oob_error;
    std::vector<std::vector<size_t>> confusion_matrix;
    std::vector<std::pair<std::string, double>> variable_importance;
    std::vector<double> ntree_oob_error;
    std::vector<size_t> predicted_model;
    std::vector<std::vector<size_t>> votes;
    std::vector<double> post_proba;
};

template<class MatrixType>
ModelChoiceResults ModelChoice_fun(Reftable<MatrixType> &reftable,
                                   MatrixXd statobs,
                                   const cxxopts::ParseResult opts,
                                   bool quiet = false);

template<class MatrixType>
ModelChoiceResults ModelChoice_fun(Reftable<MatrixType> &reftable,
                                   std::vector<double> statobs,
                                   const cxxopts::ParseResult opts,
                                   bool quiet = false);