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
    std::vector<std::vector<size_t>> oob_votes;
    std::vector<double> post_proba;
    std::vector<std::vector<double>> pudlo_weights;
    std::vector<std::vector<std::vector<double>>> pudlo_weights_all;
    std::vector<std::vector<double>> luhardin_weights;
    std::vector<std::vector<double>> post_proba_all;
    std::vector<std::vector<double>> post_proba_all2;
    std::vector<std::vector<double>> post_proba_all3;
    std::vector<std::vector<double>> model_errors;
    std::vector<std::vector<double>> model_errors2;
    std::vector<std::vector<double>> model_errors3;
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