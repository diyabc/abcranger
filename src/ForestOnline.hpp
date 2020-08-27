#pragma once

#include <vector>
#include <iostream>
#include <random>
#include <ctime>
#include <memory>
#ifndef OLD_WIN_R_BUILD
#include <thread>
#include <chrono>
#include <mutex>
#include <condition_variable>
#endif

#include "globals.h"
#include "Tree.h"
#include "Data.h"
#include "tqdm.hpp"

namespace ranger {

class ForestOnline {
public:
  ForestOnline();

  ForestOnline(const ForestOnline&) = delete;
  ForestOnline& operator=(const ForestOnline&) = delete;

  virtual ~ForestOnline() = default;

  // Init from c++ main or Rcpp from R
  void init(std::string dependent_variable_name, MemoryMode memory_mode, std::unique_ptr<Data> input_data, std::unique_ptr<Data> predict_data, uint mtry,
      std::string output_prefix, uint num_trees, uint seed, uint num_threads, ImportanceMode importance_mode,
      uint min_node_size, std::string status_variable_name, bool prediction_mode, bool sample_with_replacement,
      const std::vector<std::string>& unordered_variable_names, bool memory_saving_splitting, SplitRule splitrule,
      bool predict_all, std::vector<double>& sample_fraction, double alpha, double minprop, bool holdout,
      PredictionType prediction_type, uint num_random_splits, bool order_snps, uint max_depth, size_t oob_weights = 0);
  virtual void initInternal(std::string status_variable_name) = 0;

  // Grow or predict
  void run(bool verbose, bool compute_oob_error);

  // Write results to output files
  void writeOutput();
  virtual void writeOutputInternal() = 0;
  virtual void writeConfusionFile() = 0;
  virtual void writePredictionFile() = 0;
  void writeImportanceFile();
  std::vector<std::pair<std::string,double>> getImportance();

  // Save ForestOnline to file
  void saveToFile();
  virtual void saveToFileInternal(std::ofstream& outfile) = 0;

  std::unique_ptr<Data> releaseData() {
    return std::move(data);
  }
  std::unique_ptr<Data> releasePred() {
    return std::move(predict_data);
  }
  std::vector<std::vector<std::vector<size_t>>> getChildNodeIDs() {
    std::vector<std::vector<std::vector<size_t>>> result;
    for (auto& tree : trees) {
      result.push_back(tree->getChildNodeIDs());
    }
    return result;
  }
  std::vector<std::vector<size_t>> getSplitVarIDs() {
    std::vector<std::vector<size_t>> result;
    for (auto& tree : trees) {
      result.push_back(tree->getSplitVarIDs());
    }
    return result;
  }
  std::vector<std::vector<double>> getSplitValues() {
    std::vector<std::vector<double>> result;
    for (auto& tree : trees) {
      result.push_back(tree->getSplitValues());
    }
    return result;
  }
  const std::vector<double>& getVariableImportance() const {
    return variable_importance;
  }
  double getOverallPredictionError() const {
    return overall_prediction_error;
  }
  const std::vector<std::vector<std::vector<double>>>& getPredictions() const {
    return predictions;
  }
  size_t getDependentVarId() const {
    return dependent_varID;
  }
  size_t getNumTrees() const {
    return num_trees;
  }
  uint getMtry() const {
    return mtry;
  }
  uint getMinNodeSize() const {
    return min_node_size;
  }
  size_t getNumIndependentVariables() const {
    return num_independent_variables;
  }

  const std::vector<bool>& getIsOrderedVariable() const {
    return data->getIsOrderedVariable();
  }

  std::vector<std::vector<size_t>> getInbagCounts() const {
    std::vector<std::vector<size_t>> result;
    for (auto& tree : trees) {
      result.push_back(tree->getInbagCounts());
    }
    return result;
  }

  const std::vector<std::vector<size_t>>& getSnpOrder() const {
    return data->getSnpOrder();
  }
  // Verbose output stream, cout if verbose==true, logfile if not
  std::ostream* verbose_out;

protected:
  void grow();
  virtual void growInternal() = 0;
  virtual void calculateAfterGrow(size_t tree_idx, bool oob) = 0;

  // Predict using existing tree from file and data as prediction data
  void predict();
  virtual void allocatePredictMemory() = 0;
  virtual void predictInternal(size_t tree_idx) = 0;

  void computePredictionError();
  virtual void computePredictionErrorInternal() = 0;

  void computePermutationImportance();

  // Multithreading methods for growing/prediction/importance, called by each thread
  void growTreesInThread(uint thread_idx, std::vector<double>* variable_importance, const Data* input_data, const Data* predict_data);
  void predictTreesInThread(uint thread_idx, const Data* prediction_data, bool oob_prediction);
  void predictInternalInThread(uint thread_idx);
  void computeTreePermutationImportanceInThread(uint thread_idx, std::vector<double>& importance,
      std::vector<double>& variance);

  // Load ForestOnline from file
  void loadFromFile(std::string filename);
  virtual void loadFromFileInternal(std::ifstream& infile) = 0;

  // Set split select weights and variables to be always considered for splitting
  void setSplitWeightVector(std::vector<std::vector<double>>& split_select_weights);
  void setAlwaysSplitVariables(const std::vector<std::string>& always_split_variable_names);

  // Show progress every few seconds
#ifdef OLD_WIN_R_BUILD
  void showProgress(std::string operation, clock_t start_time, clock_t& lap_time);
#else
  void showProgress(std::string operation, size_t max_progress);
#endif


  size_t num_trees;
  uint mtry;
  uint min_node_size;
  size_t num_variables;
  size_t num_independent_variables;
  uint seed;
  size_t dependent_varID;
  size_t num_samples;
  size_t num_oob_weights;
  bool prediction_mode;
  MemoryMode memory_mode;
  bool sample_with_replacement;
  bool memory_saving_splitting;
  SplitRule splitrule;
  bool predict_all;
  bool keep_inbag;
  std::vector<double> sample_fraction;
  bool holdout;
  PredictionType prediction_type;
  uint num_random_splits;
  uint max_depth;
  std::vector<size_t> tree_order;
  // MAXSTAT splitrule
  double alpha;
  double minprop;

  // Multithreading
  uint num_threads;
  std::vector<uint> thread_ranges;
#ifndef OLD_WIN_R_BUILD
  std::mutex mutex,mutex_post;
#endif

  std::vector<std::unique_ptr<Tree>> trees;
  std::unique_ptr<Data> data;
  std::unique_ptr<Data> predict_data;

  std::vector<std::vector<std::vector<double>>> predictions;
  double overall_prediction_error;

  // Weight vector for selecting possible split variables, one weight between 0 (never select) and 1 (always select) for each variable
  // Deterministic variables are always selected
  std::vector<size_t> deterministic_varIDs;
  std::vector<size_t> split_select_varIDs;
  std::vector<std::vector<double>> split_select_weights;

  // Bootstrap weights
  std::vector<double> case_weights;

  // Pre-selected bootstrap samples (per tree)
  std::vector<std::vector<size_t>> manual_inbag;

  // Random number generator
  std::mt19937_64 random_number_generator;

  std::string output_prefix;
  ImportanceMode importance_mode;

  // Variable importance for all variables in ForestOnline
  std::vector<double> variable_importance;

  // Computation progress (finished trees)
  size_t progress;
  tqdm bar;
#ifdef R_BUILD
  size_t aborted_threads;
  bool aborted;
#endif
};

} // namespace ranger
