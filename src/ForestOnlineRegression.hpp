#pragma once

#include <iostream>
#include <vector>

#include "globals.h"
#include "ForestOnline.hpp"

namespace ranger {

class ForestOnlineRegression: public ForestOnline {
public:
  ForestOnlineRegression() = default;

  ForestOnlineRegression(const ForestOnlineRegression&) = delete;
  ForestOnlineRegression& operator=(const ForestOnlineRegression&) = delete;

  virtual ~ForestOnlineRegression() override = default;

  void writeOOBErrorFile();   
  void writeWeightsFile();
  void writeConfusionFile() override;
  std::vector<std::pair<double,double>> getWeights();
  // subset of oob for predictions
  std::map<size_t,size_t> oob_subset;

private:
  void initInternal(std::string status_variable_name) override;
  void growInternal() override;
  void calculateAfterGrow(size_t tree_idx, bool oob) override;
  void allocatePredictMemory() override;
  void predictInternal(size_t tree_idx) override;
  void computePredictionErrorInternal() override;
  void writeOutputInternal() override;
  void writePredictionFile() override;
  void saveToFileInternal(std::ofstream& outfile) override;
  void loadFromFileInternal(std::ifstream& infile) override;

  // OOb counts for regression
  std::vector<size_t> samples_oob_count;
//  std::vector<size_t> samples_terminalnodes;
  // Storing prediction sum by tree
  std::vector<double> prediction_sum;
  std::mutex mutex_pred_0, mutex_pred_1, mutex_pred_4, mutex_pred_5;
  std::vector<std::mutex> mutex_samples;

private:
  double getTreePrediction(size_t tree_idx, size_t sample_idx) const;
  size_t getTreePredictionTerminalNodeID(size_t tree_idx, size_t sample_idx) const;
  const std::vector<size_t>& getInbagCounts(size_t tree_idx);
};

} // namespace ranger
