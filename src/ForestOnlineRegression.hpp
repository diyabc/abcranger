#pragma once

#include <iostream>
#include <vector>
#include <map>

#include "globals.h"
#include "ForestOnline.hpp"

namespace ranger {

class ForestOnlineRegression: public ForestOnline {
public:
  ForestOnlineRegression() = default;

  ForestOnlineRegression(const ForestOnlineRegression&) = delete;
  ForestOnlineRegression& operator=(const ForestOnlineRegression&) = delete;

  virtual ~ForestOnlineRegression() override = default;

  void writeConfusionFile() override;
  void writeOOBErrorFile();   
  std::vector<std::pair<double,std::vector<double>>> getWeights();
  void writeWeightsFile();


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

//  std::vector<size_t> samples_terminalnodes;
  // Storing prediction sum by tree
  std::vector<double> prediction_sum;
  std::mutex mutex_pred_0, mutex_pred_1, mutex_pred_4, mutex_pred_5;

private:
  double getTreePrediction(size_t tree_idx, size_t sample_idx) const;
  size_t getTreePredictionTerminalNodeID(size_t tree_idx, size_t sample_idx) const;
  const std::vector<size_t>& getInbagCounts(size_t tree_idx);
};

} // namespace ranger
