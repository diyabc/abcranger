#pragma once

#include <map>
#include <unordered_map>
#include "ForestOnline.hpp"

namespace ranger
{
class ForestOnlineClassification : public ForestOnline
{
public:
  ForestOnlineClassification() = default;

  ForestOnlineClassification(const ForestOnlineClassification &) = delete;
  ForestOnlineClassification &operator=(const ForestOnlineClassification &) = delete;

  virtual ~ForestOnlineClassification() override = default;

  const std::vector<double> &getClassValues() const
  {
    return class_values;
  }

  void setClassWeights(std::vector<double> &class_weights)
  {
    this->class_weights = class_weights;
  }

  const std::vector<std::unordered_map<double,size_t>> &getOOBVotes() const
  {
    return oob_votes;
  }

  // Manually set the outputstream for verbose out
  void setverboseOutput(std::ostream* verbose_output);
  void writeConfusionFile() override;
  std::vector<std::vector<size_t>> getConfusion();
  void writeOOBErrorFile();   
  std::vector<std::pair<double,std::vector<double>>> getWeights();
  void writeWeightsFile();

protected:
  void initInternal(std::string status_variable_name) override;
  void growInternal() override;
  void calculateAfterGrow(size_t tree_idx, bool oob) override;
  void allocatePredictMemory() override;
  void predictInternal(size_t tree_idx) override;
  void computePredictionErrorInternal() override;
  void writeOutputInternal() override;
  void writePredictionFile() override;
  void saveToFileInternal(std::ofstream &outfile) override;
  void loadFromFileInternal(std::ifstream &infile) override;

  // Classes of the dependent variable and classIDs for responses
  std::vector<double> class_values;
  std::vector<uint> response_classIDs;
  std::vector<std::vector<size_t>> sampleIDs_per_class;

  // Splitting weights
  std::vector<double> class_weights;

  // Class counts
  std::vector<std::unordered_map<double, size_t>> class_counts;
  // std::vector<std::unordered_map<double, size_t>> class_counts_internal;
  std::vector<std::unordered_map<double, size_t>> class_count;

  // Table with classifications and true classes
  std::map<std::pair<double, double>, size_t> classification_table;

  // OOB votes
  std::vector<std::unordered_map<double, size_t>> oob_votes;

  // OOB weights
  std::vector<std::vector<double>> oob_weights;

private:
  double getTreePrediction(size_t tree_idx, size_t sample_idx) const;
  size_t getTreePredictionTerminalNodeID(size_t tree_idx, size_t sample_idx) const;
  const std::vector<size_t>& getInbagCounts(size_t tree_idx);
};
} // namespace ranger