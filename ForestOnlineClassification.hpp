#pragma once

#include <map>
#include "Forest.h"

namespace ranger
{
class ForestOnlineClassification : public Forest
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

  // Manually set the outputstream for verbose out
  void setverboseOutput(std::ostream* verbose_output);

protected:
  void initInternal(std::string status_variable_name) override;
  void growInternal() override;
  void allocatePredictMemory() override;
  void predictInternal(size_t sample_idx) override;
  void computePredictionErrorInternal() override;
  void writeOutputInternal() override;
  void writeConfusionFile() override;
  void writePredictionFile() override;
  void saveToFileInternal(std::ofstream &outfile) override;
  void loadFromFileInternal(std::ifstream &infile) override;

  // Classes of the dependent variable and classIDs for responses
  std::vector<double> class_values;
  std::vector<uint> response_classIDs;
  std::vector<std::vector<size_t>> sampleIDs_per_class;

  // Splitting weights
  std::vector<double> class_weights;

  // Table with classifications and true classes
  std::map<std::pair<double, double>, size_t> classification_table;

private:
  double getTreePrediction(size_t tree_idx, size_t sample_idx) const;
  size_t getTreePredictionTerminalNodeID(size_t tree_idx, size_t sample_idx) const;
};
} // namespace ranger