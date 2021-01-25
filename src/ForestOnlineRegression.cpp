#include <algorithm>
#include <stdexcept>
#include <string>
#include <cmath>
#include <map>
#include <random>
#include <range/v3/all.hpp>

#include "utility.h"
#include "ForestOnlineRegression.hpp"
#include "TreeRegression.h"
#include "Data.h"

using namespace ranges;

namespace ranger
{

void ForestOnlineRegression::initInternal(std::string status_variable_name)
{

  keep_inbag = true;
  // If mtry not set, use floored square root of number of independent variables
  if (mtry == 0)
  {
    long temp = std::round(sqrt((double)(num_variables - 1)));
    mtry = std::max(1l, temp);
  }

  // Set minimal node size
  if (min_node_size == 0)
  {
    min_node_size = DEFAULT_MIN_NODE_SIZE_REGRESSION;
  }

  // Sort data if memory saving mode
  if (!memory_saving_splitting)
  {
    data->sort();
  }
}

void ForestOnlineRegression::growInternal()
{
  trees.reserve(num_trees);
  for (size_t i = 0; i < num_trees; ++i)
  {
    trees.push_back(make_unique<TreeRegression>());
  }

  // predictions = std::vector<std::vector<std::vector<double>>>(1,
  //     std::vector<std::vector<double>>(1, std::vector<double>(num_samples, 0)));
  samples_oob_count.resize(num_samples, 0);
}

thread_local std::vector<size_t> samples_terminalnodes;

void ForestOnlineRegression::allocatePredictMemory()
{
  size_t num_prediction_samples = predict_data->getNumRows();
  predictions = std::vector<std::vector<std::vector<double>>>(6);
  /// predictions oob
  predictions[0] = std::vector<std::vector<double>>(1, std::vector<double>(num_samples,NAN));
    // OOB square error on n-trees (cumulative)
  predictions[2] = std::vector<std::vector<double>>(1,std::vector<double>(num_trees,0.0));
  // tree predictions
//  samples_terminalnodes.resize(num_samples);
  // Mutex for each sample
  mutex_samples = std::vector<std::mutex>(num_samples);
  // predictions[3] = std::vector<std::vector<double>>(1, std::vector<double>(num_samples));
  if (num_oob_weights > 0) {
    predictions[5] = std::vector<std::vector<double>>(num_oob_weights,std::vector<double>(num_samples,0));

    auto oob_subsetpreview = views::ints(static_cast<size_t>(0),num_samples) | to<std::vector>();
    shuffle(oob_subsetpreview);

    oob_subset = oob_subsetpreview 
      | views::slice(static_cast<size_t>(0),num_oob_weights)
      // | views::sample(num_oob_weights)
      | views::enumerate
      | views::transform([](auto p){ return std::make_pair(p.second,static_cast<size_t>(p.first)); })
      | to<std::map<size_t,size_t>>();
  }

  if (predict_all) 
    predictions[4] = std::vector<std::vector<double>>(num_prediction_samples, std::vector<double>(num_samples,0.0));
  if (prediction_type == TERMINALNODES)
    predictions[1] = std::vector<std::vector<double>>(num_prediction_samples, std::vector<double>(num_trees));
  else   /// predictions on data
    predictions[1] = std::vector<std::vector<double>>(1, std::vector<double>(num_prediction_samples,0.0));
  
  if (num_oob_weights > 0) predictions.push_back(std::vector<std::vector<double>>(0, std::vector<double>(num_samples,0.0)));
}

void ForestOnlineRegression::predictInternal(size_t tree_idx)
{
  // if (predict_all || prediction_type == TERMINALNODES) {
  //   // Get all tree predictions
  //   for (size_t tree_idx = 0; tree_idx < num_trees; ++tree_idx) {
  //     if (prediction_type == TERMINALNODES) {
  //       predictions[0][sample_idx][tree_idx] = getTreePredictionTerminalNodeID(tree_idx, sample_idx);
  //     } else {
  //       predictions[0][sample_idx][tree_idx] = getTreePrediction(tree_idx, sample_idx);
  //     }
  //   }
  // } else {
  //   // Mean over trees
  //   double prediction_sum = 0;
  //   for (size_t tree_idx = 0; tree_idx < num_trees; ++tree_idx) {
  //     prediction_sum += getTreePrediction(tree_idx, sample_idx);
  //   }
  //   predictions[0][0][sample_idx] = prediction_sum / num_trees;
  // }
  const std::vector<size_t>& inbag_count = getInbagCounts(tree_idx);

  for (size_t sample_idx = 0; sample_idx < predict_data->getNumRows(); ++sample_idx)
  {
    if (prediction_type == TERMINALNODES)
        predictions[1][sample_idx][tree_idx] = static_cast<double>(getTreePredictionTerminalNodeID(tree_idx, sample_idx));
    else {
      auto value = getTreePrediction(tree_idx, sample_idx);
      // if (std::isnan(value)) throw std::runtime_error("NaN value");
      if (std::isnan(value)) next;
      mutex_samples[sample_idx].lock();
      predictions[1][0][sample_idx] += value;
      mutex_samples[sample_idx].unlock();
      if (predict_all) {
        auto node = getTreePredictionTerminalNodeID(tree_idx, sample_idx);
        size_t Lb = 0;
        for (size_t sample_internal_idx = 0; sample_internal_idx < data->getNumRows(); ++sample_internal_idx) {
              auto nb = inbag_count[sample_internal_idx];
              if (nb > 0 && samples_terminalnodes[sample_internal_idx] == node) 
                Lb += nb;
        }
        for (size_t sample_internal_idx = 0; sample_internal_idx < data->getNumRows(); ++sample_internal_idx) {
              auto nb = inbag_count[sample_internal_idx];
              if (nb > 0 && samples_terminalnodes[sample_internal_idx] == node) {
                mutex_samples[sample_idx].lock();
                predictions[4][sample_idx][sample_internal_idx] += static_cast<double>(nb)/static_cast<double>(Lb);
                mutex_samples[sample_idx].unlock();
              }
        }
      }
    }
  }
}


void ForestOnlineRegression::calculateAfterGrow(size_t tree_idx, bool oob)
{
  // For each tree loop over OOB samples and count classes
  double se = 0.0;
  size_t num_predictions = 0;
  if (samples_terminalnodes.empty()) samples_terminalnodes.resize(num_samples);
  // std::map<double,size_t> leaves_map;
  const std::vector<size_t>& inbag_count = getInbagCounts(tree_idx);
  for (size_t sample_idx = 0; sample_idx < num_samples; sample_idx++) {
    double value = getTreePrediction(tree_idx, sample_idx);
    size_t node = getTreePredictionTerminalNodeID(tree_idx, sample_idx);
    auto nb = inbag_count[sample_idx];
    if (nb > 0) { // INBAG{
        samples_terminalnodes[sample_idx] = node;
    }
  }
  for (size_t sample_idx = 0; sample_idx < num_samples; sample_idx++) {
    double value = getTreePrediction(tree_idx, sample_idx);
    size_t node = getTreePredictionTerminalNodeID(tree_idx, sample_idx);
    auto nb = inbag_count[sample_idx];
    if (nb == 0) { // OOB
      mutex_samples[sample_idx].lock();
      if(std::isnan(predictions[0][0][sample_idx]))
        predictions[0][0][sample_idx] = 0.0;
      predictions[0][0][sample_idx] += value;
      ++samples_oob_count[sample_idx];
      mutex_samples[sample_idx].unlock();
      if (num_oob_weights > 0) {
        auto tofind = oob_subset.find(sample_idx);
        if (tofind != oob_subset.end()) {
          size_t oob_idx = tofind->second;
          size_t Lb = 0;

          for (size_t sample_internal_idx = 0; sample_internal_idx < num_samples; ++sample_internal_idx) {
                auto nb = inbag_count[sample_internal_idx];
                if (nb > 0 && samples_terminalnodes[sample_internal_idx] == node) {
                  Lb += nb;
                }
          }
          for (size_t sample_internal_idx = 0; sample_internal_idx < num_samples; ++sample_internal_idx) {
                auto nb = inbag_count[sample_internal_idx];
                if (nb > 0 && samples_terminalnodes[sample_internal_idx] == node) {
                  mutex_samples[sample_internal_idx].lock();
                  predictions[5][oob_idx][sample_internal_idx] += static_cast<double>(nb)/static_cast<double>(Lb);
                  mutex_samples[sample_internal_idx].unlock();
                }
          }
        }
      }
    }
    if (samples_oob_count[sample_idx] > 0) {
      auto real_value = data->get(sample_idx,dependent_varID);
      auto oob_value = predictions[0][0][sample_idx]/static_cast<double>(samples_oob_count[sample_idx]) - real_value;
      se += oob_value * oob_value;
      num_predictions++;
    }
  }
  predictions[2][0][tree_idx] = se/static_cast<double>(num_predictions);
}

void ForestOnlineRegression::computePredictionErrorInternal()
{

  // For each sample sum over trees where sample is OOB
  // for (size_t tree_idx = 0; tree_idx < num_trees; ++tree_idx) {
  //   for (size_t sample_idx = 0; sample_idx < trees[tree_idx]->getNumSamplesOob(); ++sample_idx) {
  //     size_t sampleID = trees[tree_idx]->getOobSampleIDs()[sample_idx];
  //     double value = getTreePrediction(tree_idx, sample_idx);

  //     predictions[0][0][sampleID] += value;
  //     ++samples_oob_count[sampleID];
  //   }
  // }

  // MSE with predictions and true data
  size_t num_predictions = 0;
  overall_prediction_error = 0;
  for (size_t i = 0; i < predictions[0][0].size(); ++i)
  {
    if (samples_oob_count[i] > 0)
    {
      ++num_predictions;
      predictions[0][0][i] /= (double)samples_oob_count[i];
      double predicted_value = predictions[0][0][i];
      double real_value = data->get(i, dependent_varID);
      overall_prediction_error += (predicted_value - real_value) * (predicted_value - real_value);
    }
    else
    {
      predictions[0][0][i] = NAN;
    }
  }

  overall_prediction_error /= (double)num_predictions;

  if (prediction_type != TERMINALNODES)
    for (auto sample_idx = 0; sample_idx < predict_data->getNumRows(); ++sample_idx)
      predictions[1][0][sample_idx] /= static_cast<double>(num_trees);

  if (predict_all) {
    for (size_t sample_idx = 0; sample_idx < predict_data->getNumRows(); ++sample_idx) 
      for (size_t sample_internal_idx = 0; sample_internal_idx < data->getNumRows(); ++sample_internal_idx)
        predictions[4][sample_idx][sample_internal_idx] /= static_cast<double>(num_trees);

    for (auto sample_idx: oob_subset) 
      for (size_t sample_internal_idx = 0; sample_internal_idx < data->getNumRows(); ++sample_internal_idx)
        predictions[5][sample_idx.second][sample_internal_idx] /= static_cast<double>(samples_oob_count[sample_idx.first]);
  }

  std::vector<double> sort_oob_trees(num_trees);
  for(auto i = 0; i < num_trees; i++) sort_oob_trees[i] = predictions[2][0][tree_order[i]];
  predictions[2][0] = sort_oob_trees;  

}

// #nocov start
void ForestOnlineRegression::writeOutputInternal()
{
  if (verbose_out)
  {
    *verbose_out << "Tree type:                         "
                 << "Regression" << std::endl;
  }
}

void ForestOnlineRegression::writeOOBErrorFile() {
  // Open confusion file for writing
  std::string filename = output_prefix + ".ooberror";
  std::ofstream outfile;
  outfile.open(filename, std::ios::out);
  if (!outfile.good())
  {
    throw std::runtime_error("Could not write to oob error file: " + filename + ".");
  }

  for (auto tree_idx = 0; tree_idx < num_trees; tree_idx++) {
    outfile << predictions[2][0][tree_idx] << std::endl;
  }
}

std::vector<std::pair<double,double>> ForestOnlineRegression::getWeights() {
  std::vector<std::pair<double,double>> res;
  for (auto sample_idx = 0; sample_idx < num_samples; sample_idx++) 
    res.push_back(std::make_pair(data->get(sample_idx,dependent_varID),predictions[4][0][sample_idx]));
  return res;
}

void ForestOnlineRegression::writeWeightsFile() {
  // Open confusion file for writing
  std::string filename = output_prefix + ".predweights";
  std::ofstream outfile;
  outfile.open(filename, std::ios::out);
  if (!outfile.good())
  {
    throw std::runtime_error("Could not write to predweights file: " + filename + ".");
  }
  outfile << "value,weight" << std::endl;
  for(auto& kv: getWeights())
    outfile << kv.first << "," << kv.second << std::endl;
}

void ForestOnlineRegression::writeConfusionFile()
{

  // Open confusion file for writing
  std::string filename = output_prefix + ".confusion";
  std::ofstream outfile;
  outfile.open(filename, std::ios::out);
  if (!outfile.good())
  {
    throw std::runtime_error("Could not write to confusion file: " + filename + ".");
  }

  // Write confusion to file
  outfile << "Overall OOB prediction error (MSE): " << overall_prediction_error << std::endl;

  outfile.close();
  if (verbose_out)
    *verbose_out << "Saved prediction error to file " << filename << "." << std::endl;
}

void ForestOnlineRegression::writePredictionFile()
{

  // Open prediction file for writing
  std::string filename = output_prefix + ".prediction";
  std::ofstream outfile;
  outfile.open(filename, std::ios::out);
  if (!outfile.good())
  {
    throw std::runtime_error("Could not write to prediction file: " + filename + ".");
  }

  // Write
  outfile << "Predictions: " << std::endl;
  if (predict_all)
  {
    for (size_t k = 0; k < num_trees; ++k)
    {
      outfile << "Tree " << k << ":" << std::endl;
      for (size_t i = 0; i < predictions.size(); ++i)
      {
        for (size_t j = 0; j < predictions[i].size(); ++j)
        {
          outfile << predictions[i][j][k] << std::endl;
        }
      }
      outfile << std::endl;
    }
  }
  else
  {
    for (size_t i = 0; i < predictions.size(); ++i)
    {
      for (size_t j = 0; j < predictions[i].size(); ++j)
      {
        for (size_t k = 0; k < predictions[i][j].size(); ++k)
        {
          outfile << predictions[i][j][k] << std::endl;
        }
      }
    }
  }

  if (verbose_out)
    *verbose_out << "Saved predictions to file " << filename << "." << std::endl;
}

void ForestOnlineRegression::saveToFileInternal(std::ofstream &outfile)
{

  // Write num_variables
  outfile.write((char *)&num_variables, sizeof(num_variables));

  // Write treetype
  TreeType treetype = TREE_REGRESSION;
  outfile.write((char *)&treetype, sizeof(treetype));
}

void ForestOnlineRegression::loadFromFileInternal(std::ifstream &infile)
{

  // Read number of variables
  size_t num_variables_saved;
  infile.read((char *)&num_variables_saved, sizeof(num_variables_saved));

  // Read treetype
  TreeType treetype;
  infile.read((char *)&treetype, sizeof(treetype));
  if (treetype != TREE_REGRESSION)
  {
    throw std::runtime_error("Wrong treetype. Loaded file is not a regression forest.");
  }

  for (size_t i = 0; i < num_trees; ++i)
  {

    // Read data
    std::vector<std::vector<size_t>> child_nodeIDs;
    readVector2D(child_nodeIDs, infile);
    std::vector<size_t> split_varIDs;
    readVector1D(split_varIDs, infile);
    std::vector<double> split_values;
    readVector1D(split_values, infile);

    // If dependent variable not in test data, change variable IDs accordingly
    if (num_variables_saved > num_variables)
    {
      for (auto &varID : split_varIDs)
      {
        if (varID >= dependent_varID)
        {
          --varID;
        }
      }
    }

    // Create tree
    trees.push_back(std::make_unique<TreeRegression>(child_nodeIDs, split_varIDs, split_values));
  }
}

double ForestOnlineRegression::getTreePrediction(size_t tree_idx, size_t sample_idx) const
{
  const auto &tree = dynamic_cast<const TreeRegression &>(*trees[tree_idx]);
  return tree.getPrediction(sample_idx);
}

size_t ForestOnlineRegression::getTreePredictionTerminalNodeID(size_t tree_idx, size_t sample_idx) const
{
  const auto &tree = dynamic_cast<const TreeRegression &>(*trees[tree_idx]);
  return tree.getPredictionTerminalNodeID(sample_idx);
}


const std::vector<size_t>& ForestOnlineRegression::getInbagCounts(size_t tree_idx)
{
  const auto &tree = dynamic_cast<const TreeRegression &>(*trees[tree_idx]);
  return tree.getInbagCounts();
}
// #nocov end

} // namespace ranger
