#include <unordered_map>
#include <algorithm>
#include <iterator>
#include <random>
#include <stdexcept>
#include <cmath>
#include <string>

#include "utility.h"
#include "ForestOnlineClassification.hpp"
#include "TreeClassification.h"
#include "Data.h"

namespace ranger
{

void ForestOnlineClassification::initInternal(std::string status_variable_name)
{

  // If mtry not set, use floored square root of number of independent variables.
  if (mtry == 0)
  {
    unsigned long temp = sqrt((double)(num_variables - 1));
    mtry = std::max((unsigned long)1, temp);
  }

  // Set minimal node size
  if (min_node_size == 0)
  {
    min_node_size = DEFAULT_MIN_NODE_SIZE_CLASSIFICATION;
  }

  // Create class_values and response_classIDs
  if (!prediction_mode)
  {
    for (size_t i = 0; i < num_samples; ++i)
    {
      double value = data->get(i, dependent_varID);

      // If classID is already in class_values, use ID. Else create a new one.
      uint classID = find(class_values.begin(), class_values.end(), value) - class_values.begin();
      if (classID == class_values.size())
      {
        class_values.push_back(value);
      }
      response_classIDs.push_back(classID);
    }
  }

  // Create sampleIDs_per_class if required
  if (sample_fraction.size() > 1)
  {
    sampleIDs_per_class.resize(sample_fraction.size());
    for (auto &v : sampleIDs_per_class)
    {
      v.reserve(num_samples);
    }
    for (size_t i = 0; i < num_samples; ++i)
    {
      size_t classID = response_classIDs[i];
      sampleIDs_per_class[classID].push_back(i);
    }
  }

  // Set class weights all to 1
  class_weights = std::vector<double>(class_values.size(), 1.0);

  // Sort data if memory saving mode
  if (!memory_saving_splitting)
  {
    data->sort();
  }
}

void ForestOnlineClassification::growInternal()
{
  trees.reserve(num_trees);
  for (size_t i = 0; i < num_trees; ++i)
  {
    trees.push_back(
        std::make_unique<TreeClassification>(&class_values, &response_classIDs, &sampleIDs_per_class, &class_weights));
  }

    // Class counts for samples
  class_counts.reserve(num_samples);
  // class_counts_internal.reserve(num_samples);
  for (size_t i = 0; i < num_samples; ++i)
  {
    class_counts.push_back(std::unordered_map<double, size_t>());
    // class_counts_internal.push_back(std::unordered_map<double, size_t>());
  }

}

void ForestOnlineClassification::allocatePredictMemory()
{
  size_t num_prediction_samples = predict_data->getNumRows();
  class_count = std::vector<std::unordered_map<double, size_t>>(num_prediction_samples);
  predictions = std::vector<std::vector<std::vector<double>>>(3);
  // Predictions on the OOB set 
  predictions[0] = std::vector<std::vector<double>>(1,std::vector<double>(num_samples));
  // OOB Error classifications on n-trees (non-cumulative)
  predictions[2] = std::vector<std::vector<double>>(1,std::vector<double>(num_trees,0.0));
  // predictions[2] = std::vector<std::vector<double>>(1, std::vector<double>(num_samples));
  if (predict_all || prediction_type == TERMINALNODES)
  {
    // Predictions on the provided samples by each tree
    predictions[1] = std::vector<std::vector<double>>(num_prediction_samples, std::vector<double>(num_trees));
  }
  else
  {
    // Predictions on the provided samples
    predictions[1] = std::vector<std::vector<double>>(1, std::vector<double>(num_prediction_samples));
  }
}

void ForestOnlineClassification::predictInternal(size_t tree_idx)
{
  // if (predict_all || prediction_type == TERMINALNODES)
  // {
  //   // Get all tree predictions
  //   for (size_t tree_idx = 0; tree_idx < num_trees; ++tree_idx)
  //   {
  //     if (prediction_type == TERMINALNODES)
  //     {
  //       predictions[0][sample_idx][tree_idx] = getTreePredictionTerminalNodeID(tree_idx, sample_idx);
  //     }
  //     else
  //     {
  //       predictions[0][sample_idx][tree_idx] = getTreePrediction(tree_idx, sample_idx);
  //     }
  //   }
  // }
  // else
  // {
  //   // Count classes over trees and save class with maximum count
  //   std::unordered_map<double, size_t> class_count;
  //   for (size_t tree_idx = 0; tree_idx < num_trees; ++tree_idx)
  //   {
  //     ++class_count[getTreePrediction(tree_idx, sample_idx)];
  //   }
  //   predictions[0][0][sample_idx] = mostFrequentValue(class_count, random_number_generator);
  // }
    for (size_t sample_idx = 0; sample_idx < predict_data->getNumRows(); ++sample_idx)
  {
    if (predict_all || prediction_type == TERMINALNODES) {
      if (prediction_type == TERMINALNODES)
      {
        predictions[1][sample_idx][tree_idx] = getTreePredictionTerminalNodeID(tree_idx, sample_idx);
      }
      else
      {
        predictions[1][sample_idx][tree_idx] = getTreePrediction(tree_idx, sample_idx);
      }
    } else {
      mutex_post.lock();
      ++class_count[sample_idx][getTreePrediction(tree_idx, sample_idx)];
      mutex_post.unlock();
    }
  }

}

void ForestOnlineClassification::calculateAfterGrow(size_t tree_idx, bool oob) {
  // For each tree loop over OOB samples and count classes
      double to_add = 0.0;
      auto numOOB = trees[tree_idx]->getNumSamplesOob();
      for (size_t sample_idx = 0; sample_idx < numOOB; ++sample_idx)
      {
        size_t sampleID = trees[tree_idx]->getOobSampleIDs()[sample_idx];
        auto res = static_cast<size_t>(getTreePrediction(tree_idx, sample_idx));
        mutex_post.lock();
        ++class_counts[sampleID][res];
        if (!class_counts[sampleID].empty())
          to_add += (mostFrequentValue(class_counts[sampleID], random_number_generator) == data->get(sampleID,dependent_varID)) ? 0.0 : 1.0;
        mutex_post.unlock();
      }
      predictions[2][0][tree_idx] += to_add/static_cast<double>(numOOB);
    // for (size_t sample_idx = 0; sample_idx < num_samples; sample_idx++) {
    //   if (!class_counts[sample_idx].empty())
    //     predictions[2][0][tree_idx] += (mostFrequentValue(class_counts[sample_idx], random_number_generator) == data->get(sample_idx,dependent_varID)) ? 0.0 : 1.0;
    // }

    // else 
    //   for (size_t sample_idx = 0; sample_idx < num_samples; ++sample_idx)
    //   {
    //     ++class_counts_internal[sample_idx][getTreePrediction(tree_idx,sample_idx)];
    //   };
}

void ForestOnlineClassification::computePredictionErrorInternal()
{
  // For each tree loop over OOB samples and count classes
  // for (size_t tree_idx = 0; tree_idx < num_trees; ++tree_idx) {
  //   for (size_t sample_idx = 0; sample_idx < trees[tree_idx]->getNumSamplesOob(); ++sample_idx) {
  //     size_t sampleID = trees[tree_idx]->getOobSampleIDs()[sample_idx];
  //     ++class_counts[sampleID][getTreePrediction(tree_idx, sample_idx)];
  //   }
  // }

  // Compute majority vote for each sample
  // predictions = std::vector<std::vector<std::vector<double>>>(1,
  //                                                             std::vector<std::vector<double>>(1, std::vector<double>(num_samples)));
  for (size_t i = 0; i < num_samples; ++i)
  {
    if (!class_counts[i].empty())
    {
      predictions[0][0][i] = mostFrequentValue(class_counts[i], random_number_generator);
    }
    else
    {
      predictions[0][0][i] = NAN;
    }
      // predictions[2][0][i] = mostFrequentValue(class_counts_internal[i], random_number_generator)
  }

  // Compare predictions with true data
  size_t num_missclassifications = 0;
  size_t num_predictions = 0;
  for (size_t i = 0; i < predictions[0][0].size(); ++i)
  {
    double predicted_value = predictions[0][0][i];
    if (!std::isnan(predicted_value))
    {
      ++num_predictions;
      double real_value = data->get(i, dependent_varID);
      if (predicted_value != real_value)
      {
        ++num_missclassifications;
      }
      ++classification_table[std::make_pair(real_value, predicted_value)];
    }
  }
  overall_prediction_error = (double)num_missclassifications / (double)num_predictions;

  if (!(predict_all || prediction_type == TERMINALNODES))
    for(auto sample_idx = 0; sample_idx < predict_data->getNumRows(); sample_idx++) {
      predictions[1][0][sample_idx] = mostFrequentValue(class_count[sample_idx], random_number_generator);
    }
  std::vector<double> sort_oob_trees(num_trees);
  for(auto i = 0; i < num_trees; i++) sort_oob_trees[i] = predictions[2][0][tree_order[i]];
  predictions[2][0] = sort_oob_trees;
}

// #nocov start
void ForestOnlineClassification::writeOutputInternal()
{
  if (verbose_out)
  {
    *verbose_out << "Tree type:                         "
                 << "Classification" << std::endl;
  }
}

void ForestOnlineClassification::writeOOBErrorFile() {
  // Open confusion file for writing
  std::string filename = output_prefix + ".ooberror";
  std::ofstream outfile;
  outfile.open(filename, std::ios::out);
  if (!outfile.good())
  {
    throw std::runtime_error("Could not write to OOBError file: " + filename + ".");
  }

  for (auto tree_idx = 0; tree_idx < num_trees; tree_idx++) {
    outfile << predictions[2][0][tree_idx] << std::endl;
  }
}

std::vector<std::vector<size_t>> ForestOnlineClassification::getConfusion() {
  std::vector<std::vector<size_t>> res(class_values.size(),std::vector<size_t>(class_values.size()));

  std::sort(class_values.begin(),class_values.end());
  double classtot;
  double classok;
  for(auto i = 0; i < class_values.size(); i++)
    for(auto j = 0; j < class_values.size(); j++) {
      size_t predicted_value = class_values[i];
      size_t real_value = class_values[j];
      res[i][j] = classification_table[std::make_pair(real_value, predicted_value)];
    }

  return res;
}

void ForestOnlineClassification::writeConfusionFile()
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
  outfile << "Overall OOB prediction error (Fraction missclassified): " << overall_prediction_error << std::endl;
  outfile << std::endl;
  outfile << "Class specific prediction errors:" << std::endl;
  outfile << "           ";

  std::sort(class_values.begin(),class_values.end());
  for (auto &class_value : class_values)
  {
    outfile << "     " << class_value;
  }
  outfile << " class.error" << std::endl;
  double classtot;
  double classok;
  for (auto &predicted_value : class_values)
  {
    outfile << "predicted " << predicted_value << "     ";
    classtot = 0.0;
    for (auto &real_value : class_values)
    {
      size_t value = classification_table[std::make_pair(real_value, predicted_value)];
      if (real_value == predicted_value) classok = value;
      classtot += value;
      outfile << value;
      if (value < 10)
      {
        outfile << "     ";
      }
      else if (value < 100)
      {
        outfile << "    ";
      }
      else if (value < 1000)
      {
        outfile << "   ";
      }
      else if (value < 10000)
      {
        outfile << "  ";
      }
      else if (value < 100000)
      {
        outfile << " ";
      }
    }
    outfile << " " << (classtot - classok) / classtot << std::endl;
  }

  outfile.close();
  // if (verbose_out)
  //   *verbose_out << "Saved confusion matrix to file " << filename << "." << std::endl;
}

void ForestOnlineClassification::writePredictionFile()
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
      for (size_t i = 1; i < predictions.size(); ++i)
      {
        for (size_t j = 0; j < predictions[i].size(); ++j)
        {
          outfile << predictions[1][j][k] << std::endl;
        }
      }
      outfile << std::endl;
    }
  }
  else
  {
    for (size_t i = 1; i < predictions.size(); ++i)
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

void ForestOnlineClassification::saveToFileInternal(std::ofstream &outfile)
{

  // Write num_variables
  outfile.write((char *)&num_variables, sizeof(num_variables));

  // Write treetype
  TreeType treetype = TREE_CLASSIFICATION;
  outfile.write((char *)&treetype, sizeof(treetype));

  // Write class_values
  saveVector1D(class_values, outfile);
}

void ForestOnlineClassification::loadFromFileInternal(std::ifstream &infile)
{

  // Read number of variables
  size_t num_variables_saved;
  infile.read((char *)&num_variables_saved, sizeof(num_variables_saved));

  // Read treetype
  TreeType treetype;
  infile.read((char *)&treetype, sizeof(treetype));
  if (treetype != TREE_CLASSIFICATION)
  {
    throw std::runtime_error("Wrong treetype. Loaded file is not a classification forest.");
  }

  // Read class_values
  readVector1D(class_values, infile);

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
    trees.push_back(
        std::make_unique<TreeClassification>(child_nodeIDs, split_varIDs, split_values, &class_values, &response_classIDs));
  }
}

double ForestOnlineClassification::getTreePrediction(size_t tree_idx, size_t sample_idx) const
{
  const auto &tree = dynamic_cast<const TreeClassification &>(*trees[tree_idx]);
  return tree.getPrediction(sample_idx);
}

size_t ForestOnlineClassification::getTreePredictionTerminalNodeID(size_t tree_idx, size_t sample_idx) const
{
  const auto &tree = dynamic_cast<const TreeClassification &>(*trees[tree_idx]);
  return tree.getPredictionTerminalNodeID(sample_idx);
}

void ForestOnlineClassification::setverboseOutput(std::ostream* verbose_output) {
  this->verbose_out = verbose_output;
}

// #nocov end

} // namespace ranger
