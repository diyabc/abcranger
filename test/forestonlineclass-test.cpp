#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "ForestOnlineClassification.hpp"
#include "readstatobs.hpp"
#include "readreftable.hpp"
#include "matutils.hpp"
#include "DataDense.h"
using namespace ranger;
using namespace Eigen;

std::vector<double> DEFAULT_SAMPLE_FRACTION = std::vector<double>({1});

#include "unique_cast.hpp"

TEST_CASE("Online Ranger classifier")
{
    auto myread = readreftable("headerRF.txt", "reftableRF.bin", 0);
    auto nstat = myread.stats_names.size();
    MatrixXd statobs(1,nstat);
    statobs = Map<MatrixXd>(readStatObs("statobsRF.txt").data(),1,nstat);
    auto datastatobs = unique_cast<DataDense, Data>(std::make_unique<DataDense>(statobs, myread.stats_names, 1, myread.stats_names.size()));
    addScen(myread);
    auto datastats = unique_cast<DataDense, Data>(std::make_unique<DataDense>(myread.stats, myread.stats_names, myread.nrec, myread.stats_names.size()));
    // ForestOnlineClassification forestclass;
    ForestOnlineClassification forestclass;
    auto ntree = 500;
    auto nthreads = 8;

    // void ranger::Forest::init(std::string dependent_variable_name,
    //                           ranger::MemoryMode memory_mode,
    //                           std::unique_ptr<...> input_data,
    //                           ranger::uint mtry,
    //                           std::string output_prefix,
    //                           ranger::uint num_trees,
    //                           ranger::uint seed,
    //                           ranger::uint num_threads,
    //                           ranger::ImportanceMode importance_mode,
    //                           ranger::uint min_node_size,
    //                           std::string status_variable_name,
    //                           bool prediction_mode,
    //                           bool sample_with_replacement,
    //                           const std::vector<...> &unordered_variable_names,
    //                           bool memory_saving_splitting,
    //                           ranger::SplitRule splitrule,
    //                           bool predict_all,
    //                           std::vector<...> &sample_fraction,
    //                           double alpha,
    //                           double minprop,
    //                           bool holdout,
    //                           ranger::PredictionType prediction_type,
    //                           ranger::uint num_random_splits,
    //                           bool order_snps,
    //                           ranger::uint max_depth)
    forestclass.init("Y",                       // dependant variable
                     MemoryMode::MEM_DOUBLE,    // memory mode double or float
                     std::move(datastats),    // data
                     std::move(datastatobs),  // predict
                     0,                         // mtry, if 0 sqrt(m -1) but should be m/3 in regression
                     "onlineranger_out",              // output file name prefix
                     ntree,                     // number of trees
                     123456,                    // seed rd()
                     nthreads,                  // number of threads
                     DEFAULT_IMPORTANCE_MODE,  // Default IMP_NONE
                     0,                         // default min node size (classif = 1, regression 5)
                     "",                        // status variable name, only for survival
                     false,                     // prediction mode (true = predict)
                     true,                      // replace
                     std::vector<string>(0),        // unordered variables names
                     false,                     // memory_saving_splitting
                     DEFAULT_SPLITRULE,         // gini for classif variance for  regression
                     false,                     // predict_all
                     DEFAULT_SAMPLE_FRACTION,   // sample_fraction 1 if replace else 0.632
                     DEFAULT_ALPHA,             // alpha
                     DEFAULT_MINPROP,           // miniprop
                     false,                     // holdout
                     DEFAULT_PREDICTIONTYPE,    // prediction type
                     DEFAULT_NUM_RANDOM_SPLITS, // num_random_splits
                     false,                     //order_snps
                     DEFAULT_MAXDEPTH);         // max_depth
    // forestclass.setverboseOutput(&cout);
    forestclass.run(true, true);
    auto preds = forestclass.getPredictions();
    auto oob_prior_error = forestclass.getOverallPredictionError();
    CHECK(oob_prior_error == Approx(0.231833).margin(RFTEST_TOLERANCE));
    CHECK(preds[1][0][0] == 3.0);
}