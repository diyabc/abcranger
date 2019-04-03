#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "ForestRegression.h"
#include "readstatobs.hpp"
#include "readreftable.hpp"
#include "matutils.hpp"
#include "DataDense.h"
#include "test-error.hpp"

#include "various.hpp"

using namespace ranger;


TEST_CASE("Standard Ranger Regresser")
{
    auto myread = readreftable("headerRF.txt", "reftableRF.bin", 0);
    auto nstat = myread.stats_names.size();
    auto colnames = myread.stats_names;
    addCols(myread.stats, Map<VectorXd>(error.data(),error.size()));
    colnames.push_back("Y");
    auto datastats = unique_cast<DataDense, Data>(std::make_unique<DataDense>(myread.stats, colnames, myread.nrec, nstat + 1));
    ForestRegression forestreg;
    auto ntree = 500;
    auto nthreads = 8;

    forestreg.init("Y",                       // dependant variable
                     MemoryMode::MEM_DOUBLE,    // memory mode double or float
                     std::move(datastats),    // data
                     0,                         // mtry, if 0 sqrt(m -1) but should be m/3 in regression
                     "originalranger_out",              // output file name prefix
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
    forestreg.run(true,true);
    auto oob_prior_error = forestreg.getOverallPredictionError();
    CHECK(oob_prior_error == Approx(0.148368).margin(RFTEST_TOLERANCE));

}