#define BOOST_TEST_MODULE ForestOnlineRegCpp
#include <boost/test/unit_test.hpp>

#include "readreftable.hpp"

#include "ForestOnlineRegression.hpp"
#include "readstatobs.hpp"
#include "matutils.hpp"
#include "DataDense.h"
#include "test-error.hpp"
#include "forestonlinereg-predsR.hpp"
#include "ks.hpp"

using namespace ranger;
using namespace Eigen;

std::vector<double> DEFAULT_SAMPLE_FRACTION = std::vector<double>({1});

template <class T_SRC, class T_DEST>
std::unique_ptr<T_DEST> unique_cast(std::unique_ptr<T_SRC> &&src)
{
    if (!src)
        return std::unique_ptr<T_DEST>();

    // Throws a std::bad_cast() if this doesn't work out
    T_DEST *dest_ptr = &dynamic_cast<T_DEST &>(*src.get());

    src.release();
    std::unique_ptr<T_DEST> ret(dest_ptr);
    return ret;
}

BOOST_AUTO_TEST_CASE(InitForestOnlineReg, *boost::unit_test::tolerance(RFTEST_TOLERANCE))
{
    size_t nref = 0;
    auto myread = readreftable("headerRF.txt", "reftableRF.bin", nref);
    auto nstat = myread.stats_names.size();
    MatrixXd statobs(1,nstat);
    statobs = Map<MatrixXd>(readStatObs("statobsRF.txt").data(),1,nstat);
    auto colnames = myread.stats_names;
    auto datastatobs = unique_cast<DataDense, Data>(std::make_unique<DataDense>(statobs, colnames, 1, nstat + 1));
    if (nref != 0 && nref <= error.size()) 
        error.erase(error.begin() + nref,error.end());
    addCols(myread.stats, Map<VectorXd>(error.data(),nref));
    colnames.push_back("Y");
    auto datastats = unique_cast<DataDense, Data>(std::make_unique<DataDense>(myread.stats, colnames, myread.nrec, nstat + 1));
    ForestOnlineRegression forestreg;
    auto ntree = 500;
    auto nthreads = 8;

    forestreg.init("Y",                       // dependant variable
                     MemoryMode::MEM_DOUBLE,    // memory mode double or float
                     std::move(datastats),    // data
                     std::move(datastatobs),  // predict
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
    auto preds = forestreg.getPredictions();
    auto oob_prior_error = forestreg.getOverallPredictionError();
    BOOST_TEST(oob_prior_error == 0.148368);
}

BOOST_AUTO_TEST_CASE(InitForestOnlineRegKS)
{
    size_t nref = 1000;
    auto myread = readreftable("headerRF.txt", "reftableRF.bin", nref);
    auto nstat = myread.stats_names.size();
    MatrixXd statobs(1,nstat);
    statobs = Map<MatrixXd>(readStatObs("statobsRF.txt").data(),1,nstat);
    auto colnames = myread.stats_names;
    auto datastatobs = unique_cast<DataDense, Data>(std::make_unique<DataDense>(statobs, colnames, 1, nstat));
    if (nref != 0 && nref <= error.size()) 
        error.erase(error.begin() + nref,error.end());
    addCols(myread.stats, Map<VectorXd>(error.data(),nref));
    colnames.push_back("Y");
    auto datastats = unique_cast<DataDense, Data>(std::make_unique<DataDense>(myread.stats, colnames, myread.nrec, nstat + 1));
    auto ntree = 50;
    auto nthreads = 8;
    auto ntest = predsR.size();
    std::vector<double> mypredsR(ntest);

    for(auto i = 0; i < ntest; i++){    
        ForestOnlineRegression forestreg;
        forestreg.init("Y",                       // dependant variable
                        MemoryMode::MEM_DOUBLE,    // memory mode double or float
                        std::move(datastats),    // data
                        std::move(datastatobs),  // predict
                        0,                         // mtry, if 0 sqrt(m -1) but should be m/3 in regression
                        "originalranger_out",              // output file name prefix
                        ntree,                     // number of trees
                        0,                    // seed rd()
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
        auto preds = forestreg.getPredictions();
        mypredsR[i] = preds[1][0][0]; 
        datastats = forestreg.releaseData();
        datastatobs = forestreg.releasePred();
    }
    auto D = KSTest(predsR,mypredsR);
    auto pvalue = 1-psmirnov2x(D,ntest,ntest);
    BOOST_TEST ( pvalue >= 0.05 );
}

// BOOST_AUTO_TEST_CASE(InitForestOnlineReg, *boost::unit_test::tolerance(RFTEST_TOLERANCE))
// {
//     size_t nref = 0;
//     auto myread = readreftable("headerRF.txt", "reftableRF.bin", nref);
//     auto nstat = myread.stats_names.size();
//     MatrixXd statobs(1,nstat);
//     statobs = Map<MatrixXd>(readStatObs("statobsRF.txt").data(),1,nstat);
//     auto colnames = myread.stats_names;
//     auto datastatobs = unique_cast<DataDense, Data>(std::make_unique<DataDense>(statobs, colnames, myread.nrec, nstat + 1));
//     if (nref != 0 && nref <= error.size()) 
//         error.erase(error.begin() + nref,error.end());
//     addCols(myread.stats, Map<VectorXd>(error.data(),nref));
//     colnames.push_back("Y");
//     auto datastats = unique_cast<DataDense, Data>(std::make_unique<DataDense>(myread.stats, colnames, myread.nrec, nstat + 1));
//     ForestOnlineRegression forestreg;
//     auto ntree = 500;
//     auto nthreads = 8;

//     forestreg.init("Y",                       // dependant variable
//                      MemoryMode::MEM_DOUBLE,    // memory mode double or float
//                      std::move(datastats),    // data
//                      std::move(datastatobs),  // predict
//                      0,                         // mtry, if 0 sqrt(m -1) but should be m/3 in regression
//                      "originalranger_out",              // output file name prefix
//                      ntree,                     // number of trees
//                      123456,                    // seed rd()
//                      nthreads,                  // number of threads
//                      DEFAULT_IMPORTANCE_MODE,  // Default IMP_NONE
//                      0,                         // default min node size (classif = 1, regression 5)
//                      "",                        // status variable name, only for survival
//                      false,                     // prediction mode (true = predict)
//                      true,                      // replace
//                      std::vector<string>(0),        // unordered variables names
//                      false,                     // memory_saving_splitting
//                      DEFAULT_SPLITRULE,         // gini for classif variance for  regression
//                      false,                     // predict_all
//                      DEFAULT_SAMPLE_FRACTION,   // sample_fraction 1 if replace else 0.632
//                      DEFAULT_ALPHA,             // alpha
//                      DEFAULT_MINPROP,           // miniprop
//                      false,                     // holdout
//                      DEFAULT_PREDICTIONTYPE,    // prediction type
//                      DEFAULT_NUM_RANDOM_SPLITS, // num_random_splits
//                      false,                     //order_snps
//                      DEFAULT_MAXDEPTH);         // max_depth
//     forestreg.run(true,true);
//     auto preds = forestreg.getPredictions();
//     auto oob_prior_error = forestreg.getOverallPredictionError();
//     BOOST_TEST(oob_prior_error == 0.148368);
//     BOOST_TEST(preds[1][0][0] == 0.242067);
//     // double predicted = 0.0;
//     // for(auto i = 0; i < ntree; i++) predicted += preds[1][0][i];
//     // predicted /= static_cast<double>(ntree);
//     // BOOST_TEST(predicted == 0.159167);
// }