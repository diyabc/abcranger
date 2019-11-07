#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "readreftable.hpp"

#include "ForestOnlineRegression.hpp"
#include "readstatobs.hpp"
#include "matutils.hpp"
#include "DataDense.h"
#include "test-error.hpp"
#include "csv-eigen.hpp"
#include "ks.hpp"

#include "various.hpp"

using namespace ranger;
using namespace Eigen;

TEST_CASE("Online Ranger Regressor")
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
    nref = nref == 0 ? error.size() : nref;
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
                     0,                  // number of threads
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
    CHECK(oob_prior_error == Approx(0.148368).margin(RFTEST_TOLERANCE));
}

TEST_CASE("Online Ranger Regressor Distribution")
{
    size_t nref = 1000;

    MatrixXd E = read_matrix_file("regression.csv",',');
    std::vector<double> predsR = E.col(0) | to_vector;

    MatrixXd F = read_matrix_file("error.csv",',');
    VectorXd error = F(seq(0,nref-1),0);

    auto myread = readreftable("headerRF.txt", "reftableRF.bin", nref);
    auto nstat = myread.stats_names.size();
    MatrixXd statobs(1,nstat);
    statobs = Map<MatrixXd>(readStatObs("statobsRF.txt").data(),1,nstat);
    auto colnames = myread.stats_names;
    auto datastatobs = unique_cast<DataDense, Data>(std::make_unique<DataDense>(statobs, colnames, 1, nstat));
    // if (nref != 0 && nref <= error.size()) 
    //     error.erase(error.begin() + nref,error.end());
    addCols(myread.stats, error);
    colnames.push_back("Y");
    auto ntree = 50;
    auto nthreads = 8;
    auto ntest = 100;
    std::vector<double> mypredsR(ntest);

    for(auto i = 0; i < ntest; i++){
        loadbar(i,ntest);    
        ForestOnlineRegression forestreg;
        auto datastats = unique_cast<DataDense, Data>(std::make_unique<DataDense>(myread.stats, colnames, nref, nstat + 1));
        forestreg.init("Y",                       // dependant variable
                        MemoryMode::MEM_DOUBLE,    // memory mode double or float
                        std::move(datastats),    // data
                        std::move(datastatobs),  // predict
                        0,                         // mtry, if 0 sqrt(m -1) but should be m/3 in regression
                        "originalranger_out",              // output file name prefix
                        ntree,                     // number of trees
                        r(),                    // seed rd()
                        0,                  // number of threads
                        DEFAULT_IMPORTANCE_MODE,  // Default IMP_NONE
                        5,                         // default min node size (classif = 1, regression 5)
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
        auto& datareleased = static_cast<DataDense&>(*datastats.get());
        myread.stats = std::move(datareleased.data);
        datastatobs = forestreg.releasePred();
    }
    std::cout << (mypredsR | views::all) << std::endl;
    auto D = KSTest(predsR,mypredsR);
    auto pvalue = 1-psmirnov2x(D,predsR.size(),ntest);
    CHECK( pvalue >= 0.05 );
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