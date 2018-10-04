#define BOOST_TEST_MODULE ForestClassCpp
#include <boost/test/unit_test.hpp>

// #include "ForestOnlineClassification.hpp"
#include "ForestClassification.h"
#include "readstatobs.hpp"
#include "readreftable.hpp"
#include "matutils.hpp"
#include "DataDouble.h"

using namespace ranger;

std::vector<double> DEFAULT_SAMPLE_FRACTION = std::vector<double>({1});

template <class T_SRC, class T_DEST>
std::unique_ptr<T_DEST> unique_cast(std::unique_ptr<T_SRC> &&src)
{
    if (!src) return std::unique_ptr<T_DEST>();

    // Throws a std::bad_cast() if this doesn't work out
    T_DEST *dest_ptr = &dynamic_cast<T_DEST &>(*src.get());

    src.release();
    std::unique_ptr<T_DEST> ret(dest_ptr);
    return ret;
}

BOOST_AUTO_TEST_CASE(InitForestClass) {
    auto myread = readreftable("headerRF.txt","reftableRF.bin", 3000);
    auto statobs = readStatObs("statobsRF.txt");
    auto nstat = myread.stats_names.size();
    addCol(myread.stats,myread.scenarios);
    auto colnames = myread.stats_names;
    colnames.push_back("Y");
    auto datastatobs = unique_cast<DataDouble,Data>(std::make_unique<DataDouble>(myread.stats,colnames,myread.nrec,nstat+1));
    // ForestOnlineClassification forestclass;
    ForestClassification forestclass;
    auto ntree = 1;
    auto nthreads = 8;

    forestclass.init("Y", // dependant variable
                     MemoryMode::MEM_DOUBLE, // memory mode double or float
                     std::move(datastatobs), // data
                     0, // mtry, if 0 sqrt(m -1) but should be m/3 in regression
                     "ranger_out", // output file name prefix 
                     ntree, // number of trees
                     rd(), // seed
                     nthreads, // number of threads
                     ImportanceMode::IMP_GINI, // Default IMP_NONE
                     0, // default min node size (classif = 1, regression 5)
                     "", // status variable name, only for survival
                     false, // prediction mode (true = predict)
                     true, // replace
                     myread.stats_names, // unordered variables names
                     false, // memory_saving_splitting
                     DEFAULT_SPLITRULE, // gini for classif variance for  regression
                     false, // predict_all
                     DEFAULT_SAMPLE_FRACTION, // sample_fraction 1 if replace else 0.632
                     DEFAULT_ALPHA, // alpha
                     DEFAULT_MINPROP, // miniprop
                     false, // holdout
                     DEFAULT_PREDICTIONTYPE, // prediction type
                     DEFAULT_NUM_RANDOM_SPLITS, // num_random_splits
                     false); // order_snps
    // forestclass.setverboseOutput(&cout);
    forestclass.run(true);
    auto preds = forestclass.getPredictions();
    forestclass.writeOutput();

    BOOST_TEST( 0 == 0 );
}