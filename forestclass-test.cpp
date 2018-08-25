#define BOOST_TEST_MODULE ForestClassCpp
#include <boost/test/unit_test.hpp>

#include "ForestOnlineClassification.hpp"
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
    // auto myread = readreftable("headerRF.txt","reftableRF.bin", 3000);
    // auto statobs = readStatObs("statobsRF.txt");
    // auto nstat = myread.stats_names.size();
    // addCol(myread.stats,myread.scenarios);
    // auto colnames = myread.stats_names;
    // colnames.push_back("Y");
    // auto datastatobs = unique_cast<DataDouble,Data>(std::make_unique<DataDouble>(myread.stats,colnames,myread.nrec,nstat+1));
    // ForestOnlineClassification forestclass;
    // auto ntree = 1;
    // auto nthreads = 1;

    // forestclass.init("Y", 
    //                  MemoryMode::MEM_DOUBLE,
    //                  std::move(datastatobs),
    //                  0,
    //                  "ranger_out",
    //                  ntree,
    //                  rd(),
    //                  nthreads,
    //                  ImportanceMode::IMP_GINI,
    //                  0,
    //                  "",
    //                  false,
    //                  true,
    //                  myread.stats_names,
    //                  false,
    //                  DEFAULT_SPLITRULE,
    //                  false,
    //                  DEFAULT_SAMPLE_FRACTION,
    //                  DEFAULT_ALPHA,
    //                  DEFAULT_MINPROP,
    //                  false,
    //                  DEFAULT_PREDICTIONTYPE,
    //                  DEFAULT_NUM_RANDOM_SPLITS,
    //                  false);
    // forestclass.setverboseOutput(&cout);
    // forestclass.run(true);
    // auto preds = forestclass.getPredictions();
    // forestclass.writeOutput();

    BOOST_TEST( 0 == 0 );
}