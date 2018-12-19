#include "ForestOnlineClassification.hpp"
#include "ForestOnlineRegression.hpp"
#include "readstatobs.hpp"
#include "readreftable.hpp"
#include "matutils.hpp"
#include "DataDense.h"
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

int main()
{
    size_t nref = 3000;
    auto myread = readreftable("headerRF.txt", "reftableRF.bin", nref);
    auto nstat = myread.stats_names.size();
    size_t noisecols = 5;
    size_t K = myread.nrecscen.size();
    MatrixXd statobs(1, nstat);
    statobs = Map<MatrixXd>(readStatObs("statobsRF.txt").data(), 1, nstat);
    addLda(myread, statobs);
    addNoise(myread, statobs, noisecols);
    addScen(myread);
    std::vector<string> varwithouty(myread.stats_names.size()-1);
    for(auto i = 0; i < varwithouty.size(); i++) varwithouty[i] = myread.stats_names[i];

    auto datastatobs = unique_cast<DataDense, Data>(std::make_unique<DataDense>(statobs, varwithouty, 1, varwithouty.size()));
    auto datastats = unique_cast<DataDense, Data>(std::make_unique<DataDense>(myread.stats, myread.stats_names, myread.nrec, myread.stats_names.size()));

    ForestOnlineClassification forestclass;
    auto ntree = 1000;
    auto nthreads = 8;
    forestclass.init("Y",                       // dependant variable
                     MemoryMode::MEM_DOUBLE,    // memory mode double or float
                     std::move(datastats),      // data
                     std::move(datastatobs),    // predict
                     0,                         // mtry, if 0 sqrt(m -1) but should be m/3 in regression
                     "onlineranger_out",        // output file name prefix
                     ntree,                     // number of trees
                     123457,                    // seed rd()
                     nthreads,                  // number of threads
                     ranger::IMP_GINI,          // Default IMP_NONE
                     0,                         // default min node size (classif = 1, regression 5)
                     "",                        // status variable name, only for survival
                     false,                     // prediction mode (true = predict)
                     true,                      // replace
                     std::vector<string>(0),    // unordered variables names
                     false,                     // memory_saving_splitting
                     DEFAULT_SPLITRULE,         // gini for classif variance for  regression
                     true,                     // predict_all
                     DEFAULT_SAMPLE_FRACTION,   // sample_fraction 1 if replace else 0.632
                     DEFAULT_ALPHA,             // alpha
                     DEFAULT_MINPROP,           // miniprop
                     false,                     // holdout
                     DEFAULT_PREDICTIONTYPE,    // prediction type
                     DEFAULT_NUM_RANDOM_SPLITS, // num_random_splits
                     false,                     //order_snps
                     DEFAULT_MAXDEPTH);         // max_depth

    forestclass.run(true, true);
    auto preds = forestclass.getPredictions();
    auto oob_prior_error = forestclass.getOverallPredictionError();
    forestclass.writeConfusionFile();
    forestclass.writeImportanceFile();
    vector<double> votes(K);
    for(auto& tree_pred : preds[1][0]) votes[static_cast<size_t>(tree_pred-1)]++;
    size_t predicted_model = std::distance(votes.begin(),std::max_element(votes.begin(),votes.end()));
    std::cout << "Predicted model : " << predicted_model + 1 << std::endl;
    std::cout << "Votes : " << std::endl;
    for(auto i = 0; i < votes.size(); i++) {
        std::cout << "class " << i+1 << " : " << votes[i]/ntree << endl;
    }
    bool machin = false;
    auto dataptr = forestclass.releaseData();
    auto& datareleased = static_cast<DataDense&>(*dataptr.get());
    size_t ycol = datareleased.getNumCols() - 1;
    for(size_t i = 0; i < preds[0][0].size(); i++) {
        datareleased.set(ycol,i,preds[0][0][i] == myread.scenarios[i] ? 1.0 : 0.0, machin);  
    }
    auto statobsreleased = forestclass.releasePred();
    ForestOnlineRegression forestreg;


    forestreg.init("Y",                       // dependant variable
                     MemoryMode::MEM_DOUBLE,    // memory mode double or float
                     std::move(dataptr),    // data
                     std::move(statobsreleased),  // predict
                     0,                         // mtry, if 0 sqrt(m -1) but should be m/3 in regression
                     "originalranger_out",              // output file name prefix
                     ntree,                     // number of trees
                     123457,                    // seed rd()
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
                     DEFAULT_MAXDEPTH);   

    forestreg.run(true,true);
    auto predserr = forestreg.getPredictions();
    std::cout << "Post proba : " << predserr[1][0][0] << std::endl;
    // for(auto i = 0; i < ntree; i++) 
    //     std::cout << i << " -> " << preds[2][0][i]/nref << std::endl;
    std::cout << "OK !" << std::endl;
}