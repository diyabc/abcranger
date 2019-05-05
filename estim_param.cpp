#include <fmt/format.h>
#include "ForestOnlineRegression.hpp"
#include "readstatobs.hpp"
#include "readreftable.hpp"
#include "matutils.hpp"
#include "various.hpp"
#include "DataDense.h"
#include "pls-eigen.hpp"
#include <range/v3/all.hpp>

using namespace ranger;
using namespace Eigen;
using namespace ranges;

auto print = [](int i) { std::cout << i << ' '; };

int main()
{
    // size_t nref = 3000;
    const std::string& outfile = "estimparam_out";
    double p_threshold_PLS = 0.99;

    size_t ncores = 8;
    auto myread = readreftable("headerRF.txt", "reftableRF.bin");
    auto nstat = myread.stats_names.size();
    MatrixXd statobs(1, nstat);
    statobs = Map<MatrixXd>(readStatObs("statobsRF.txt").data(), 1, nstat);

    std::string parameter_of_interest = "ra";
    auto param_enum = myread.params_names | view::enumerate | to_vector;
    const auto& param_found = ranges::find_if(param_enum,
        [&parameter_of_interest](const auto& s) { return s.second == parameter_of_interest; });
    size_t param_num = param_found->first;
    // auto ranum = rind(myread.pa)

    size_t noisecols = 5;
    size_t K = myread.nrecscen.size();
    auto nparam = myread.params_names.size();

    const double chosenscen = 1.0;
    std::vector<size_t> indexesModel = myread.scenarios 
        | view::enumerate
        | view::filter([chosenscen](const auto& a){ return a.second == chosenscen; })
        | view::keys;
    
    myread.stats = std::move(myread.stats(indexesModel,all)).eval();
    myread.params = std::move(myread.params(indexesModel,param_num)).eval();

    size_t n = indexesModel.size();
    size_t ntrain = n;
    size_t ntest = 0;
    // size_t ntrain = 1000;
    // size_t ntest = 50;

    auto tosplit = view::ints(static_cast<size_t>(0),n)
        | to_vector
        | action::shuffle(gen);
    std::vector<size_t> indicesTrain = tosplit | view::take(ntrain);
    std::vector<size_t> indicesTest  = tosplit | view::slice(ntrain,n);

    VectorXd y = myread.params(indicesTrain,0);
    // MatrixXd ytest = myread.params(indicesTest,0);
    std::cout << y.mean() << std::endl;

    MatrixXd x = myread.stats(indicesTrain,all);
    // MatrixXd xtest = myread.stats(indicesTest,all);

    // indicesTest = view::ints(static_cast<size_t>(0),n-ntrain)
    //     | view::sample(ntest,gen);

    size_t ncomp_total = static_cast<size_t>(lround(1.0 * static_cast<double>(nstat)));
    MatrixXd Projection;
    RowVectorXd mean,std;
    // VectorXd percentYvar = pls(myread.stats(indicesTrain,all),
    //                    myread.params(indicesTrain,0),
    //                    ncomp_total,Projection, mean, std);
    VectorXd percentYvar = pls(x,
                               y,
                               ncomp_total,Projection, mean, std);

    const std::string& pls_filename = outfile + ".plsvar";
    std::ofstream pls_file;
    pls_file.open(pls_filename, std::ios::out);
    for(auto& v: percentYvar.array()) pls_file << v << std::endl;
    pls_file.close();
    double p_var_PLS = percentYvar(percentYvar.rows()-1) * p_threshold_PLS;

    const auto& enum_p_var_PLS = percentYvar
                        | view::enumerate
                        | to_vector;  
    size_t nComposante_sel = 
        ranges::find_if(enum_p_var_PLS,
                        [&p_var_PLS](auto v) { return v.second > p_var_PLS; })->first-1;

    std::cout << "Selecting only " << nComposante_sel << " pls components." << std::endl;

    // for(auto& s: myread.stats_names) plsweights_file << fmt::format(" {:>12}",s);
    // plsweights_file << std::endl;
    double sumPlsweights = Projection.col(0).array().abs().sum();
    auto weightedPlsfirst = Projection.col(0)/sumPlsweights;


    const std::string& plsweights_filename = outfile + ".plsweights";
    std::ofstream plsweights_file;
    plsweights_file.open(plsweights_filename, std::ios::out);
    for(auto& p : view::zip(myread.stats_names, weightedPlsfirst)
        | to_vector
        | action::sort([](auto& a, auto& b){ return std::abs(a.second) > std::abs(b.second); }))
        plsweights_file << p.first << " " << p.second << std::endl;

    plsweights_file.close();

    Reftable myreadTrain = {
        ntrain,
        myread.nrecscen,
        myread.nparam,
        {parameter_of_interest},
        myread.stats_names,
        x,
        y,
        std::vector<double>{}
    };
    auto Xc = (x.array().rowwise()-mean.array()).rowwise()/std.array();
    addCols(myreadTrain.stats,(Xc.matrix() * Projection).leftCols(nComposante_sel));
    MatrixXd xobs(statobs);
    auto Xcobs = (xobs.array().rowwise()-mean.array()).rowwise()/std.array();
    addCols(xobs,(Xcobs.matrix() * Projection).leftCols(nComposante_sel));
    for(auto i = 0; i < nComposante_sel; i++)
        myreadTrain.stats_names.push_back("Comp " + std::to_string(i+1));

    addNoise(myreadTrain, statobs, noisecols);
    std::vector<string> varwithouty = myreadTrain.stats_names;
    auto datastatobs = unique_cast<DataDense, Data>(std::make_unique<DataDense>(statobs,varwithouty, 1, varwithouty.size()));
    addCols(myreadTrain.stats,y);
    myreadTrain.stats_names.push_back("Y");
//    addScen(myreadTrain);
    auto datastats = unique_cast<DataDense, Data>(std::make_unique<DataDense>(myreadTrain.stats,myreadTrain.stats_names, ntrain, myreadTrain.stats_names.size()));

    size_t ntree = 1000;
    size_t nthreads = 8;

    ForestOnlineRegression forestreg;
    forestreg.init("Y",                       // dependant variable
                     MemoryMode::MEM_DOUBLE,    // memory mode double or float
                     std::move(datastats),    // data
                     std::move(datastatobs),  // predict
                     static_cast<double>(myreadTrain.stats_names.size()-1)/3.0,                         // mtry, if 0 sqrt(m -1) but should be m/3 in regression
                     outfile,              // output file name prefix
                     ntree,                     // number of trees
                     0,                    // seed rd()
                     nthreads,                  // number of threads
                     ImportanceMode::IMP_GINI,  // Default IMP_NONE
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
    forestreg.writeImportanceFile();
    forestreg.writeOOBErrorFile();
    forestreg.writeConfusionFile();
    // for(auto i = 0; i < ntrain; i++) std::cout << y(i) <<  " " << preds[4][0][i] << std::endl;

    std::cout << "Prediction : " << preds[1][0][0] << std::endl;
    // double prediction = 0.0;
    // for(auto i =0; i < ntree; i++) prediction += preds[1][0][i];
    // prediction /= static_cast<double>(ntree);
    // std::cout << "Prediction : " << prediction << std::endl;

    // double expectation = 0.0;
    // for(auto i = 0; i < ntrain; i++) expectation += y(i) * preds[4][0][i];
    // expectation /= static_cast<double>(ntree);
    // std::cout << "Expectation : " << expectation << std::endl;


    // addLda(myread, statobs);
    // addNoise(myread, statobs, noisecols);
    // addScen(myread);
    // std::vector<string> varwithouty(myread.stats_names.size()-1);
    // for(auto i = 0; i < varwithouty.size(); i++) varwithouty[i] = myread.stats_names[i];

    // auto datastatobs = unique_cast<DataDense, Data>(std::make_unique<DataDense>(statobs, varwithouty, 1, varwithouty.size()));
    // auto datastats = unique_cast<DataDense, Data>(std::make_unique<DataDense>(myread.stats, myread.stats_names, myread.nrec, myread.stats_names.size()));

    // auto ntree = 1000;
    // auto nthreads = 8;
}