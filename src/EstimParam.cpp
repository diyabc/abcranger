#include <fmt/format.h>


// using namespace boost::accumulators;
// // typedef accumulator_set<double, stats<tag::weighted_tail_quantile<right> >, double> accumulator_t;

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>
#include <boost/accumulators/statistics/weighted_tail_quantile.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/weighted_mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <boost/accumulators/statistics/weighted_variance.hpp>
#include <boost/accumulators/statistics/median.hpp>
#include <boost/accumulators/statistics/weighted_median.hpp>
// #include <boost/accumulators/statistics/weighted_extended_p_square.hpp>

#include <cmath>

#include "EstimParam.hpp"
#include "ForestOnlineRegression.hpp"
#include "matutils.hpp"
#include "various.hpp"
#include "pls-eigen.hpp"
#include "parse_parexpr.hpp"
#include "forestQuantiles.hpp"

#include "DataDense.h"
#include "cxxopts.hpp"
#include <algorithm>
#include <fstream>
#include "range/v3/all.hpp"



namespace bacc = boost::accumulators;
using namespace ranger;
using namespace Eigen;
using namespace ranges;

EstimParamResults EstimParam_fun(Reftable &myread,
                                 std::vector<double> origobs,
                                 const cxxopts::ParseResult &opts,
                                 bool quie)
{
    size_t nref, ntree, nthreads, noisecols, seed, minnodesize, ntrain, ntest;
    std::string outfile, parameter_of_interest;
    double chosenscen;   
    bool plsok,seeded;

    nref = opts["n"].as<size_t>();
    ntree = opts["t"].as<size_t>();
    nthreads = opts["j"].as<size_t>();
    noisecols = opts["c"].as<size_t>();
    seeded = opts.count("s") != 0;
    if (seeded)
        seed = opts["s"].as<size_t>();
    minnodesize = opts["m"].as<size_t>();
    outfile = opts["o"].as<std::string>();
    if (opts.count("ntrain") == 0 
        || opts.count("ntest") == 0 
        || opts.count("parameter") == 0
        || opts.count("chosenscen") == 0) {
        std::cout << "Error : please provide ntrain, ntest, parameter and chosenscen arguments." << std::endl;
        exit(1);
    }
    ntrain = opts["ntrain"].as<size_t>();
    ntest = opts["ntest"].as<size_t>();
    chosenscen = static_cast<double>(opts["chosenscen"].as<size_t>());
    parameter_of_interest = opts["parameter"].as<std::string>();
    plsok = opts.count("nopls") == 0;

    double p_threshold_PLS = 0.99;
    std::vector<double> samplefract{std::min(1e5,static_cast<double>(myread.nrec))/static_cast<double>(myread.nrec)};
    auto nstat = myread.stats_names.size();
    MatrixXd statobs(1, nstat);

    statobs = Map<MatrixXd>(origobs.data(), 1, nstat);

    std::size_t p1,p2;
    op_type op;
    parse_paramexpression(myread.params_names,parameter_of_interest,op, p1, p2);


    size_t K = myread.nrecscen.size();
    auto nparam = myread.params_names.size();

    std::vector<size_t> indexesModel = myread.scenarios 
        | view::enumerate
        | view::filter([chosenscen](const auto& a){ return a.second == chosenscen; })
        | view::keys;

    size_t n = indexesModel.size();
    if (n < ntest + ntrain) {
        std::cout << "Error : insufficient samples for the test/train requested sizes (" << n << " samples)." << std::endl;
        exit(1);
    }

    myread.stats = std::move(myread.stats(indexesModel,all)).eval();
    VectorXd paramof(n);
    switch(op) {
        case op_type::none : 
            paramof = myread.params(indexesModel,p1);
            break;
        case op_type::divide :
            paramof = myread.params(indexesModel,p1).array() / myread.params(indexesModel,p2).array();
            break;
        case op_type::multiply :
            paramof = myread.params(indexesModel,p1)*myread.params(indexesModel,p2);
            break;
    }

    // myread.params = std::move(myread.params(indexesModel,param_num)).eval();
    myread.params = paramof;
    if (myread.params.array().isNaN().any()) {
        std::cout << "Error : there is some nan in the parameter data." << std::endl;
        exit(1);
    }

    auto tosplit = view::ints(static_cast<size_t>(0),n)
        | to_vector
        | action::shuffle(gen);
    std::vector<size_t> indicesTrain = tosplit | view::take(ntrain);
    std::vector<size_t> indicesTest  = tosplit | view::slice(ntrain,ntrain+ntest);

    VectorXd y = myread.params(indicesTrain,0);
    MatrixXd x = myread.stats(indicesTrain,all);

    indicesTest = view::ints(static_cast<size_t>(0),n-ntrain)
        | view::sample(ntest,gen);
    VectorXd ytest = myread.params(indicesTest,0);
    MatrixXd xtest = myread.stats(indicesTest,all);
    addRows(statobs,xtest);

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

    EstimParamResults res;

    if (plsok) {
        size_t ncomp_total = static_cast<size_t>(lround(1.0 * static_cast<double>(nstat)));
        MatrixXd Projection;
        RowVectorXd mean,std;
        VectorXd percentYvar = pls(x,
                                y,
                                ncomp_total,Projection, mean, std);

        const std::string& pls_filename = outfile + ".plsvar";
        std::ofstream pls_file;
        pls_file.open(pls_filename, std::ios::out);
        res.plsvar = std::vector<double>(percentYvar.size());
        for(auto i = 0; i < percentYvar.size(); i++) {
            pls_file << percentYvar(i) << std::endl;
            res.plsvar[i] = percentYvar(i);
        }
        pls_file.close();
        size_t nComposante_sel = percentYvar.size();
        // double p_var_PLS = percentYvar(percentYvar.rows()-1) * p_threshold_PLS;

        // const auto& enum_p_var_PLS = percentYvar
        //                     | view::enumerate
        //                     | to_vector;  
        // size_t nComposante_sel = 
        //     ranges::find_if(enum_p_var_PLS,
        //                     [&p_var_PLS](auto v) { return v.second > p_var_PLS; })->first;

        std::cout << "Selecting only " << nComposante_sel << " pls components." << std::endl;

        double sumPlsweights = Projection.col(0).array().abs().sum();
        auto weightedPlsfirst = Projection.col(0)/sumPlsweights;

        const std::string& plsweights_filename = outfile + ".plsweights";
        std::ofstream plsweights_file;
        plsweights_file.open(plsweights_filename, std::ios::out);
        for(auto& p : view::zip(myread.stats_names, weightedPlsfirst)
            | to_vector
            | action::sort([](auto& a, auto& b){ return std::abs(a.second) > std::abs(b.second); })) {
                plsweights_file << p.first << " " << p.second << std::endl;
                res.plsweights.push_back(p);
            }

        plsweights_file.close();

        auto Xc = (x.array().rowwise()-mean.array()).rowwise()/std.array();
        addCols(myreadTrain.stats,(Xc.matrix() * Projection).leftCols(nComposante_sel));
        auto Xcobs = (statobs.array().rowwise()-mean.array()).rowwise()/std.array();
        addCols(statobs,(Xcobs.matrix() * Projection).leftCols(nComposante_sel));
        for(auto i = 0; i < nComposante_sel; i++)
            myreadTrain.stats_names.push_back("Comp " + std::to_string(i+1));

    }


    addNoise(myreadTrain, statobs, noisecols);
    std::vector<string> varwithouty = myreadTrain.stats_names;
    auto datastatobs = unique_cast<DataDense, Data>(std::make_unique<DataDense>(statobs,varwithouty, ntest + 1, varwithouty.size()));
    addCols(myreadTrain.stats,y);
    myreadTrain.stats_names.push_back("Y");

    auto datastats = unique_cast<DataDense, Data>(std::make_unique<DataDense>(myreadTrain.stats,myreadTrain.stats_names, ntrain, myreadTrain.stats_names.size()));

    ForestOnlineRegression forestreg;
    forestreg.init("Y",                       // dependant variable
                     MemoryMode::MEM_DOUBLE,    // memory mode double or float
                     std::move(datastats),    // data
                     std::move(datastatobs),  // predict
                     static_cast<double>(myreadTrain.stats_names.size()-1)/3.0,                         // mtry, if 0 sqrt(m -1) but should be m/3 in regression
                     outfile,              // output file name prefix
                     ntree,                     // number of trees
                     (seeded ? seed : r()),                    // seed rd()
                     nthreads,                  // number of threads
                     ImportanceMode::IMP_GINI,  // Default IMP_NONE
                     0,                         // default min node size (classif = 1, regression 5)
                     "",                        // status variable name, only for survival
                     false,                     // prediction mode (true = predict)
                     true,                      // replace
                     std::vector<string>(0),        // unordered variables names
                     false,                     // memory_saving_splitting
                     DEFAULT_SPLITRULE,         // gini for classif variance for  regression
                     true,                     // predict_all
                     samplefract,   // sample_fraction 1 if replace else 0.632
                     DEFAULT_ALPHA,             // alpha
                     DEFAULT_MINPROP,           // miniprop
                     false,                     // holdout
                     DEFAULT_PREDICTIONTYPE,    // prediction type
                     DEFAULT_NUM_RANDOM_SPLITS, // num_random_splits
                     false,                     //order_snps
                     DEFAULT_MAXDEPTH);         // max_depth
    forestreg.run(true,true);
    auto preds = forestreg.getPredictions();
    // Variable Importance
    res.variable_importance = forestreg.getImportance();
    forestreg.writeImportanceFile();
    // OOB error by number of trees
    res.ntree_oob_error = preds[2][0];
    forestreg.writeOOBErrorFile();
    // Values/weights
    res.values_weights = forestreg.getWeights();
    forestreg.writeWeightsFile();

    std::vector<double> probs{0.05,0.5,0.95};

    std::ostringstream os;
    // os << fmt::format("   real value");
    os << fmt::format("  expectation");
    os << fmt::format("     variance");
    // os << fmt::format(" variance.cdf");
    for(auto prob : probs) os << fmt::format("  quant. {:>0.2f}",prob);
    os << std::endl;
    bacc::accumulator_set<double, bacc::stats<bacc::tag::mean, 
                                              bacc::tag::median(bacc::with_p_square_quantile)>>
            CIacc, CI_relatifacc;
    bacc::accumulator_set<double, bacc::stats<bacc::tag::mean>> 
        MSEacc,NMSEacc, NMAEacc;
    std::vector<double> obs;
    std::copy(y.begin(),y.end(),std::back_inserter(obs));
    for(auto j = 0; j < ntest+1; j++) {
        // os << fmt::format("{:>13.3f}",(j == 0? NAN : ytest(j-1)));
        
        double expectation = 0.0;
        double variance = 0.0;
        // double variance2 = 0.0;
        for(auto i = 0; i < ntrain; i++) {
            expectation += preds[4][j][i] * y(i);
            if (!std::isnan(preds[0][0][i])) {
                double rest = y(i) - preds[0][0][i];
                variance += preds[4][j][i] * rest * rest;
            }
            double rest = y(i) - preds[1][0][j];
            // variance2 += preds[4][j][i] * rest * rest; 
        }
        std::vector<double> quants = forestQuantiles(obs,preds[4][j],probs);
        if (j == 0) {            
            os << fmt::format("{:>13.6f}{:>13.6f}",expectation,variance);
            res.expectation = expectation;
            res.variance = variance;
            res.quantiles = quants;
            for(auto quant : quants) os << fmt::format("{:>13.6f}",quant);
            os << std::endl;
        } else {
            auto reality = ytest(j-1);
            auto diff = expectation - reality;
            auto sqdiff = diff * diff;
            auto CI = quants[2] - quants[0];
            MSEacc(sqdiff);
            NMSEacc(sqdiff / reality);
            NMAEacc(std::abs(diff / reality));
            CIacc(CI);
            CI_relatifacc(CI / reality);
        }
    }
    std::cout << os.str();
    std::cout.flush();

    const std::string& predict_filename = outfile + ".predictions";
    std::ofstream predict_file;
    predict_file.open(predict_filename, std::ios::out);
    if (!predict_file.good()) {
        throw std::runtime_error("Could not write to prediction file: " + predict_filename + ".");
    }
    predict_file << os.str();
    predict_file.flush();
    predict_file.close();

    os.clear();
    os.str("");
    res.MSE = bacc::mean(MSEacc);    
    os << fmt::format("{:>19} : {:<13}","MSE", res.MSE) << std::endl;
    res.NMSE = bacc::mean(NMSEacc);
    os << fmt::format("{:>19} : {:<13}","NMSE", res.NMSE) << std::endl;
    res.NMAE = bacc::mean(NMAEacc);
    os << fmt::format("{:>19} : {:<13}","NMAE", res.NMAE) << std::endl;
    res.meanCI = bacc::mean(CIacc);
    os << fmt::format("{:>19} : {:<13}","mean CI", res.meanCI) << std::endl;
    res.meanrelativeCI = bacc::mean(CI_relatifacc);
    os << fmt::format("{:>19} : {:<13}","mean relative CI", res.meanrelativeCI) << std::endl;
    res.medianCI = bacc::median(CIacc);
    os << fmt::format("{:>19} : {:<13}","median CI",res.medianCI) << std::endl;
    res.medianrelativeCI = bacc::median(CI_relatifacc);
    os << fmt::format("{:>19} : {:<13}","median relative CI", res.medianrelativeCI) << std::endl;

    std::cout << std::endl << "Test statistics" << std::endl;
    std::cout << os.str();
    std::cout.flush();

    const std::string& teststats_filename = outfile + ".teststats";
    std::ofstream teststats_file;
    teststats_file.open(teststats_filename, std::ios::out);
    if (!teststats_file.good()) {
        throw std::runtime_error("Could not write to teststats file " + teststats_filename + ".");        
    }
    teststats_file << os.str();
    teststats_file.flush();
    teststats_file.close();

    return res;
}