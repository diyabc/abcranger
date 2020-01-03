#include <fmt/format.h>

#include <cmath>

#include "EstimParam.hpp"
#include "ForestOnlineRegression.hpp"
#include "matutils.hpp"
#include "various.hpp"
#include "pls-eigen.hpp"
#include "parse_parexpr.hpp"
#include "forestQuantiles.hpp"

#include "DataDense.hpp"
#include "cxxopts.hpp"
#include <algorithm>
#include <fstream>
#include "range/v3/all.hpp"

using namespace ranger;
using namespace Eigen;
using namespace ranges;

EstimParamResults EstimParam_fun(Reftable &myread,
                                 std::vector<double> origobs,
                                 const cxxopts::ParseResult &opts,
                                 bool quiet)
{
    size_t nref, ntree, nthreads, noisecols, seed, minnodesize, ntest;
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
    ntest = opts["noob"].as<size_t>();
    chosenscen = static_cast<double>(opts["chosenscen"].as<size_t>());
    parameter_of_interest = opts["parameter"].as<std::string>();
    plsok = opts.count("nolinear") == 0;

    outfile = (opts.count("output") == 0) ? "estimparam_out" : opts["o"].as<std::string>();


    double p_threshold_PLS = 0.99;
    std::vector<double> samplefract{std::min(1e5,static_cast<double>(myread.nrec))/static_cast<double>(myread.nrec)};
    auto nstat = myread.stats_names.size();
    MatrixXd statobs(1, nstat);

    statobs = Map<MatrixXd>(origobs.data(), 1, nstat);

    std::size_t p1,p2;
    op_type op;
    parse_paramexpression(myread.params_names,parameter_of_interest,op, p1, p2);

    auto nparam = myread.params_names.size();

    size_t n = myread.nrec;

    VectorXd y(n);
    switch(op) {
        case op_type::none : 
            y = myread.params(all,p1);
            break;
        case op_type::divide :
            y = myread.params(all,p1).array() / myread.params(all,p2).array();
            break;
        case op_type::multiply :
            y = myread.params(all,p1)*myread.params(all,p2);
            break;
    }

    // myread.params = std::move(myread.params(indexesModel,param_num)).eval();
    if (y.array().isNaN().any()) {
        std::cout << "Error : there is some nan in the parameter data." << std::endl;
        exit(1);
    }


    EstimParamResults res;

    if (plsok) {
        size_t ncomp_total = static_cast<size_t>(lround(1.0 * static_cast<double>(nstat)));
        MatrixXd Projection;
        RowVectorXd mean,std;
        VectorXd percentYvar = pls(myread.stats,
                                y,
                                ncomp_total,Projection, mean, std,true);

        const std::string& pls_filename = outfile + ".plsvar";
        std::ofstream pls_file;
        if (!quiet) pls_file.open(pls_filename, std::ios::out);
        res.plsvar = std::vector<double>(percentYvar.size());
        for(auto i = 0; i < percentYvar.size(); i++) {
            if (!quiet) pls_file << percentYvar(i) << std::endl;
            res.plsvar[i] = percentYvar(i);
        }
        if (!quiet) pls_file.close();
        size_t nComposante_sel = percentYvar.size();

        if (!quiet) std::cout << "Selecting only " << nComposante_sel << " pls components." << std::endl;

        double sumPlsweights = Projection.col(0).array().abs().sum();
        auto weightedPlsfirst = Projection.col(0)/sumPlsweights;

        const std::string& plsweights_filename = outfile + ".plsweights";
        std::ofstream plsweights_file;
        if (!quiet) plsweights_file.open(plsweights_filename, std::ios::out);
        for(auto& p : views::zip(myread.stats_names, weightedPlsfirst)
            | to_vector
            | actions::sort([](auto& a, auto& b){ return std::abs(a.second) > std::abs(b.second); })) {
                if (!quiet) plsweights_file << p.first << " " << p.second << std::endl;
                res.plsweights.push_back(p);
            }

        plsweights_file.close();

        auto Xc = (myread.stats.array().rowwise()-mean.array()).rowwise()/std.array();
        addCols(myread.stats,(Xc.matrix() * Projection).leftCols(nComposante_sel));
        auto Xcobs = (statobs.array().rowwise()-mean.array()).rowwise()/std.array();
        addCols(statobs,(Xcobs.matrix() * Projection).leftCols(nComposante_sel));
        for(auto i = 0; i < nComposante_sel; i++)
            myread.stats_names.push_back("Comp " + std::to_string(i+1));

    }


    addNoise(myread, statobs, noisecols);
    std::vector<string> varwithouty = myread.stats_names;
    auto datastatobs = unique_cast<DataDense, Data>(std::make_unique<DataDense>(statobs,varwithouty, 1, varwithouty.size()));
    addCols(myread.stats,y);
    myread.stats_names.push_back("Y");

    auto datastats = unique_cast<DataDense, Data>(std::make_unique<DataDense>(myread.stats,myread.stats_names, nref, myread.stats_names.size()));

    ForestOnlineRegression forestreg;
    forestreg.init("Y",                       // dependant variable
                     MemoryMode::MEM_DOUBLE,    // memory mode double or float
                     std::move(datastats),    // data
                     std::move(datastatobs),  // predict
                     std::max(std::floor(static_cast<double>(myread.stats_names.size()-1)/3.0),1.0),                         // mtry, if 0 sqrt(m -1) but should be m/3 in regression
                     outfile,              // output file name prefix
                     ntree,                     // number of trees
                     (seeded ? seed : r()),                    // seed rd()
                     nthreads,                  // number of threads
                     ImportanceMode::IMP_GINI,  // Default IMP_NONE
                     5,                         // default min node size (classif = 1, regression 5)
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
                     DEFAULT_MAXDEPTH,
                     ntest);         // max_depth
    if (!quiet) forestreg.verbose_out = &std::cout;
    forestreg.run(!quiet,true);
    auto preds = forestreg.getPredictions();
    // Variable Importance
    res.variable_importance = forestreg.getImportance();
    if (!quiet) forestreg.writeImportanceFile();
    // OOB error by number of trees
    res.ntree_oob_error = preds[2][0];
    if (!quiet) forestreg.writeOOBErrorFile();
    // Values/weights
    res.values_weights = forestreg.getWeights();
    if (!quiet) forestreg.writeWeightsFile();


    auto dataptr2 = forestreg.releaseData();
    auto& datareleased2 = static_cast<DataDense&>(*dataptr2.get());
    datareleased2.data.conservativeResize(NoChange,nstat);
    myread.stats = std::move(datareleased2.data);
    myread.stats_names.resize(nstat);

    std::vector<double> probs{0.05,0.5,0.95};

    double MSE = 0;
    double NMSE = 0;
    double NMAE = 0;
    std::vector<double> rCI;
    std::vector<double> relativeCI;
    std::ostringstream os;
    // os << fmt::format("   real value");
    os << fmt::format("  expectation");
    os << fmt::format("     variance");
    os << fmt::format("    local mae");
    // os << fmt::format(" variance.cdf");
    for(auto prob : probs) os << fmt::format("  quant. {:>0.2f}",prob);
    os << std::endl;
    std::vector<double> obs;
    std::copy(y.begin(),y.end(),std::back_inserter(obs));

        
    double expectation = 0.0;
    double variance = 0.0;
    double mae = 0.0;
    // double variance2 = 0.0;
    for(auto i = 0; i < nref; i++) {
        expectation += preds[4][0][i] * y(i);
        if (!std::isnan(preds[0][0][i])) {
            double rest = y(i) - preds[0][0][i];
            variance += preds[4][0][i] * rest * rest;
            mae += preds[4][0][i] * std::abs(rest/y(i));
        }
        // double rest = y(i) - preds[1][0][j];
        // variance2 += preds[4][j][i] * rest * rest; 
    }
    std::vector<double> quants = forestQuantiles(obs,preds[4][0],probs);
    os << fmt::format("{:>13.6f}{:>13.6f}{:>13.6f}",expectation,variance,mae);
    res.expectation = expectation;
    res.variance = variance;
    res.quantiles = quants;
    for(auto quant : quants) os << fmt::format("{:>13.6f}",quant);
    os << std::endl;

    for(auto p : forestreg.oob_subset) {
        size_t j = p.second;
        std::vector<double> quants = forestQuantiles(obs,preds[5][j],probs);
        auto reality = y(p.first);
        auto diff = preds[0][0][p.first] - reality;
        // auto diff = expectation - reality;
        auto sqdiff = diff * diff;
        auto CI = quants[2] - quants[0];
        MSE += sqdiff;
        NMSE += sqdiff / reality;
        NMAE += std::abs(diff / reality);
        rCI.push_back(CI);
        relativeCI.push_back(CI / reality);
    }
    if (!quiet) {
        std::cout << os.str();
        std::cout.flush();
    }

    const std::string& predict_filename = outfile + ".predictions";
    std::ofstream predict_file;
    if (!quiet)  {
        predict_file.open(predict_filename, std::ios::out);
            if (!predict_file.good()) {
            throw std::runtime_error("Could not write to prediction file: " + predict_filename + ".");
        }
        predict_file << os.str();
        predict_file.flush();
        predict_file.close();
    }

    os.clear();
    os.str("");
    res.MSE = MSE/static_cast<double>(ntest); 
    os << fmt::format("{:>19} : {:<13}","MSE", res.MSE) << std::endl;
    res.NMSE = NMSE/static_cast<double>(ntest);
    os << fmt::format("{:>19} : {:<13}","NMSE", res.NMSE) << std::endl;
    res.NMAE = NMAE/static_cast<double>(ntest);
    os << fmt::format("{:>19} : {:<13}","NMAE", res.NMAE) << std::endl;
    res.meanCI = ranges::accumulate(rCI,0.0)/static_cast<double>(ntest);
    os << fmt::format("{:>19} : {:<13}","mean CI", res.meanCI) << std::endl;
    res.meanrelativeCI = ranges::accumulate(relativeCI,0.0)/static_cast<double>(ntest);
    os << fmt::format("{:>19} : {:<13}","mean relative CI", res.meanrelativeCI) << std::endl;
    res.medianCI = median(rCI);
    os << fmt::format("{:>19} : {:<13}","median CI",res.medianCI) << std::endl;
    res.medianrelativeCI = median(relativeCI);
    os << fmt::format("{:>19} : {:<13}","median relative CI", res.medianrelativeCI) << std::endl;

    if (!quiet) {
        std::cout << std::endl << "Test statistics" << std::endl;
        std::cout << os.str();
        std::cout.flush();
    }

    const std::string& teststats_filename = outfile + ".teststats";
    std::ofstream teststats_file;
    if (!quiet) {
        teststats_file.open(teststats_filename, std::ios::out);
        if (!teststats_file.good()) {
            throw std::runtime_error("Could not write to teststats file " + teststats_filename + ".");        
        }
        teststats_file << os.str();
        teststats_file.flush();
        teststats_file.close();
    }

    return res;
}