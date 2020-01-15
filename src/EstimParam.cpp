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
                                 const cxxopts::ParseResult opts,
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
    // res.oob_error = preds[2][0][ntree-1];
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

    std::vector<double> rCI,relativeCI,lrCI,lrelativeCI;
    std::ostringstream os;

    std::vector<double> obs;
    std::copy(y.begin(),y.end(),std::back_inserter(obs));

        
    double expectation = 0.0;
    double variance = 0.0;
    // double mae = 0.0;
    // double variance2 = 0.0;
    for(auto i = 0; i < nref; i++) {
        expectation += preds[4][0][i] * y(i);
        if (!std::isnan(preds[0][0][i])) {
            double rest = y(i) - preds[0][0][i];
            variance += preds[4][0][i] * rest * rest;
            // mae += preds[4][0][i] * std::abs(rest/y(i));
        }
        // double rest = y(i) - preds[1][0][j];
        // variance2 += preds[4][j][i] * rest * rest; 
    }
    std::vector<double> quants = forestQuantiles(obs,preds[4][0],probs);
    std::map<std::string,double> point_estimates{
        { "Expectation", expectation },
        { "Median", quants[1] },
        { "Quantile_0.05", quants[0] },
        { "Quantile_0.95", quants[2] },
        { "Variance", variance }
    };

    os << "Parameter estimation (point estimates)" << std::endl;
    for(auto p: point_estimates) os << fmt::format("{:>14}",p.first);
    os << std::endl;
    for(auto p: point_estimates) os << fmt::format("{:>14.6f}",p.second);
    os << std::endl;
    for(auto p: point_estimates) res.point_estimates.insert(p);
    os << std::endl;

    std::map<std::string,std::string> global_local{
        { "global", "Global (prior) errors" },
        { "local", "Local (posterior) errors" }
    };

    std::map<std::string,std::string> mean_median_ci{
        { "mean", "Computed from the mean taken as point estimate" },
        { "median", "Computed from the median taken as point estimate" },
        { "ci", "Confidence interval measures"}
    };

    std::vector<std::string> mean_median_ci_l{"mean","median","ci"};

    std::vector<std::string> computed{"NMAE","MSE","NMSE"};

    std::map<std::string,std::string> ci{
        { "cov", "90% coverage" },
        { "meanCI", "Mean 90% CI" },
        { "meanRCI", "Mean relative 90% CI" },
        { "medianCI", "Median 90% CI" },
        { "medianRCI", "Median relative 90% CI" }
    };

    for(auto g_l : global_local) {
        for(auto c: computed) {
            res.errors[g_l.first]["mean"][c] = 0.0;
            res.errors[g_l.first]["median"][c] = 0.0;
        }
        for(auto c: ci) {
            if (g_l.first == "global")
                res.errors[g_l.first]["ci"][c.first] = 0.0;
        }
    }

    std::vector<double> ones(nref,1.0);
    double postmed = quants[1];

    double sumw = 0.0;
    for(auto p : forestreg.oob_subset) {
        // auto oobsumw = 0.0;
        // for (auto w : preds[5][p.second]) oobsumw += w; 
        // // // std::cout << "oob : " << p.first << " " << oobsumw << std::endl;
        // for(auto i = 0; i < nref; i++)  preds[5][p.second][i] /= oobsumw;
        std::vector<double> quants = forestQuantiles(obs,preds[5][p.second],probs);
        auto reality = y(p.first);
        auto w = preds[4][0][p.first];
        sumw += w;
        // auto w = preds[5][p.second][p.first] * nref;
        auto diff = preds[0][0][p.first] - reality;
        auto diff2 = (quants[1] - reality) ;
        auto sqdiff = diff * diff;
        auto sqdiff2 = diff2 * diff2;
        auto CI = (quants[2] - quants[0]);
        // std::cout << "CI : " << p.first << " " << CI << std::endl;
        res.errors["global"]["mean"]["NMAE"] += std::abs(diff / reality);
        res.errors["global"]["mean"]["MSE"] += sqdiff;
        res.errors["global"]["mean"]["NMSE"] += sqdiff / reality;
        res.errors["global"]["median"]["NMAE"] += std::abs(diff2 / reality);
        res.errors["global"]["median"]["MSE"] += sqdiff2;
        res.errors["global"]["median"]["NMSE"] += sqdiff2 / reality;
        res.errors["local"]["mean"]["NMAE"] += w * std::abs(diff / reality);
        res.errors["local"]["mean"]["MSE"] += w * sqdiff;
        res.errors["local"]["mean"]["NMSE"] += w * sqdiff / reality;
        res.errors["local"]["median"]["NMAE"] += w * std::abs(diff2 / reality);
        res.errors["local"]["median"]["MSE"] += w * sqdiff2;
        res.errors["local"]["median"]["NMSE"] += w * sqdiff2 / reality;
        double inside = ((reality <= quants[2]) && (reality >= quants[0])) ? 1.0 : 0.0;
        res.errors["global"]["ci"]["cov"] += inside;
        rCI.push_back(CI);
        relativeCI.push_back(CI / reality);
        // lrCI.push_back(w * CI);
        // lrelativeCI.push_back(w * CI / reality);
    }
    if (!quiet) {
        std::cout << std::endl;
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
    
    // std::cout << "Sum weights : " << sumw << std::endl;

    os.clear();
    os.str("");
    double acc;
    for(auto g_l : global_local) {
        os << g_l.second << std::endl;
        auto normalize = (g_l.first == "global") ? ntest : sumw;
        for(auto m: mean_median_ci_l) {
            if (m != "ci") {
                os << mean_median_ci[m] << std::endl;
                for (auto n : computed) {
                    res.errors[g_l.first][m][n] /= static_cast<double>(normalize);
                    os << fmt::format("{:>25} : {:<13}",n, res.errors[g_l.first][m][n]) << std::endl;
                }
            }
            else {
                if (g_l.first == "global") {
                    os << mean_median_ci[m] << std::endl;
                    for(auto c : ci) {
                        // normalize = ntest;                        
                        if (c.first == "cov") {
                            acc = res.errors[g_l.first][m][c.first]/static_cast<double>(normalize);
                        } else {
                            if(c.first == "meanCI") 
                                acc = ranges::accumulate(rCI,0.0)/static_cast<double>(normalize);
                            else if (c.first == "meanRCI")
                                acc = ranges::accumulate(relativeCI,0.0)/static_cast<double>(normalize);
                            else if (c.first == "medianCI")
                                acc = median(rCI);
                            else if (c.first == "medianRCI")
                                acc = median(relativeCI);
                        }
                        res.errors[g_l.first][m][c.first] = acc;
                        os << fmt::format("{:>25} : {:<13}",c.second, acc) << std::endl;

                    }

                }

            }
        }
        os << std::endl;
    }

    if (!quiet) {
        std::cout << os.str();
        std::cout.flush();
    }

    const std::string& teststats_filename = outfile + ".oobstats";
    std::ofstream teststats_file;
    if (!quiet) {
        teststats_file.open(teststats_filename, std::ios::out);
        if (!teststats_file.good()) {
            throw std::runtime_error("Could not write to oobstats file " + teststats_filename + ".");        
        }
        teststats_file << os.str();
        teststats_file.flush();
        teststats_file.close();
    }

    return res;
}