#include <fmt/format.h>

#include <cmath>

#include "EstimParam.hpp"
#include "ForestOnlineRegression.hpp"
#include "matutils.hpp"
#include "various.hpp"
#include "pls-eigen.hpp"
#include "parse_parexpr.hpp"
#include "forestQuantiles.hpp"
#include "threadpool.hpp"

#include "DataDense.hpp"
#include "cxxopts.hpp"
#include <algorithm>
#include <fstream>
#include "range/v3/all.hpp"
#ifdef PYTHON_OUTPUT
#include <pybind11/pybind11.h>
namespace py = pybind11;
#endif

using namespace ranger;
using namespace Eigen;
using namespace ranges;

template <class MatrixType>
EstimParamResults EstimParam_fun(Reftable<MatrixType> &myread,
                                 MatrixXd statobs,
                                 const cxxopts::ParseResult opts,
                                 bool quiet,
                                 bool weights)
{
    size_t nref, ntree, nthreads, noisecols, seed, minnodesize, ntest;
    std::string outfile, parameter_of_interest;
    double chosenscen, plsmaxvar;
    bool plsok, seeded;

    ntree = opts["t"].as<size_t>();
    nthreads = opts["j"].as<size_t>();
    noisecols = opts["c"].as<size_t>();
    seeded = opts.count("s") != 0;
    if (seeded)
        seed = opts["s"].as<size_t>();
    minnodesize = opts["m"].as<size_t>();
    ntest = opts["noob"].as<size_t>();
    plsmaxvar = opts["plsmaxvar"].as<double>();
    chosenscen = static_cast<double>(opts["chosenscen"].as<size_t>());
    parameter_of_interest = opts["parameter"].as<std::string>();
    plsok = opts.count("nolinear") == 0;

    outfile = (opts.count("output") == 0) ? "estimparam_out" : opts["o"].as<std::string>();

    double p_threshold_PLS = 0.99;
    std::vector<double> samplefract{std::min(1e5, static_cast<double>(myread.nrec)) / static_cast<double>(myread.nrec)};
    auto nstat = myread.stats_names.size();
    // MatrixXd statobs(1, nstat);
    MatrixXd emptyrow(1, 0);
    size_t num_samples = statobs.rows();

    // statobs = Map<MatrixXd>(origobs.data(), 1, nstat);

    std::size_t p1, p2;
    op_type op;
    parse_paramexpression(myread.params_names, parameter_of_interest, op, p1, p2);

    auto nparam = myread.params_names.size();

    nref = myread.nrec;
    ntest = std::min(nref, ntest);

    VectorXd y(nref);
    switch (op)
    {
    case op_type::none:
        y = myread.params(all, p1);
        break;
    case op_type::divide:
        y = myread.params(all, p1).array() / myread.params(all, p2).array();
        break;
    case op_type::multiply:
        y = myread.params(all, p1) * myread.params(all, p2);
        break;
    }

    // myread.params = std::move(myread.params(indexesModel,param_num)).eval();
    if (y.array().isNaN().any())
    {
        std::cout << "Error : there is some nan in the parameter data." << std::endl;
        exit(1);
    }

    EstimParamResults res;
    MatrixXd data_extended(nref, 0);

    if (plsok)
    {
        size_t ncomp_total = static_cast<size_t>(lround(1.0 * static_cast<double>(nstat)));
        MatrixXd Projection;
        RowVectorXd mean, std;
        size_t nComposante_sel;

        const std::string &pls_filename = outfile + ".plsvar";
        std::ofstream pls_file;
        if (!quiet)
            pls_file.open(pls_filename, std::ios::out);

        auto validvars = filterConstantVars(myread.stats);
        if (plsmaxvar == 0.0)
        {
            std::cout << "Using elbow method for selecting PLS axes" << std::endl;
            VectorXd percentYvar = pls(myread.stats(all,validvars),
                                       y,
                                       (ncomp_total - (nstat - validvars.size())), Projection, mean, std, true);
            nComposante_sel = percentYvar.size();
            for (auto i = 0; i < nComposante_sel; i++)
                res.plsvar.push_back(percentYvar(i));
        }
        else
        {
            VectorXd percentYvar = pls(myread.stats(all,validvars),
                                       y,
                                       (ncomp_total - (nstat - validvars.size())), Projection, mean, std, false);
            auto maxres = percentYvar[percentYvar.size() - 1];
            for (auto i = 0; i < percentYvar.size(); i++)
                if (percentYvar(i) <= plsmaxvar * maxres)
                    res.plsvar.push_back(percentYvar(i));
            nComposante_sel = res.plsvar.size();
        }

        //res.plsvar = std::vector<double>(percentYvar.size());
        for (auto p : res.plsvar)
            if (!quiet)
            {
                pls_file << p << std::endl;
                pls_file.close();
            }

        if (!quiet)
            std::cout << "Selecting only " << nComposante_sel << " pls components." << std::endl;

        double sumPlsweights = Projection.col(0).array().abs().sum();
        auto weightedPlsfirst = Projection.col(0) / sumPlsweights;

        const std::string &plsweights_filename = outfile + ".plsweights";
        std::ofstream plsweights_file;
        if (!quiet)
            plsweights_file.open(plsweights_filename, std::ios::out);
        for (auto &p : views::zip(myread.stats_names, weightedPlsfirst) | to_vector | actions::sort([](auto &a, auto &b) { return std::abs(a.second) > std::abs(b.second); }))
        {
            if (!quiet)
                plsweights_file << p.first << " " << p.second << std::endl;
            res.plsweights.push_back(p);
        }

        plsweights_file.close();

        auto Xc = (myread.stats(all,validvars).array().rowwise() - mean.array()).rowwise() / std.array();
        addCols(data_extended, (Xc.matrix() * Projection).leftCols(nComposante_sel));
        auto Xcobs = (statobs(all,validvars).array().rowwise() - mean.array()).rowwise() / std.array();
        addCols(statobs, (Xcobs.matrix() * Projection).leftCols(nComposante_sel));
        for (auto i = 0; i < nComposante_sel; i++)
            myread.stats_names.push_back("Comp " + std::to_string(i + 1));
    }

    if (!quiet)
    {
        const std::string &settings_filename = outfile + ".settings";
        std::ofstream settings_file;
        settings_file.open(settings_filename, std::ios::out);

        settings_file << "Parameter estimation analyses proceeded using: " << std::endl;
        settings_file << "- "
                      << "Parameter name: " << parameter_of_interest << std::endl;
        settings_file << "- "
                      << "Scenario " << chosenscen << std::endl;
        settings_file << "- " << myread.nrec << " simulated datasets" << std::endl;
        settings_file << "- " << ntree << " trees" << std::endl;
        settings_file << "- "
                      << "Minimum node size of " << (minnodesize == 0 ? 5 : minnodesize) << std::endl;
        settings_file << "- " << myread.stats.cols() << " summary statistics" << std::endl;
        if (plsok)
        {
            settings_file << "- " << data_extended.cols() << " axes of summary statistics PLS linear combination" << std::endl;
        }
        settings_file << "- " << noisecols << " noise variables" << std::endl;
        settings_file << "- " << ntest << " out-of-band samples used as test" << std::endl;
        settings_file.close();
    }

    addNoise(myread, data_extended, statobs, noisecols);
    std::vector<string> varwithouty = myread.stats_names;
    auto datastatobs = unique_cast<DataDense<MatrixXd>, Data>(std::make_unique<DataDense<MatrixXd>>(statobs, emptyrow, varwithouty, num_samples, varwithouty.size()));
    addCols(data_extended, y);
    myread.stats_names.push_back("Y");

    auto datastats = unique_cast<DataDense<MatrixType>, Data>(std::make_unique<DataDense<MatrixType>>(myread.stats, data_extended, myread.stats_names, nref, myread.stats_names.size()));

    ForestOnlineRegression forestreg;
    forestreg.init("Y",                                                                                 // dependant variable
                   MemoryMode::MEM_DOUBLE,                                                              // memory mode double or float
                   std::move(datastats),                                                                // data
                   std::move(datastatobs),                                                              // predict
                   std::max(std::floor(static_cast<double>(myread.stats_names.size() - 1) / 3.0), 1.0), // mtry, if 0 sqrt(m -1) but should be m/3 in regression
                   outfile,                                                                             // output file name prefix
                   ntree,                                                                               // number of trees
                   (seeded ? seed : r()),                                                               // seed rd()
                   nthreads,                                                                            // number of threads
                   ImportanceMode::IMP_GINI,                                                            // Default IMP_NONE
                   minnodesize,                                                                         // default min node size (classif = 1, regression 5)
                   "",                                                                                  // status variable name, only for survival
                   false,                                                                               // prediction mode (true = predict)
                   true,                                                                                // replace
                   std::vector<string>(0),                                                              // unordered variables names
                   false,                                                                               // memory_saving_splitting
                   DEFAULT_SPLITRULE,                                                                   // gini for classif variance for  regression
                   true,                                                                                // predict_all
                   samplefract,                                                                         // sample_fraction 1 if replace else 0.632
                   DEFAULT_ALPHA,                                                                       // alpha
                   DEFAULT_MINPROP,                                                                     // miniprop
                   false,                                                                               // holdout
                   DEFAULT_PREDICTIONTYPE,                                                              // prediction type
                   DEFAULT_NUM_RANDOM_SPLITS,                                                           // num_random_splits
                   false,                                                                               //order_snps
                   DEFAULT_MAXDEPTH,
                   ntest); // max_depth
    if (!quiet)
        forestreg.verbose_out = &std::cout;
    forestreg.run(!quiet, true);
    auto preds = forestreg.getPredictions();
    // Variable Importance
    res.variable_importance = forestreg.getImportance();
    if (!quiet)
        forestreg.writeImportanceFile();
    // OOB error by number of trees
    res.ntree_oob_error = preds[2][0];
    // res.oob_error = preds[2][0][ntree-1];
    if (!quiet)
        forestreg.writeOOBErrorFile();
    // Values/weights
    res.values_weights = forestreg.getWeights();
    if (!quiet)
        forestreg.writeWeightsFile();

    //    auto dataptr2 = forestreg.releaseData();
    //    auto& datareleased2 = static_cast<DataDense<MatrixType>&>(*dataptr2.get());
    //    datareleased2.data.conservativeResize(NoChange,nstat);
    //    myread.stats = std::move(datareleased2.data);
    myread.stats_names.resize(nstat);

    std::vector<double> probs{0.05, 0.5, 0.95};

    std::vector<double> rCI, relativeCI;
    std::ostringstream os;

    std::vector<double> obs;
    std::copy(y.begin(), y.end(), std::back_inserter(obs));

    if (weights)
        res.oob_weights = MatrixXd(ntest, nref);
    res.oob_map = forestreg.oob_subset;

    std::vector<double> expectation(num_samples, 0.0);
    std::vector<double> variance(num_samples, 0.0);
    std::map<std::string, std::string> global_local{
        {"global", "Global (prior) errors"},
        {"local", "Local (posterior) errors"}};

    std::map<std::string, std::string> mean_median_ci{
        {"mean", "Computed from the mean taken as point estimate"},
        {"median", "Computed from the median taken as point estimate"},
        {"ci", "Confidence interval measures"}};

    std::vector<std::string> mean_median_ci_l{"mean", "median", "ci"};

    std::vector<std::string> computed{"NMAE", "MSE", "NMSE"};

    std::map<std::string, std::string> ci{
        {"cov", "90% coverage"},
        {"meanCI", "Mean 90% CI"},
        {"meanRCI", "Mean relative 90% CI"},
        {"medianCI", "Median 90% CI"},
        {"medianRCI", "Median relative 90% CI"}};

    res.errors["local"] = std::vector<std::map<std::string, // Mean/Median/CI
                                               std::map<std::string, double>>>(num_samples);
    res.errors["global"] = std::vector<std::map<std::string, // Mean/Median/CI
                                                std::map<std::string, double>>>(1);

    for (auto g_l : global_local)
    {
        for (auto c : computed)
        {
            for (size_t j = 0; j < num_samples; j++)
            {
                if (g_l.first == "local" || j == 0)
                {
                    res.errors[g_l.first][j]["mean"][c] = 0.0;
                    res.errors[g_l.first][j]["median"][c] = 0.0;
                }
            }
        }
        if (g_l.first == "global")
            for (auto c : ci)
            {
                res.errors[g_l.first][0]["ci"][c.first] = 0.0;
            }
    }

    // double variance2 = 0.0;
    std::vector<double> sumw(num_samples, 0.0);
    std::vector<std::vector<double>> quants = forestQuantiles_b(obs, preds[4], probs);

    std::vector<std::vector<double>> quants_w = forestQuantiles_b(obs, preds[5], probs);
    // std::mutex mutex_quant;

    for (auto i = 0; i < nref; i++)
    {
        auto pred_oob = preds[0][0][i];
        auto reality = y(i);
        double diff = reality - pred_oob;
        auto sqdiff = diff * diff;
        // ThreadPool::ParallelFor<size_t>(0, nref,[&](auto i){
        if (!std::isnan(pred_oob))
        {
            res.errors["global"][0]["mean"]["NMAE"] += std::abs(diff / reality);
            res.errors["global"][0]["mean"]["MSE"] += sqdiff;
            res.errors["global"][0]["mean"]["NMSE"] += sqdiff / reality;
        }
        for (auto j = 0; j < num_samples; j++)
        {
            auto w = preds[4][j][i];
            // mutex_quant.lock();
            expectation[j] += w * reality;
            if (!std::isnan(pred_oob))
            {
                res.errors["local"][j]["mean"]["NMAE"] += w * std::abs(diff / reality);
                res.errors["local"][j]["mean"]["MSE"] += w * sqdiff;
                res.errors["local"][j]["mean"]["NMSE"] += w * sqdiff / reality;
            }
        }
        // mutex_quant.unlock();
        if (i < ntest)
        {
            auto p = *(std::next(forestreg.oob_subset.begin(), i));
            if (weights)
                for (auto i = 0; i < nref; i++)
                    res.oob_weights(p.second, i) = preds[5][p.second][i];
            // std::vector<double> quants_oob = forestQuantiles(obs,preds[5][p.second],probs);
            std::vector<double> quants_oob = quants_w[p.second];
            auto reality = y(p.first);
            auto diff = quants_oob[1] - reality;
            auto sqdiff = diff * diff;
            auto CI = quants_oob[2] - quants_oob[0];
            res.errors["global"][0]["median"]["NMAE"] += std::abs(diff / reality);
            res.errors["global"][0]["median"]["MSE"] += sqdiff;
            res.errors["global"][0]["median"]["NMSE"] += sqdiff / reality;
            double inside = ((reality <= quants_oob[2]) && (reality >= quants_oob[0])) ? 1.0 : 0.0;
            res.errors["global"][0]["ci"]["cov"] += inside;
            rCI.push_back(CI);
            relativeCI.push_back(CI / reality);
            for (auto j = 0; j < num_samples; j++)
            {
                auto w = preds[4][j][p.first];
                sumw[j] += w;
                res.errors["local"][j]["median"]["NMAE"] += w * std::abs(diff / reality);
                res.errors["local"][j]["median"]["MSE"] += w * sqdiff;
                res.errors["local"][j]["median"]["NMSE"] += w * sqdiff / reality;
            }
            // mutex_quant.lock();
            // mutex_quant.unlock();
        }
    }
    // });

    if (sumw[0] == 0.0 && ntest != 0 && !quiet)
        std::cout << "Warning : not enough oob samples to have local errors with median as point estimates" << std::endl;
    res.point_estimates = std::vector<std::map<string, double>>(num_samples);
    for (auto j = 0; j < num_samples; j++)
    {
        std::map<std::string, double> point_estimates{
            {"Expectation", expectation[j]},
            {"Median", quants[j][1]},
            {"Quantile_0.05", quants[j][0]},
            {"Quantile_0.95", quants[j][2]},
            {"Variance", res.errors["local"][j]["mean"]["MSE"]}};
        for (auto p : point_estimates)
            res.point_estimates[j].insert(p);
    }

    os << "Parameter estimation (point estimates)" << std::endl;
    if (num_samples > 1) os << fmt::format("{:>14}", "Target n°");
    for (auto p : res.point_estimates[0]) os << fmt::format("{:>14}", p.first);
    os << std::endl;
    for (auto j = 0; j < num_samples; j++)
    {
        if (num_samples > 1)
            os << fmt::format("{:>14}", j + 1);
        for (auto p : res.point_estimates[j])
            os << fmt::format(" {:>13.6g}", p.second);
        os << std::endl;
    }

    if (!quiet)
    {
        std::cout << std::endl;
        std::cout << os.str();
        std::cout.flush();
    }

    const std::string &predict_filename = outfile + ".predictions";
    std::ofstream predict_file;
    if (!quiet)
    {
        predict_file.open(predict_filename, std::ios::out);
        if (!predict_file.good())
        {
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
    for (auto g_l : global_local)
    {
        os << g_l.second;
        if (num_samples > 1 && g_l.first == "local") os << ", list of targets" << std::endl;
        size_t loop_j = g_l.first == "global" ? 1 : num_samples;
        for (size_t j = 0; j < loop_j; j++)
        {
            os << std::endl;
            if (g_l.first == "local" && num_samples > 1)
                os << "///////////// Target n° " << j + 1 << std::endl;
            for (auto m : mean_median_ci_l)
            {
                double normalize;
                if (g_l.first == "global")
                    normalize = (m != "mean") ? ntest : nref;
                else
                    normalize = (m == "median") ? sumw[j] : 1;
                if (m != "ci")
                {
                    os << mean_median_ci[m] << std::endl;
                    for (auto n : computed)
                    {
                        res.errors[g_l.first][j][m][n] /= static_cast<double>(normalize);
                        os << fmt::format("{:>25} : {:<13}", n, res.errors[g_l.first][j][m][n]) << std::endl;
                    }
                }
                else
                {
                    if (g_l.first == "global")
                    {
                        os << mean_median_ci[m] << std::endl;
                        for (auto c : ci)
                        {
                            if (c.first == "cov")
                            {
                                acc = res.errors[g_l.first][j][m][c.first] / static_cast<double>(normalize);
                            }
                            else
                            {
                                if (c.first == "meanCI")
                                    acc = ranges::accumulate(rCI, 0.0) / static_cast<double>(normalize);
                                else if (c.first == "meanRCI")
                                    acc = ranges::accumulate(relativeCI, 0.0) / static_cast<double>(normalize);
                                else if (c.first == "medianCI")
                                    acc = median(rCI);
                                else if (c.first == "medianRCI")
                                    acc = median(relativeCI);
                            }
                            res.errors[g_l.first][j][m][c.first] = acc;
                            os << fmt::format("{:>25} : {:<13}", c.second, acc) << std::endl;
                        }
                    }
                }
            }
        }
        os << std::endl;
    }
    if (!quiet)
    {
        std::cout << os.str();
        std::cout.flush();
    }

    const std::string &teststats_filename = outfile + ".oobstats";
    std::ofstream teststats_file;
    if (!quiet)
    {
        teststats_file.open(teststats_filename, std::ios::out);
        if (!teststats_file.good())
        {
            throw std::runtime_error("Could not write to oobstats file " + teststats_filename + ".");
        }
        teststats_file << os.str();
        teststats_file.flush();
        teststats_file.close();
    }

    // Pour Arnaud :
    // Mean_point_estimate    Median_point_estimate    5%_quantile        95%_quantile        global_prior_NMAE_from_Mean        global_prior_NMAE_from_Median        local_posterior_NMAE_from_Mean       local_posterior_NMAE_from_ Median     90%_COVERAGE
    // std::ofstream mer_file;
    // if (!quiet) {
    //     const std::string& mer_filename = outfile + ".revision_MER";

    //     mer_file.open(mer_filename, std::ios::out);
    //     if (!mer_file.good()) {
    //         throw std::runtime_error("Could not write to MER file: " + mer_filename + ".");
    //     }
    //     mer_file << fmt::format("{:>33}","Mean_point_estimate");
    //     mer_file << fmt::format("{:>33}","Median_point_estimate");
    //     mer_file << fmt::format("{:>33}","5%_quantile");
    //     mer_file << fmt::format("{:>33}","95%_quantile");
    //     mer_file << fmt::format("{:>33}","global_prior_NMAE_from_Mean");
    //     mer_file << fmt::format("{:>33}","global_prior_NMAE_from_Median");
    //     mer_file << fmt::format("{:>33}","local_posterior_NMAE_from_Mean");
    //     mer_file << fmt::format("{:>33}","local_posterior_NMAE_from_Median");
    //     mer_file << fmt::format("{:>33}","90%_COVERAGE");
    //     mer_file << endl;
    //     mer_file << fmt::format("{:>33.6f}",expectation);
    //     mer_file << fmt::format("{:>33.6f}",quants[1]);
    //     mer_file << fmt::format("{:>33.6f}",quants[0]);
    //     mer_file << fmt::format("{:>33.6f}",quants[2]);
    //     mer_file << fmt::format("{:>33.6f}",res.errors["global"]["mean"]["NMAE"]);
    //     mer_file << fmt::format("{:>33.6f}",res.errors["global"]["median"]["NMAE"]);
    //     mer_file << fmt::format("{:>33.6f}",res.errors["local"]["mean"]["NMAE"]);
    //     mer_file << fmt::format("{:>33.6f}",res.errors["local"]["median"]["NMAE"]);
    //     mer_file << fmt::format("{:>33.6f}",res.errors["global"]["ci"]["cov"]);
    //     mer_file << endl;
    // }

    return res;
}

template <class MatrixType>
EstimParamResults EstimParam_fun(Reftable<MatrixType> &myread,
                                 std::vector<double> origobs,
                                 const cxxopts::ParseResult opts,
                                 bool quiet,
                                 bool weights)
{
    auto nstat = myread.stats_names.size();
    MatrixXd statobs(1, nstat);
    statobs = Map<MatrixXd>(origobs.data(), 1, nstat);
    return EstimParam_fun(myread,statobs,opts,quiet,weights);
}

template EstimParamResults EstimParam_fun(Reftable<MatrixXd> &myread,
                                          MatrixXd origobs,
                                          const cxxopts::ParseResult opts,
                                          bool quiet,
                                          bool weights);

template EstimParamResults EstimParam_fun(Reftable<Eigen::Ref<MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>> &myread,
                                          MatrixXd origobs,
                                          const cxxopts::ParseResult opts,
                                          bool quiet,
                                          bool weights);


template EstimParamResults EstimParam_fun(Reftable<MatrixXd> &myread,
                                          std::vector<double> origobs,
                                          const cxxopts::ParseResult opts,
                                          bool quiet,
                                          bool weights);

template EstimParamResults EstimParam_fun(Reftable<Eigen::Ref<MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>> &myread,
                                          std::vector<double> origobs,
                                          const cxxopts::ParseResult opts,
                                          bool quiet,
                                          bool weights);