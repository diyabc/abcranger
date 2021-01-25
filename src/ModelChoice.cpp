#include <fmt/format.h>
#include "ModelChoice.hpp"
#include "ForestOnlineClassification.hpp"
#include "ForestOnlineRegression.hpp"
#include "matutils.hpp"
#include "various.hpp"

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

template<class MatrixType>
ModelChoiceResults ModelChoice_fun(Reftable<MatrixType> &myread,
                                   MatrixXd statobs,
                                   const cxxopts::ParseResult opts,
                                   bool quiet)
{
    size_t ntree, nthreads, noisecols, seed, minnodesize;
    std::string outfile;
    bool lda,seeded;

    ntree = opts["t"].as<size_t>();
    nthreads = opts["j"].as<size_t>();
    noisecols = opts["c"].as<size_t>();
    seeded = opts.count("s") != 0;
    if (seeded)
        seed = opts["s"].as<size_t>();
    minnodesize = opts["m"].as<size_t>();
    lda =  opts.count("nolinear") == 0;
    outfile = (opts.count("output") == 0) ? "modelchoice_out" : opts["o"].as<std::string>();


    std::vector<double> samplefract{std::min(1e5,static_cast<double>(myread.nrec))/static_cast<double>(myread.nrec)};
    auto nstat = myread.stats_names.size();
    size_t K = myread.nrecscen.size();
    MatrixXd emptyrow(1,0);
    size_t num_samples = statobs.rows();

    size_t n = myread.nrec;


    MatrixXd data_extended(n,0);

    if (!quiet) {
        const std::string& settings_filename = outfile + ".settings";
        std::ofstream settings_file;
        settings_file.open(settings_filename, std::ios::out);

        settings_file << "Model choice analyses proceeded using: " << std::endl;
        settings_file << "- " << myread.nrec << " simulated datasets" << std::endl;
        settings_file << "- " << ntree << " trees" << std::endl;
        settings_file << "- " << "Minimum node size of " << (minnodesize == 0 ? 1 : minnodesize) << std::endl;
        settings_file << "- " << myread.stats.cols() << " summary statistics" << std::endl;
        if (lda) {
            settings_file << "- " << (K - 1) << " axes of summary statistics LDA linear combination" << std::endl;
        }
        settings_file << "- " << noisecols << " noise variables" << std::endl;
        settings_file.close();
    }


    if (lda) {
        addLda(myread, data_extended, statobs);
        const std::string& lda_filename = outfile + ".lda";
        std::ofstream lda_file;
        if (!quiet) {
            MatrixXd ldastatobs = statobs(all,lastN(data_extended.cols()));
            MatrixXd toprint(n+ldastatobs.rows(),data_extended.cols());
            toprint << ldastatobs, data_extended;
            lda_file.open(lda_filename, std::ios::out);
            lda_file << "# First lines (" << ldastatobs.rows() << ") are observed data" << std::endl;
            lda_file << toprint << std::endl;
        }
        lda_file.close();
    }

    addNoise(myread, data_extended, statobs, noisecols);

    addScen(myread,data_extended);
    std::vector<string> varwithouty(myread.stats_names.size()-1);
    for(auto i = 0; i < varwithouty.size(); i++) varwithouty[i] = myread.stats_names[i];


    auto datastatobs = unique_cast<DataDense<MatrixXd>, Data>(std::make_unique<DataDense<MatrixXd>>(statobs, emptyrow, varwithouty, num_samples, varwithouty.size()));
    auto datastats = unique_cast<DataDense<MatrixType>, Data>(std::make_unique<DataDense<MatrixType>>(myread.stats, data_extended, myread.stats_names, myread.nrec, myread.stats_names.size()));
    ForestOnlineClassification forestclass;
    forestclass.init("Y",                       // dependant variable
                     MemoryMode::MEM_DOUBLE,    // memory mode double or float
                     std::move(datastats),      // data
                     std::move(datastatobs),    // predict
                     0,                         // mtry, if 0 sqrt(m -1) but should be m/3 in regression
                     outfile,        // output file name prefix
                     ntree,                     // number of trees
                     (seeded ? seed : r()),                    // seed rd()
                     nthreads,                  // number of threads
                     ranger::IMP_GINI,          // Default IMP_NONE
                     minnodesize,                         // default min node size (classif = 1, regression 5)
                     "",                        // status variable name, only for survival
                     false,                     // prediction mode (true = predict)
                     true,                      // replace
                     std::vector<string>(0),    // unordered variables names
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

    ModelChoiceResults res;
    if (!quiet) {
        forestclass.verbose_out = &std::cout;
        std::cout << "///////////////////////////////////////// First forest (training on ABC output)" << std::endl;
    }

    forestclass.run(!quiet, true);

    auto preds = forestclass.getPredictions();
    // Overall oob error
    res.oob_error = forestclass.getOverallPredictionError();
    // Confusion Matrix
    res.confusion_matrix = forestclass.getConfusion();
    if (!quiet) forestclass.writeConfusionFile();
    // Variable Importance
    res.variable_importance = forestclass.getImportance();
    if (!quiet) forestclass.writeImportanceFile();
    // OOB error by number of trees;
    res.ntree_oob_error =  preds[2][0];
    if (!quiet) forestclass.writeOOBErrorFile();

    size_t nobs = statobs.rows();
    res.votes = std::vector<std::vector<size_t>>(num_samples,std::vector<size_t>(K));
    res.predicted_model = std::vector<size_t>(num_samples);
    for(auto i = 0; i <  num_samples; i++) {
        for(auto& tree_pred : preds[1][i]) res.votes[i][static_cast<size_t>(tree_pred-1)]++;
        res.predicted_model[i] = std::distance(res.votes[i].begin(),std::max_element(res.votes[i].begin(),res.votes[i].end()));
    }

    size_t ycol = data_extended.cols() - 1;

    for(size_t i = 0; i < preds[0][0].size(); i++) 
        if (!std::isnan(preds[0][0][i]))
            data_extended(i,ycol) = preds[0][0][i] == myread.scenarios[i] ? 1.0 : 0.0;

    // bool machin = false;
    auto dataptr = forestclass.releaseData();
    auto& datareleased = static_cast<DataDense<MatrixType>&>(*dataptr.get());
    // size_t ycol = datareleased.getNumCols() - 1;
    
    // for(size_t i = 0; i < preds[0][0].size(); i++) {
    //     if (!std::isnan(preds[0][0][i]))
    //         datareleased.set(ycol,i,preds[0][0][i] == myread.scenarios[i] ? 1.0 : 0.0, machin);
    // }

    // std::vector<size_t> defined_preds = preds[0][0]
    //     | views::enumerate
    //     | views::filter([](auto d){ return !std::isnan(d.second); })
    //     | views::keys
    //     | to<std::vector>();
    // datareleased.filterRows(defined_preds);

    auto statobsreleased = forestclass.releasePred();
    ForestOnlineRegression forestreg;


    forestreg.init("Y",                       // dependant variable
                     MemoryMode::MEM_DOUBLE,    // memory mode double or float
                     std::move(dataptr),    // data
                     std::move(statobsreleased),  // predict
                     0,                         // mtry, if 0 sqrt(m -1) but should be m/3 in regression
                     outfile,              // output file name prefix
                     ntree,                     // number of trees
                     (seeded ? seed : r()),                    // seed rd()
                     nthreads,                  // number of threads
                     DEFAULT_IMPORTANCE_MODE,  // Default IMP_NONE
                     5,                         // default min node size (classif = 1, regression 5)
                     "",                        // status variable name, only for survival
                     false,                     // prediction mode (true = predict)
                     true,                      // replace
                     std::vector<string>(0),        // unordered variables names
                     false,                     // memory_saving_splitting
                     DEFAULT_SPLITRULE,         // gini for classif variance for  regression
                     false,                     // predict_all
                     samplefract,   // sample_fraction 1 if replace else 0.632
                     DEFAULT_ALPHA,             // alpha
                     DEFAULT_MINPROP,           // miniprop
                     false,                     // holdout
                     DEFAULT_PREDICTIONTYPE,    // prediction type
                     DEFAULT_NUM_RANDOM_SPLITS, // num_random_splits
                     false,                     //order_snps
                     DEFAULT_MAXDEPTH);   

    if (!quiet) forestreg.verbose_out = &std::cout;
    if (!quiet) {
        forestclass.verbose_out = &std::cout;
        std::cout << "///////////////////////////////////////// Second forest (training on error)" << std::endl;
    }

    forestreg.run(!quiet,true);

    // auto dataptr2 = forestreg.releaseData();
    // auto& datareleased2 = static_cast<DataDense&>(*dataptr2.get());
    // datareleased2.data.conservativeResize(NoChange,nstat);
    // myread.stats = std::move(datareleased2.data);
    myread.stats_names.resize(nstat);

    auto predserr = forestreg.getPredictions();
    res.post_proba = std::vector<double>(num_samples);
    for(auto i = 0; i < num_samples; i++) res.post_proba[i] = predserr[1][0][i];
    const std::string& predict_filename = outfile + ".predictions";

    std::ostringstream os;
    if (num_samples > 1) os << fmt::format("{:>14}", "Target n°");
    for(auto i = 0; i < K; i++) {
        os << fmt::format("{:>14}",fmt::format("votes model{0}",i+1));
    }
    os << fmt::format(" selected model");
    os << fmt::format("  post proba\n");
    for (auto j = 0; j < num_samples; j++) {
        if (num_samples > 1)
            os << fmt::format("{:>14}", j + 1);
        for(auto i = 0; i < K; i++) {
            os << fmt::format(" {:>13}",res.votes[j][i]);
        }
        os << fmt::format("{:>15}", res.predicted_model[j] + 1);
        os << fmt::format("{:12.3f}\n",res.post_proba[j]);
    }
    if (!quiet) std::cout << os.str();
    std::cout.flush();

    std::ofstream predict_file;
    if (!quiet) {
        predict_file.open(predict_filename, std::ios::out);
        if (!predict_file.good()) {
            throw std::runtime_error("Could not write to prediction file: " + predict_filename + ".");
        }
        predict_file << os.str();
        predict_file.flush();
        predict_file.close();
    }

    // Pour Arnaud
    // Global_error_rate    Local_error rate*    Vote_S1   Vote_S2    Vote_S2    Vote_S3    Vote_S5    Vote_S6   Posterior_probability_S3(best)
    // std::ofstream mer_file;
    // if (!quiet) {
    //     const std::string& mer_filename = outfile + ".revision_MER";

    //     mer_file.open(mer_filename, std::ios::out);
    //     if (!mer_file.good()) {
    //         throw std::runtime_error("Could not write to MER file: " + mer_filename + ".");
    //     }
    //     mer_file << fmt::format("Global_error_rate");
    //     mer_file << fmt::format(" Local_error_rate");            
    //     for(auto i = 0; i < votes.size(); i++) {
    //         mer_file << fmt::format("{:>13}",fmt::format("Vote_S{0}",i+1));
    //     }
    //     mer_file << fmt::format("{:>32}\n",fmt::format("Posterior_probability_S{0}(best)",predicted_model + 1));
    //     mer_file << fmt::format("{:17.3f}",res.ntree_oob_error[ntree-1]);
    //     mer_file << fmt::format("{:17.3f}",1-res.post_proba);
    //     for(auto i = 0; i < votes.size(); i++) {
    //         mer_file << fmt::format("{:>13}",votes[i]);
    //     }
    //     mer_file << fmt::format("{:32.3f}\n",res.post_proba);
    //     mer_file.flush();
    //     mer_file.close();
    // }


    return res;
}

template<class MatrixType>
ModelChoiceResults ModelChoice_fun(Reftable<MatrixType> &myread,
                                   std::vector<double> origobs,
                                   const cxxopts::ParseResult opts,
                                   bool quiet)
{
    auto nstat = myread.stats_names.size();
    MatrixXd statobs(1, nstat);
    statobs = Map<MatrixXd>(origobs.data(), 1, nstat);
    return ModelChoice_fun(myread,statobs,opts,quiet);
}

template 
ModelChoiceResults ModelChoice_fun(Reftable<MatrixXd> &myread,
                                   MatrixXd obs,
                                   const cxxopts::ParseResult opts,
                                   bool quiet);

template 
ModelChoiceResults ModelChoice_fun(Reftable<Eigen::Ref<MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>> &myread,
                                   MatrixXd obs,
                                   const cxxopts::ParseResult opts,
                                   bool quiet);

template 
ModelChoiceResults ModelChoice_fun(Reftable<MatrixXd> &myread,
                                   std::vector<double> obs,
                                   const cxxopts::ParseResult opts,
                                   bool quiet);

template 
ModelChoiceResults ModelChoice_fun(Reftable<Eigen::Ref<MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>> &myread,
                                   std::vector<double> obs,
                                   const cxxopts::ParseResult opts,
                                   bool quiet);
