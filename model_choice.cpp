#include "fmt/format.h"
#include "ForestOnlineClassification.hpp"
#include "ForestOnlineRegression.hpp"
#include "readstatobs.hpp"
#include "readreftable.hpp"
#include "matutils.hpp"
#include "various.hpp"

#include "DataDense.h"
#include "cxxopts.hpp"
#include <algorithm>

using namespace ranger;
using namespace Eigen;

int main(int argc, char* argv[])
{
    size_t nref, ntree, nthreads, noisecols, seed, minnodesize;
    std::string headerfile,reftablefile,statobsfile,outfile;   
    bool lda,seeded;

    try {
        cxxopts::Options options(argv[0], " - ABC Random Forest/Model choice command line options");

        options
            .positional_help("[optional args]")
            .show_positional_help();

        options.add_options()
            ("h,header","Header file",cxxopts::value<std::string>()->default_value("headerRF.txt"))
            ("r,reftable","Reftable file",cxxopts::value<std::string>()->default_value("reftableRF.bin"))
            ("b,statobs","Statobs file",cxxopts::value<std::string>()->default_value("statobsRF.txt"))
            ("o,output","Prefix output",cxxopts::value<std::string>()->default_value("modelchoice_out"))
            ("n,nref","Number of samples, 0 means all",cxxopts::value<size_t>()->default_value("0"))
            ("m,minnodesize","Minimal node size. 0 means 1 for classification or 5 for regression",cxxopts::value<size_t>()->default_value("0"))
            ("t,ntree","Number of trees",cxxopts::value<size_t>()->default_value("500"))
            ("j,threads","Number of threads, 0 means all",cxxopts::value<size_t>()->default_value("0"))
            ("s,seed","Seed, 0 means generated",cxxopts::value<size_t>()->default_value("0"))
            ("c,noisecolumns","Number of noise columns",cxxopts::value<size_t>()->default_value("5"))
            ("l,lda","Enable LDA",cxxopts::value<bool>()->default_value("true"))
            ("help", "Print help")
            ;
        auto result = options.parse(argc,argv);

        if (result.count("help")) {
          std::cout << options.help({"", "Group"}) << std::endl;
            exit(0);
        }

        nref = result["n"].as<size_t>();
        ntree = result["t"].as<size_t>();
        nthreads = result["j"].as<size_t>();
        noisecols = result["c"].as<size_t>();
        seeded = result.count("s") != 0;
        if (seeded)
            seed = result["s"].as<size_t>();
        minnodesize = result["m"].as<size_t>();
        headerfile = result["h"].as<std::string>();
        reftablefile = result["r"].as<std::string>();
        statobsfile = result["b"].as<std::string>();
        outfile = result["o"].as<std::string>();
        lda = result["l"].as<bool>();

    } catch (const cxxopts::OptionException& e)
      {
        std::cout << "error parsing options: " << e.what() << std::endl;
        exit(1);
    } 



    auto myread = readreftable(headerfile, reftablefile, nref);
    std::vector<double> samplefract{std::min(1e5,static_cast<double>(myread.nrec))/static_cast<double>(myread.nrec)};
    auto nstat = myread.stats_names.size();
    size_t K = myread.nrecscen.size();
    MatrixXd statobs(1, nstat);
    statobs = Map<MatrixXd>(readStatObs(statobsfile).data(), 1, nstat);

    if (lda) addLda(myread, statobs);

    addNoise(myread, statobs, noisecols);

    addScen(myread);
    std::vector<string> varwithouty(myread.stats_names.size()-1);
    for(auto i = 0; i < varwithouty.size(); i++) varwithouty[i] = myread.stats_names[i];

    auto datastatobs = unique_cast<DataDense, Data>(std::make_unique<DataDense>(statobs, varwithouty, 1, varwithouty.size()));
    auto datastats = unique_cast<DataDense, Data>(std::make_unique<DataDense>(myread.stats, myread.stats_names, myread.nrec, myread.stats_names.size()));

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

    forestclass.run(true, true);
    auto preds = forestclass.getPredictions();
    auto oob_prior_error = forestclass.getOverallPredictionError();
    forestclass.writeConfusionFile();
    forestclass.writeImportanceFile();
    forestclass.writeOOBErrorFile();
    vector<size_t> votes(K);
    for(auto& tree_pred : preds[1][0]) votes[static_cast<size_t>(tree_pred-1)]++;
    size_t predicted_model = std::distance(votes.begin(),std::max_element(votes.begin(),votes.end()));

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
                     outfile,              // output file name prefix
                     ntree,                     // number of trees
                     (seeded ? seed : r()),                    // seed rd()
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
                     samplefract,   // sample_fraction 1 if replace else 0.632
                     DEFAULT_ALPHA,             // alpha
                     DEFAULT_MINPROP,           // miniprop
                     false,                     // holdout
                     DEFAULT_PREDICTIONTYPE,    // prediction type
                     DEFAULT_NUM_RANDOM_SPLITS, // num_random_splits
                     false,                     //order_snps
                     DEFAULT_MAXDEPTH);   

    forestreg.run(true,true);
    auto predserr = forestreg.getPredictions();
    const std::string& predict_filename = outfile + ".predictions";

    std::ostringstream os;
    for(auto i = 0; i < votes.size(); i++) {
        os << fmt::format(" votes model{0}",i+1);
    }
    os << fmt::format(" selected model");
    os << fmt::format(" post proba\n");
    for(auto i = 0; i < votes.size(); i++) {
        os << fmt::format("{:>13}",votes[i]);
    }
    os << fmt::format("{:>15}", predicted_model + 1);
    os << fmt::format("{:11.3f}\n",predserr[1][0][0]);
    std::cout << os.str();
    std::cout.flush();

    std::ofstream predict_file;
    predict_file.open(predict_filename, std::ios::out);
    if (!predict_file.good()) {
        throw std::runtime_error("Could not write to prediction file: " + predict_filename + ".");
    }
    predict_file << os.str();
    predict_file.flush();
    predict_file.close();

}