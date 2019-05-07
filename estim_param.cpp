#include <fmt/format.h>


// using namespace boost::accumulators;
// // typedef accumulator_set<double, stats<tag::weighted_tail_quantile<right> >, double> accumulator_t;

#include "ForestOnlineRegression.hpp"
#include "readstatobs.hpp"
#include "readreftable.hpp"
#include "matutils.hpp"
#include "various.hpp"
#include "DataDense.h"
#include "pls-eigen.hpp"
#include <range/v3/all.hpp>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>
#include <boost/accumulators/statistics/weighted_tail_quantile.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/weighted_mean.hpp>
#include <boost/accumulators/statistics/median.hpp>
#include <boost/accumulators/statistics/weighted_median.hpp>
// #include <boost/accumulators/statistics/weighted_extended_p_square.hpp>

#include "cxxopts.hpp"

namespace bacc = boost::accumulators;
using namespace ranger;
using namespace Eigen;
using namespace ranges;


//auto print = [](int i) { std::cout << i << ' '; };

int main(int argc, char* argv[])
{
    size_t nref, ntree, nthreads, noisecols, seed, minnodesize, ntrain, ntest;
    std::string headerfile,reftablefile,statobsfile,outfile, parameter_of_interest;
    double chosenscen;   
    bool plsok,seeded;

    try {
        cxxopts::Options options(argv[0], " - ABC Random Forest/Model parameter estimation command line options");

        options
            .positional_help("[optional args]")
            .show_positional_help();

        options.add_options()
            ("h,header","Header file",cxxopts::value<std::string>()->default_value("headerRF.txt"))
            ("r,reftable","Reftable file",cxxopts::value<std::string>()->default_value("reftableRF.bin"))
            ("b,statobs","Statobs file",cxxopts::value<std::string>()->default_value("statobsRF.txt"))
            ("o,output","Prefix output",cxxopts::value<std::string>()->default_value("estimparam_out"))
            ("n,nref","Number of samples, 0 means all",cxxopts::value<size_t>()->default_value("0"))
            ("m,minnodesize","Minimal node size. 0 means 1 for classification or 5 for regression",cxxopts::value<size_t>()->default_value("0"))
            ("t,ntree","Number of trees",cxxopts::value<size_t>()->default_value("500"))
            ("j,threads","Number of threads, 0 means all",cxxopts::value<size_t>()->default_value("0"))
            ("s,seed","Seed, 0 means generated",cxxopts::value<size_t>()->default_value("0"))
            ("c,noisecolumns","Number of noise columns",cxxopts::value<size_t>()->default_value("5"))
            ("p,pls","Enable PLS",cxxopts::value<bool>()->default_value("true"))
            ("chosenscen","Chosen scenario (mandatory)", cxxopts::value<size_t>())
            ("ntrain","number of training samples (mandatory)",cxxopts::value<size_t>())
            ("ntest","number of testing samples (mandatory)",cxxopts::value<size_t>())
            ("parameter","name of the parameter of interest (mandatory)",cxxopts::value<std::string>())
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
        if (result.count("ntrain") == 0 
            || result.count("ntest") == 0 
            || result.count("parameter") == 0
            || result.count("chosenscen") == 0) {
            std::cout << "Error : please provide ntrain, ntest, parameter and chosenscen arguments." << std::endl;
            exit(1);
        }
        ntrain = result["ntrain"].as<size_t>();
        ntest = result["ntest"].as<size_t>();
        chosenscen = static_cast<double>(result["chosenscen"].as<size_t>());
        parameter_of_interest = result["parameter"].as<std::string>();
        plsok = result["p"].as<bool>();

    } catch (const cxxopts::OptionException& e)
      {
        std::cout << "error parsing options: " << e.what() << std::endl;
        exit(1);
    } 



    // size_t nref = 3000;
    // const std::string& outfile = "estimparam_out";
    double p_threshold_PLS = 0.99;

    auto myread = readreftable(headerfile, reftablefile, nref);
    std::vector<double> samplefract{std::min(1e5,static_cast<double>(myread.nrec))/static_cast<double>(myread.nrec)};
    auto nstat = myread.stats_names.size();
    MatrixXd statobs(1, nstat);
    statobs = Map<MatrixXd>(readStatObs("statobsRF.txt").data(), 1, nstat);

    auto param_enum = myread.params_names | view::enumerate | to_vector;
    const auto& param_found = ranges::find_if(param_enum,
        [&parameter_of_interest](const auto& s) { return s.second == parameter_of_interest; });
    if (param_found == ranges::end(param_enum)) {
        std::cout << "Error : cannot find parameter <" << parameter_of_interest << ">." << std::endl;
        exit(1);
    }
    size_t param_num = param_found->first;

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
    myread.params = std::move(myread.params(indexesModel,param_num)).eval();
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
        for(auto& v: percentYvar.array()) pls_file << v << std::endl;
        pls_file.close();
        double p_var_PLS = percentYvar(percentYvar.rows()-1) * p_threshold_PLS;

        const auto& enum_p_var_PLS = percentYvar
                            | view::enumerate
                            | to_vector;  
        size_t nComposante_sel = 
            ranges::find_if(enum_p_var_PLS,
                            [&p_var_PLS](auto v) { return v.second > p_var_PLS; })->first;

        std::cout << "Selecting only " << nComposante_sel << " pls components." << std::endl;

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
    forestreg.writeImportanceFile();
    forestreg.writeOOBErrorFile();
    forestreg.writeWeightsFile();

    std::vector<double> probs{0.05,0.5,0.95};

    std::ostringstream os;
    // os << fmt::format("   real value");
    os << fmt::format("  expectation");
    os << fmt::format("     variance");
    for(auto prob : probs) os << fmt::format("  quant. {:>0.2f}",prob);
    os << std::endl;
    bacc::accumulator_set<double, bacc::stats<bacc::tag::mean, 
                                              bacc::tag::median(bacc::with_p_square_quantile)>>
            CIacc, CI_relatifacc;
    bacc::accumulator_set<double, bacc::stats<bacc::tag::mean>> 
        MSEacc,NMSEacc, NMAEacc;

    for(auto j = 0; j < ntest+1; j++) {
        // os << fmt::format("{:>13.3f}",(j == 0? NAN : ytest(j-1)));
        bacc::accumulator_set<double, bacc::stats<bacc::tag::weighted_tail_quantile<bacc::left> >, double> 
            accleft(bacc::left_tail_cache_size = ntrain);
        // bacc::accumulator_set<double, bacc::stats<bacc::tag::weighted_extended_p_square_quantile>, double >
        //     accp2(bacc::extended_p_square_probabilities = probs);
        double expectation = 0.0;
        double variance = 0.0;
        for(auto i = 0; i < ntrain; i++) {
            accleft(y(i), bacc::weight = preds[4][j][i]);
            expectation += preds[4][j][i] * y(i);
            variance += preds[4][j][i] * (y(i) - preds[0][0][i]) * (y(i) - preds[0][0][i]); 
        }
        expectation /= static_cast<double>(ntree);
        variance /= static_cast<double>(ntree);
        std::vector<double> quants = probs 
            | view::transform([&accleft](auto& prob){ 
                    return bacc::quantile(accleft, bacc::quantile_probability = prob);
                    }); 
        if (j == 0) {            
            os << fmt::format("{:>13.3f}{:>13.3f}",expectation,variance);
            for(auto quant : quants) os << fmt::format("{:>13.3f}",quant);
            os << std::endl;
        } else {
            auto reality = ytest(j-1);
            auto diff = expectation - reality;
            auto sqdiff = diff * diff;
            auto CI = quants[2] - quants[0];
            MSEacc(sqdiff);
            NMSEacc(sqdiff / reality);
            NMAEacc(diff / reality);
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
    os << fmt::format("{:>19} : {:<13}","MSE",bacc::mean(MSEacc)) << std::endl;
    os << fmt::format("{:>19} : {:<13}","NMSE",bacc::mean(NMSEacc)) << std::endl;
    os << fmt::format("{:>19} : {:<13}","MMAE",bacc::mean(NMAEacc)) << std::endl;
    os << fmt::format("{:>19} : {:<13}","mean CI",bacc::mean(CIacc)) << std::endl;
    os << fmt::format("{:>19} : {:<13}","mean relative CI",bacc::mean(CI_relatifacc)) << std::endl;
    os << fmt::format("{:>19} : {:<13}","median CI",bacc::mean(CIacc)) << std::endl;
    os << fmt::format("{:>19} : {:<13}","median relative CI",bacc::mean(CI_relatifacc)) << std::endl;

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

}