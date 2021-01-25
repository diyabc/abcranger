#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "readstatobs.hpp"
#include "readreftable.hpp"
#include "csv-eigen.hpp"
#include "cxxopts.hpp"
#include "ModelChoice.hpp"
#include "ks.hpp"
#include "various.hpp"
#include "tqdm.hpp"

#include "range/v3/all.hpp"

using namespace ranges;

TEST_CASE("ModelChoice KS test")
{
    size_t nref = 3000;
    std::string headerfile,reftablefile,statobsfile;
    MatrixXd E = read_matrix_file("modelchoice_runs.csv",',');

    std::vector<double> postprobasR = E.col(0) | to_vector;

    try {
        std::vector<std::string> args{"ModelChoice","-t","500","-n","3000"};
        std::vector<char *> argv;
        for(std::string &s: args) argv.push_back(&s[0]);
        argv.push_back(NULL);

        cxxopts::Options options("", "");

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
            ("nolinear","Disable LDA",cxxopts::value<bool>()->default_value("false"))
            ("help", "Print help")
            ;
        int argc = argv.size()-1;
        char ** argvc = argv.data();
        const auto opts = options.parse(argc,argvc);

        size_t nrun = 100;
        std::vector<double> postprobas(nrun);
        headerfile = opts["h"].as<std::string>();
        reftablefile = opts["r"].as<std::string>();
        statobsfile = opts["b"].as<std::string>();
        auto myread = readreftable(headerfile, reftablefile, 3000,true);
        const auto statobs = readStatObs(statobsfile);
        tqdm bar;
        for(auto i = 0; i < nrun; i++) {
            bar.progress(i,nrun);

            auto res = ModelChoice_fun(myread,statobs,opts,true);
            postprobas[i] = res.post_proba[0];
        }
        std::cout << std::endl;
        double D, pvalue;

        std::cout << "postprobas" << std::endl;
        std::cout << (postprobas | views::all) << std::endl;

        D = KSTest(postprobasR,postprobas);
        pvalue = 1-psmirnov2x(D,E.rows(),nrun);

        CHECK( pvalue >= 0.05);

    } catch (const cxxopts::OptionException& e)
      {
        std::cout << "error parsing options: " << e.what() << std::endl;
        exit(1);
    } 

}