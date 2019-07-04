#include "ModelChoice.hpp"
#include "readreftable.hpp"
#include "readstatobs.hpp"

int main(int argc, char* argv[])
{
    string headerfile, reftablefile, statobsfile;
    size_t nref;
    
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
            ("nolda","Disable LDA",cxxopts::value<bool>()->default_value("false"))
            ("help", "Print help")
            ;
        const auto opts = options.parse(argc,argv);

        headerfile = opts["h"].as<std::string>();
        reftablefile = opts["r"].as<std::string>();
        statobsfile = opts["b"].as<std::string>();
        nref = opts["n"].as<size_t>();

        auto myread = readreftable(headerfile, reftablefile, nref);
        const auto statobs = readStatObs(statobsfile);
        auto res = ModelChoice_fun(myread,statobs,opts);


        if (opts.count("help")) {
          std::cout << options.help({"", "Group"}) << std::endl;
            exit(0);
        }


    } catch (const cxxopts::OptionException& e)
      {
        std::cout << "error parsing options: " << e.what() << std::endl;
        exit(1);
    } 


}