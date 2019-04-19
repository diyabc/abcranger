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
    const std::string& outfile = "onlineranger_out";
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
    myread.params = std::move(myread.params(indexesModel,all)).eval();

    size_t ntrain = 1000;
    size_t ntest = 50;
    size_t n = indexesModel.size();
    size_t npossible_test = n - ntrain;
    auto tosplit = view::ints(static_cast<size_t>(0),n)
        | to_vector
        | action::shuffle(gen);
    std::vector<size_t> indicesTrain = tosplit | view::take(ntrain);
    std::vector<size_t> indicesTest  = tosplit | view::slice(ntrain,n);

    MatrixXd y = myread.params(indicesTrain,param_num);
    MatrixXd ytest = myread.params(indicesTest,param_num);

    MatrixXd x = myread.stats(indicesTrain,all);
    MatrixXd xtest = myread.stats(indicesTest,all);

    indicesTest = view::ints(static_cast<size_t>(0),n-ntrain)
        | view::sample(ntest,gen);

    size_t ncomp_total = static_cast<size_t>(lround(1.0 * static_cast<double>(nstat)));
    MatrixXd Pls;
    VectorXd percentYvar = pls(myread.stats(indicesTrain,all),
                       myread.params(indicesTrain,param_num),
                       ncomp_total,0.99,Pls);

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

    std::cout << "Selecting only " << nComposante_sel << " pls components.";

    // for(auto& s: myread.stats_names) plsweights_file << fmt::format(" {:>12}",s);
    // plsweights_file << std::endl;
    double sumPlsweights = Pls.col(0).array().abs().sum();
    auto weightedPlsfirst = Pls.col(0)/sumPlsweights;


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
        myread.scenarios
    };
    
    addCols(x,Pls.leftCols(nComposante_sel));
    for(auto i = 0; i < nComposante_sel; i++)
        myread.stats_names.push_back("Comp " + std::to_string(i+1));

    addNoise(myreadTrain, statobs, noisecols);

    
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