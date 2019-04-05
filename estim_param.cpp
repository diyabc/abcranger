#include "ForestOnlineRegression.hpp"
#include "readstatobs.hpp"
#include "readreftable.hpp"
#include "matutils.hpp"
#include "various.hpp"
#include "DataDense.h"
#include <range/v3/all.hpp>

using namespace ranger;
using namespace Eigen;
using namespace ranges;

auto print = [](int i) { std::cout << i << ' '; };

int main()
{
    // size_t nref = 3000;
    std::default_random_engine gen;

    size_t ncores = 8;
    auto myread = readreftable("headerRF.txt", "reftableRF.bin");
    auto nstat = myread.stats_names.size();
    MatrixXd statobs(1, nstat);
    statobs = Map<MatrixXd>(readStatObs("statobsRF.txt").data(), 1, nstat);

    std::string parameter_of_interest = "ra";


    size_t noisecols = 5;
    size_t K = myread.nrecscen.size();
    auto nparam = myread.params_names.size();

    const double chosenscen = 1.0;
    std::vector<size_t> indexesModel = myread.scenarios 
        | view::enumerate
        | view::filter([chosenscen](const auto& a){ return a.second == chosenscen; })
        | view::keys;
    
    myread.stats = myread.stats(indexesModel,all);
    myread.params = myread.params(indexesModel,all);
    size_t ntrain = 1000;
    size_t ntest = 50;
    size_t n = indexesModel.size();
    size_t npossible_test = n - ntrain;
    auto tosplit = view::ints(static_cast<size_t>(0),n)
        | to_vector
        | action::shuffle(gen);
    auto indicesTrain = tosplit | view::take(ntrain);
    auto indicesTest  = tosplit | view::slice(ntrain,n);

    auto y = myread.params(indicesTrain,parameter_of_interest);
    auto ytest = myread.params(indicesTest,parameter_of_interest);

    auto x = myread.stats(indicesTrain,all);
    auto xtest = myread.stats(indicesTest,all);

    indicesTest = view::ints(static_cast<size_t>(0),n-ntrain)
        | view::sample(ntest,gen);

    addLda(myread, statobs);
    addNoise(myread, statobs, noisecols);
    addScen(myread);
    std::vector<string> varwithouty(myread.stats_names.size()-1);
    for(auto i = 0; i < varwithouty.size(); i++) varwithouty[i] = myread.stats_names[i];

    auto datastatobs = unique_cast<DataDense, Data>(std::make_unique<DataDense>(statobs, varwithouty, 1, varwithouty.size()));
    auto datastats = unique_cast<DataDense, Data>(std::make_unique<DataDense>(myread.stats, myread.stats_names, myread.nrec, myread.stats_names.size()));

    auto ntree = 1000;
    auto nthreads = 8;
}