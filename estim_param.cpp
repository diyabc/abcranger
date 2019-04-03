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
    size_t nref = 3000;
    auto myread = readreftable("headerRF.txt", "reftableRF.bin", nref);
    std::string parameter_of_interest = "ra";
    auto nstat = myread.stats_names.size();
    auto nparam = myread.params_names.size();
    size_t noisecols = 5;
    size_t K = myread.nrecscen.size();
    MatrixXd statobs(1, nstat);
    statobs = Map<MatrixXd>(readStatObs("statobsRF.txt").data(), 1, nstat);

    const double chosenscen = 1.0;
    std::vector<size_t> indexesModel = myread.scenarios 
        | view::enumerate
        | view::filter([chosenscen](const auto& a){ return a.second == chosenscen; })
        | view::keys;
//    ranges::for_each(indexesModel, print);
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