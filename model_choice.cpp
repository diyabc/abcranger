#include "ForestOnlineClassification.hpp"
#include "readstatobs.hpp"
#include "readreftable.hpp"
#include "matutils.hpp"
#include "DataDense.h"
using namespace ranger;
using namespace Eigen;

std::vector<double> DEFAULT_SAMPLE_FRACTION = std::vector<double>({1});

template <class T_SRC, class T_DEST>
std::unique_ptr<T_DEST> unique_cast(std::unique_ptr<T_SRC> &&src)
{
    if (!src)
        return std::unique_ptr<T_DEST>();

    // Throws a std::bad_cast() if this doesn't work out
    T_DEST *dest_ptr = &dynamic_cast<T_DEST &>(*src.get());

    src.release();
    std::unique_ptr<T_DEST> ret(dest_ptr);
    return ret;
}

int main() {
    auto myread = readreftable("headerRF.txt", "reftableRF.bin", 5000);
    auto nstat = myread.stats_names.size();
    size_t nref = 3000;
    size_t noisecols = 5;
    size_t K = myread.nrecscen.size();
    MatrixXd statobs(1,nstat);
    statobs = Map<MatrixXd>(readStatObs("statobsRF.txt").data(),1,nstat);
    addLda(myread, statobs);
    addNoise(myread, statobs, noisecols);
    auto datastatobs = unique_cast<DataDense, Data>(std::make_unique<DataDense>(statobs, myread.stats_names, 1, myread.stats_names.size()));
    addScen(myread);
    auto datastats = unique_cast<DataDense, Data>(std::make_unique<DataDense>(myread.stats, myread.stats_names, myread.nrec, myread.stats_names.size()));
    // ForestOnlineClassification forestclass;
    ForestOnlineClassification forestclass;
    auto ntree = 1000;
    auto nthreads = 8;
    std::cout << "OK !" << std::endl;
}