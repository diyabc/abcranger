#include <vector>
#include <algorithm>
#include <random>
#include "readreftable.hpp"

template<class T>
void addCol(std::vector<T>& v, std::vector<T>& c) {
    auto nrow = c.size();
    auto vsize = v.size();
    v.resize(vsize+nrow);
    std::copy(std::begin(c),std::end(c),std::next(std::begin(v),vsize));
}

template<class T>
void replaceCol(std::vector<T>& v, std::vector<T>& c, size_t colnum) {
    auto nrow = c.size();
    std::copy(std::begin(c),std::end(c),std::next(std::begin(v),colnum*nrow));
}

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(0.0,1.0);

std::vector<double> getNoise(size_t n) {
    std::vector<double> v(n);
    std::generate(std::begin(v),std::end(v),[] () { return dis(gen); });
    return v;
}

void addNoiseCols(Reftable& rf, size_t ncols) {
    for(auto i = 0; i < ncols; i++) {
        addCol(rf.stats,getNoise(rf.nrec));
        rf.stats_names.push_back("NOISE" + std::to_string(i));
    }
}