#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "readreftable.hpp"
#include "statobsTest.hpp"
#include "readstatobs.hpp"
#include "threadpool.hpp"
#include "floatvectormatcher.hpp"

#include <highfive/H5File.hpp>
#include <highfive/H5Easy.hpp>

#include "H5Cpp.h"    
#include <random>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>

#include <range/v3/all.hpp>
using namespace ranges;

std::vector<std::string> readcolnames(H5::DataSet& dataset, const std::string& attr_name) {
    H5::Attribute attr(dataset.openAttribute(attr_name.c_str()));
    hsize_t dim = 0;
    attr.getSpace().getSimpleExtentDims(&dim);
    vector<string> res(dim);
    char **rdata = new char*[dim];
    H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
    attr.read(str_type,(void*)rdata);
    for (auto iStr = 0; iStr < dim; iStr++) {
        res[iStr] = rdata[iStr];
        // delete[] rdata[iStr];
    }
    delete[] rdata;
    return res;
}

TEST_CASE("Read Column Names from h5 file") {
    auto myread = readreftable("headerRF.txt","reftableRF.bin");
    auto file = H5::H5File("reftable.h5", H5F_ACC_RDONLY);
    auto dataset_stats = file.openDataSet("/stats");
    CHECK_THAT(myread.stats_names, Catch::Equals(readcolnames(dataset_stats,"stats_names")));
    auto dataset_params = file.openDataSet("/params");
    CHECK_THAT(myread.params_names, Catch::Equals(readcolnames(dataset_params,"params_names")));
}


template<typename T>
void test_against_field(HighFive::File& file, std::string field, T& p)
{
    T op;
    HighFive::DataSet dataset = file.getDataSet(field);
    dataset.read(op);
    CHECK(op == p);
}

bool equalifnan(double const& t1, double const& t2) {
    return ((std::isnan(t1) && std::isnan(t2)) || t1 == t2);
}


using namespace HighFive;

void test_random_lines(HighFive::File& file, const string& dataname, MatrixXd p){
    DataSet dataset =
        file.getDataSet(dataname);
    std::vector<std::vector<double>> data;
    dataset.read(data);
    auto nrow = data.size();
    auto ncol = data[0].size();
    // BOOST_TEST(p.size() == ncol*nrow);
    size_t nloop = std::min((size_t) 200u,ncol);
    std::vector<size_t> indices(ncol);
    std::iota(std::begin(indices),std::end(indices),0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(std::begin(indices),std::end(indices),g);
    
    ThreadPool::ParallelFor((size_t) 0u, nloop, [&] (size_t j){
        auto i = indices[j];
        const auto& expected = Catch::Matchers::Approx<std::vector<double>,decltype(p.row(i))>(data[i]);
        CHECK_THAT( p.row(i), expected);
    }); 
}

TEST_CASE("Check various components of a reftable") {
    auto myread = readreftable("headerRF.txt","reftableRF.bin");
    File file("reftable.h5", File::ReadOnly);
    test_against_field(file,"nrec",myread.nrec);
    test_against_field(file,"nrecscen",myread.nrecscen);
    test_against_field(file,"nparam",myread.nparam);
    test_against_field(file,"scenarios",myread.scenarios);
    test_random_lines(file,"stats",myread.stats);
    test_random_lines(file,"params",myread.params);
}

TEST_CASE("Check selected scen read") {
    auto myread = readreftable_scen("headerRF.txt","reftableRF.bin",1);
    File file("reftable.h5", File::ReadOnly);
    MatrixXd statsH5 = H5Easy::load<MatrixXd>(file,"stats");
    MatrixXd paramsH5 = H5Easy::load<MatrixXd>(file,"params");
    std::vector<double> scenarios = H5Easy::load<std::vector<double>>(file,"scenarios");
    double chosenscen = 1.0;
    std::vector<size_t> indexesModel = scenarios 
    | view::enumerate
    | view::filter([chosenscen](const auto& a){ return a.second == chosenscen; })
    | view::keys;


    statsH5 = statsH5(indexesModel,all);
    for(auto i = 0; i < indexesModel.size(); i++) {
        const auto& expected = Catch::Matchers::Approx<decltype(statsH5.row(i)),decltype(statsH5.row(i))>(myread.stats.row(i));
        CHECK_THAT( statsH5.row(i), expected);
    }
    // DataSet statsds = file.getDataSet("stats");
    // DataSet paramsds = file.getDataSet("params");
    // std::vector<double> rawstats,rawparams;
    // statsds.get
}

TEST_CASE("Read statobs from txt") {
    const auto& readstatobs = readStatObs("statobsRF.txt");
    const auto expected = Catch::Matchers::Approx<std::vector<double>,decltype(readstatobs)>(statobs).epsilon(0.001);
    CHECK_THAT( readstatobs, expected );
}
