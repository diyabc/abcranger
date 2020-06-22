#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <random>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>

#define H5_USE_EIGEN
#include <highfive/H5Easy.hpp>
// #include <highfive/H5Attribute.hpp>
// #include <highfive/H5File.hpp>
// #include <highfive/H5DataSet.hpp>
// #include <highfive/H5DataSpace.hpp>
// #include <highfive/H5DataType.hpp>
// #include <highfive/H5Object.hpp>

#include "readreftable.hpp"
#include "statobsTest.hpp"
#include "readstatobs.hpp"
#include "threadpool.hpp"

#include "H5Cpp.h"

#include <range/v3/all.hpp>
#include "floatvectormatcher.hpp"

using namespace ranges;
using namespace HighFive;

std::vector<std::string> readcolnames(H5::DataSet &dataset, const std::string &attr_name)
{
    H5::Attribute attr(dataset.openAttribute(attr_name.c_str()));
    hsize_t dim = 0;
    attr.getSpace().getSimpleExtentDims(&dim);
    vector<string> res(dim);
    char **rdata = new char *[dim];
    H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
    attr.read(str_type, (void *)rdata);
    for (auto iStr = 0; iStr < dim; iStr++)
    {
        res[iStr] = rdata[iStr];
        free(rdata[iStr]);
    }
    free(rdata);
    return res;
}

TEST_CASE("Read Column Names from h5 file")
{
    auto myread = readreftable("headerRF.txt", "reftableRF.bin");
    auto file = H5::H5File("reftable.h5", H5F_ACC_RDONLY);
    auto dataset_stats = file.openDataSet("/stats");
    CHECK_THAT(myread.stats_names, Catch::Equals(readcolnames(dataset_stats, "stats_names")));
    auto dataset_params = file.openDataSet("/params");
    CHECK_THAT(myread.params_names, Catch::Equals(readcolnames(dataset_params, "params_names")));
}

template <typename T>
void test_against_field(HighFive::File &file, std::string field, T &p)
{
    T op;
    HighFive::DataSet dataset = file.getDataSet(field);
    dataset.read(op);
    CHECK(op == p);
}

bool equalifnan(double const &t1, double const &t2)
{
    return ((std::isnan(t1) && std::isnan(t2)) || t1 == t2);
}

using namespace HighFive;

void test_random_lines(HighFive::File &file, const string &dataname, const MatrixXd& p)
{
    DataSet dataset =
        file.getDataSet(dataname);
    std::vector<std::vector<double>> data;
    dataset.read(data);
    auto ncol = data[0].size();
    size_t nloop = std::min((size_t)200u, ncol);
    std::vector<size_t> indices(ncol);
    std::iota(std::begin(indices), std::end(indices), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(std::begin(indices), std::end(indices), g);

    ThreadPool::ParallelFor((size_t)0u, nloop, [&](size_t j) {
        auto i = indices[j];
        auto row = p.row(i);
        const auto &expected = Catch::Matchers::ApproxRng<decltype(row),std::vector<double>>(row);
        CHECK_THAT(data[i], expected);
    });
}

TEST_CASE("Check various components of a reftable")
{
    auto myread = readreftable("headerRF.txt", "reftableRF.bin");
    File file("reftable.h5", File::ReadOnly);
    test_against_field(file, "nrec", myread.nrec);
    test_against_field(file, "nrecscen", myread.nrecscen);
    test_against_field(file, "nparam", myread.nparam);
    test_against_field(file, "scenarios", myread.scenarios);
    test_random_lines(file, "stats", myread.stats);
    test_random_lines(file, "params", myread.params);
}

TEST_CASE("Check selected scen read")
{
    auto myread = readreftable_scen("headerRF.txt", "reftableRF.bin", 1);
    File file("reftable.h5", File::ReadOnly);
    MatrixXd statsH5 = H5Easy::load<MatrixXd>(file, "stats");
    MatrixXd paramsH5 = H5Easy::load<MatrixXd>(file, "params");
    std::vector<double> scenarios = H5Easy::load<std::vector<double>>(file, "scenarios");
    double chosenscen = 1.0;
    std::vector<size_t> indexesModel = 
        scenarios 
        | views::enumerate 
        | views::filter([chosenscen](const auto &a) { return a.second == chosenscen; }) 
        | views::keys 
        | to<std::vector>();

    CHECK(myread.stats.isApprox(statsH5(indexesModel, all)));
}

TEST_CASE("Read statobs from txt") {
    const auto readstatobs = readStatObs("statobsRF.txt");
    const auto expected = Catch::Matchers::Approx(readstatobs).epsilon(0.001);
    CHECK_THAT( statobs, expected );
}
