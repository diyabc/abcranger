#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "readreftable.hpp"
#include "statobsTest.hpp"
#include "readstatobs.hpp"
#include "threadpool.hpp"

#include <highfive/H5File.hpp>

#include "H5Cpp.h"    
#include <random>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>

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

template<class T, class U>
boost::test_tools::predicate_result
compare_lists( T l1b, T l1e, U l2b, U l2e )
{
    
    if( std::distance(l1b,l1e) != std::distance(l2b,l2e) ) {
        boost::test_tools::predicate_result res( false );

        res.message() << "Different sizes [" << std::distance(l1b,l1e)  << "!=" << std::distance(l2b,l2e) << "]";

        return res;
    } else {
        bool res = true;
        while(res && l1b != l1e) {
            res = res && equalifnan(*(l1b++),*(l2b++));
         }
        return res;
    }
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
        // BOOST_TEST( compare_lists(&p[i*ncol],&p[(i+1)*ncol],data[i].begin(),data[i].end() ));
        // BOOST_TEST( compare_lists(std::next(std::begin(p),i*ncol),
        //                           std::next(std::begin(p),(i+1)*ncol),
        //                           data[i].begin(),
        //                           data[i].end() ));
        BOOST_TEST( compare_lists(p.row(i).array().begin(),
                                  p.row(i).array().end(),
                                  data[i].begin(),
                                  data[i].end() ));
    }); 
}

BOOST_AUTO_TEST_CASE(ReadRefTableData) {
    auto myread = readreftable("headerRF.txt","reftableRF.bin");
    File file("reftable.h5", File::ReadOnly);
    test_against_field(file,"nrec",myread.nrec);
    test_against_field(file,"nrecscen",myread.nrecscen);
    test_against_field(file,"nparam",myread.nparam);
    test_against_field(file,"scenarios",myread.scenarios);
    test_random_lines(file,"stats",myread.stats);
    test_random_lines(file,"params",myread.params);
}

BOOST_AUTO_TEST_CASE(ReadStatobs, * boost::unit_test::tolerance(0.001)) {
    BOOST_TEST( statobs == readStatObs("statobsRF.txt"), boost::test_tools::per_element());
}