#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>


#include "forestQuantiles.hpp"
#include "csv-eigen.hpp"
#include "range/v3/all.hpp"
#include "floatvectormatcher.hpp"

TEST_CASE("Meinsheuser's quantiles") {
    MatrixXd data = read_matrix_file("quantiles.csv",',');
    auto n = data.rows();
    std::vector<double> obs,weights;
    std::copy(data.col(0).begin(),data.col(0).end(),std::back_inserter(obs));
    std::copy(data.col(1).begin(),data.col(1).end(),std::back_inserter(weights));
    std::vector<double> quants = forestQuantiles(obs,weights,std::vector<double>{0.05,0.5,0.95});
    std::vector<double> values{0.06682934,0.1994667,0.8464229};
    for(auto i = 0; i < values.size(); i++)
        CHECK(quants[i] == Approx(values[i]).epsilon(1e-6));
}