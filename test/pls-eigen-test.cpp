#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include "pls-eigen.hpp"
#include "csv-eigen.hpp"

TEST_CASE("PLS with eigen") {
    MatrixXd X = read_matrix_file("gasolineX.csv",',');
    VectorXd Y = read_matrix_file("gasolineY.csv",',');
    MatrixXd R = read_matrix_file("projection.csv", ',');
    MatrixXd S = read_matrix_file("scores.csv", ',');

    MatrixXd Projection;
    RowVectorXd mean,std;
    VectorXd res = pls(X,Y,6,Projection,mean,std);
    VectorXd exp(6);
    exp << 0.3054273,
           0.7979361,
           0.9773195,
           0.9826665,
           0.9867306,
           0.9890077;
    CHECK((res - exp).lpNorm<Infinity>() == Approx(0.0).margin(1e-7));
    CHECK((Projection-R).lpNorm<Infinity>() == Approx(0.0).margin(1e-10));
    MatrixXd Scores = ((X.array().rowwise()-mean.array()).rowwise()/std.array()).matrix() * Projection;
    CHECK((Scores - S).lpNorm<Infinity>() == Approx(0.0).margin(1e-10));
}

TEST_CASE("PLS with eigen and constant feature") {
    MatrixXd Xr = read_matrix_file("gasolineXC.csv",',');
    auto X = Xr(all,filterConstantVars(Xr));
    VectorXd Y = read_matrix_file("gasolineY.csv",',');
    MatrixXd R = read_matrix_file("projection.csv", ',');
    MatrixXd S = read_matrix_file("scores.csv", ',');

    MatrixXd Projection;
    RowVectorXd mean,std;
    VectorXd res = pls(X,Y,6,Projection,mean,std);
    VectorXd exp(6);
    exp << 0.3054273,
           0.7979361,
           0.9773195,
           0.9826665,
           0.9867306,
           0.9890077;
    CHECK((res - exp).lpNorm<Infinity>() == Approx(0.0).margin(1e-7));
    CHECK((Projection-R).lpNorm<Infinity>() == Approx(0.0).margin(1e-10));
    MatrixXd Scores = ((X.array().rowwise()-mean.array()).rowwise()/std.array()).matrix() * Projection;
    CHECK((Scores - S).lpNorm<Infinity>() == Approx(0.0).margin(1e-10));
}