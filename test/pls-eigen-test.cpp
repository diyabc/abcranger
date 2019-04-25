/**
 * @file pls-eigen-test.cpp
 * @author François-David Collin <Francois-David.Collin@umontpellier.fr>
 * @brief from Dayal and MacGregor (1997) 
 *        "Improved PLS Algorithms" J. of Chemometrics. 11,73-85.
 * @version 0.1
 * @date 2019-04-25
 * 
 * @copyright Copyright (c) 2019
 * 
 */
#include <catch2/catch.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include "pls-eigen.hpp"

void split(const string& s, char c, vector<string>& v) {
    string::size_type i = 0;
    string::size_type j = s.find(c);

    while (j != string::npos) {
        v.push_back(s.substr(i, j-i));
        i = ++j;
        j = s.find(c, j);
    }
    if (j == string::npos) v.push_back(s.substr(i, s.length( )));
}

inline double string2double(const std::string& s){ std::istringstream i(s); double x = 0; i >> x; return x; }

MatrixXd read_matrix_file(string filename, char sep) {
    cerr << "Loading " << filename << endl;
    ifstream myfile(filename.c_str());
    stringstream ss;

    vector<vector<double> > M;
    if (myfile.is_open()) {
        string line;

        while ( getline(myfile,line) ) {
            //split string based on "," and store results into vector
            vector<string> fields;
            split(line, sep, fields);

            vector<double>row(fields.size());
            for( int i=0; i < fields.size(); i++ ) {
                row[i] = string2double(fields[i]);
            }
            M.push_back(row);
        }
    }

    MatrixXd X( (int) M.size(), (int) M[0].size() );
    for(int i=0; i < M.size(); i++ ) {
        for(int j=0; j < M[i].size(); j++ ) {  
            X(i,j)=M[i][j]; 
        }
    }
    return X;
}

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