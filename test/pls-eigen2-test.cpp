#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <range/v3/all.hpp>

constexpr std::size_t operator "" _z ( unsigned long long n )
    { return n; }

auto sz = 5_z;
static_assert( std::is_same< decltype( sz ), std::size_t >::value, "" );

using namespace ranges;

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include "pls-eigen2.hpp"
using std::vector;
using std::string;
using std::cerr;
using std::cout;
using std::endl;
using std::ifstream;
using std::stringstream;

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

    size_t ncomp = 6;

    PLS_Model plsm;
    plsm.initialize(X.cols(),1,ncomp);
    plsm.plsr(X,Y,KERNEL_TYPE1);

    cout << plsm.loo_validation(X, Y, RMSEP) << endl;
    cout << plsm.rmsep(X, Y, 2)[0] << endl;
    VectorXd exp(6);
    exp << 0.3054273,
           0.7979361,
           0.9773195,
           0.9826665,
           0.9867306,
           0.9890077;
    // CHECK((res - exp).lpNorm<Infinity>() == Approx(0.0).margin(1e-7));
}