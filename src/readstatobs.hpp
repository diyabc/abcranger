#pragma once
#include <iostream>
#include <fstream>
#include <iterator>
#include <regex>
#include <algorithm>
#include <iomanip>
#include <map>

using namespace std;

vector<double> readStatObs(const std::string &path) {
    ifstream statobsStream(path,ios::in);
    statobsStream >> noskipws;
    const std::string sS(istream_iterator<char>{statobsStream}, {});
    const regex stat_re(R"#(\s(-?\d+\.\d+)\s)#");    
    sregex_token_iterator itstat(begin(sS), end(sS), stat_re, {1});
    vector<double> statobs;
    auto it = itstat;
    sregex_token_iterator endregexp;
    while (it != endregexp) {
        const string st(*it++);
        statobs.push_back(stod(st));
    }
    statobsStream.close();
    return statobs;
}