#define _HAS_ITERATOR_DEBUGGING 0
#define _REGEX_MAX_STACK_COUNT 5000
#include <catch2/catch_test_macros.hpp>

#include <regex>
#include <string>

using namespace std;

string filestr = R"#(
foo
NS1 N UN[1000,30000,15000,7500]
NS2 N UN[1000,30000,15000,7500]
NS3 N LU[10,1000,15000,7500]
NS4 N UN[1000,30000,15000,7500]
NS5 N UN[1000,30000,0.0,0.0]
t1 T UN[40,45,1500,750]
BD1 T UN[0,5,0.0,0.0]
NF1 N LU[2,1000,0.0,0.0]
tm1 T UN[0,3000,0.0,0.0]
t3 T UN[85,320,1500,750]
tm3 T UN[0,3000,0.0,0.0]
t4 T UN[68,73,1500,750]
BD4 T UN[0,5,0.0,0.0]
NF4 N LU[2,1000,0.0,0.0]
tm4 T UN[0,3000,0.0,0.0]
t5 T UN[60,65,0.0,0.0]
BD5 T UN[0,5,0.0,0.0]
NF5 N LU[2,1000,0.0,0.0]
tm5 T UN[0,3000,0.0,0.0]
texp T UN[1000,30000,0.0,0.0]
Nanc N LU[1000,30000,0.0,0.0]
ra4 A UN[0.1,0.9,0.0,0.0]


)#";

// string filestr = R"#(
// foo
// NS1 N UN[1000,30000,15000,7500]
// NS2 N UN[1000,30000,15000,7500]
// NS3 N LU[10,1000,15000,7500]
// NS4 N UN[1000,30000,15000,7500]
// NS5 N UN[1000,30000,0.0,0.0]
// t1 T UN[40,45,1500,750]
// BD1 T UN[0,5,0.0,0.0]cd 
// NF1 N LU[2,1000,0.0,0.0]
// tm1 T UN[0,3000,0.0,0.0]
// t3 T UN[85,320,1500,750]
// tm3 T UN[0,3000,0.0,0.0]
// t4 T UN[68,73,1500,750]
// BD4 T UN[0,5,0.0,0.0]
// NF4 N LU[2,1000,0.0,0.0]
// tm4 T UN[0,3000,0.0,0.0]
// t5 T UN[60,65,0.0,0.0]
// BD5 T UN[0,5,0.0,0.0]
// NF5 N LU[2,1000,0.0,0.0]
// tm5 T UN[0,3000,0.0,0.0]
// texp T UN[1000,30000,0.0,0.0]
// Nanc N LU[1000,30000,0.0,0.0]
// ra4 A UN[0.1,0.9,0.0,0.0]
// Nanc<NS1
// Nanc<NS2
// Nanc<NS4
// Nanc<NS5
// texp>tm1
// texp>tm3
// texp>tm4
// texp>tm5
// DRAW UNTIL


// )#";

TEST_CASE( "Regex test on diybabc typical header" ) {
    string reparamlistrestr = R"#(\nfoo\n((?:\w+\W[^\n]*\n){22}\n\n))#";
    const regex reparamlist(reparamlistrestr);
    sregex_token_iterator endregexp;

    smatch base_match;
    CHECK(regex_search(filestr,base_match,reparamlist));
    CHECK(base_match.size() == 2);
    const string paramlistmatch = base_match[1];

    smatch line_match;
    const regex reparam(R"#((\w+)\W+\w\W+\w\w\[([^,\]]+),([^,\]]+)[,\]][^\n]*\n)#");
    sregex_token_iterator reparamit(std::begin(paramlistmatch),std::end(paramlistmatch), reparam, {1});
    CHECK(std::distance(reparamit,endregexp) == 22);
}
