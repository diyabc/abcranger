#define BOOST_TEST_MODULE RegexTest
#include <boost/test/unit_test.hpp>

#if defined(_MSC_VER)
#include <boost/regex.hpp>
using namespace boost;
#else
#include <regex>
#endif
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

BOOST_AUTO_TEST_CASE( SimpleRegex ) {
    string reparamlistrestr = R"#(\nfoo\n((?:\w+\W[^\n]*\n)*\n\n))#";
    const regex reparamlist(reparamlistrestr);
    sregex_token_iterator endregexp;

    smatch base_match;
    BOOST_CHECK(regex_search(filestr,base_match,reparamlist));
    BOOST_CHECK_EQUAL(base_match.size(),2);
    const string paramlistmatch = base_match[1];

    smatch line_match;
    const regex reparam(R"#((\w+)\W+\w\W+\w\w\[([^,\]]+),([^,\]]+)[,\]][^\n]*\n)#");
    sregex_token_iterator reparamit(std::begin(paramlistmatch),std::end(paramlistmatch), reparam, {1});
    BOOST_CHECK_EQUAL(std::distance(reparamit,endregexp),22);
}
