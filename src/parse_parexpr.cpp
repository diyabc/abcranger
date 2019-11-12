#include <range/v3/all.hpp>
#include <iostream>
#include "parse_parexpr.hpp"

using namespace ranges;

// #if defined(_MSC_VER)
// #include <boost/regex.hpp>
// using namespace boost;
// #else
#include <regex>
using namespace std;
// #endif

size_t find_param(const std::vector<std::string>& params_str, const std::string& param) {
    const auto& param_found = ranges::find(params_str,param);
    if (param_found == ranges::end(params_str)) {
        std::cout << "Error : cannot find parameter <" << param <<">." << std::endl; 
        exit(1);
    } else {
        return ranges::distance(params_str.begin(),param_found);
    }
}

void parse_paramexpression(const std::vector<std::string>& params_str,
    const std::string& to_parse,
    op_type& op, 
    size_t& p1, 
    size_t& p2)
{
    op = op_type::none;
    const regex param_re(R"#(([^\/\*]+)|((\w+)([\/\*])(\w+)))#");
    smatch base_match;
    if (regex_match(to_parse,base_match,param_re)) {
        if (base_match.size() == 6) {
            if(base_match[1].str().empty()) {
                if (base_match[4].str()[0] == '*') 
                    op = op_type::multiply;
                else if(base_match[4].str()[0] == '/')
                    op = op_type::divide;
                else {
                    std::cout << "Wrong parameter composition : " << base_match[4].str() << std::endl;
                    exit(1);
                }
                p1 = find_param(params_str,base_match[3].str());
                p2 = find_param(params_str,base_match[5].str());

            } else {
                p1 = find_param(params_str,base_match[1].str());
            }
        } else {
            std::cout << "Parser error : " << to_parse << std::endl;
            exit(1);
        }
    }
    else {
        std::cout << "Error while parsing " << to_parse << std::endl;
        exit(1);
    }

}
