#pragma once

#include <vector>
#include <string>


enum op_type
{
    none,
    divide,
    multiply
};

void parse_paramexpression(const std::vector<std::string>& params_str,
    const std::string& to_parse,
    op_type& op, 
    size_t& p1, 
    size_t& p2);