#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "parse_parexpr.hpp"

TEST_CASE("parse composite parameters") {
    std::vector<std::string> params{"ra","t"};
    std::string simpleone{"ra/t"};
    std::size_t p1,p2;
    op_type op;
    parse_paramexpression(params,simpleone, op, p1, p2);
    CHECK(p1 == 0);
    CHECK(p2 == 1);
    CHECK(op == op_type::divide);

    simpleone = "t";
    parse_paramexpression(params,simpleone, op, p1, p2);
    CHECK(p1 == 1);
    CHECK(op == op_type::none);
}