#pragma once
#include "Reftable.hpp"

Reftable<MatrixXd> readreftable(std::string headerpath, std::string reftablepath, size_t N = 0, bool quiet = false, std::string groups_opt = "");

Reftable<MatrixXd> readreftable_scen(std::string headerpath, std::string reftablepath, size_t sel_scen, size_t N = 0,  bool quiet = false);