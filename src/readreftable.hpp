#pragma once
#include "Reftable.hpp"

Reftable<MatrixXd> readreftable(string headerpath, string reftablepath, size_t N = 0, bool quiet = false);

Reftable<MatrixXd> readreftable_scen(string headerpath, string reftablepath, size_t sel_scen, size_t N = 0,  bool quiet = false);