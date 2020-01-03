#include "DataDense.hpp"

namespace ranger {

DataDense::DataDense(Eigen::MatrixXd& data, std::vector<std::string> variable_names, size_t num_rows,
    size_t num_cols) :
    data { } {
  this->data.swap(data);
  this->variable_names = variable_names;
  this->num_rows = num_rows;
  this->num_cols = num_cols;
  this->num_cols_no_snp = num_cols;
}

} // namespace ranger