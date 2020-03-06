#include "DataDense.hpp"

namespace ranger {

template<class MatrixType>
DataDense<MatrixType>::DataDense(MatrixType& _data, Eigen::MatrixXd& _data_extended, std::vector<std::string> variable_names, size_t num_rows,
    size_t num_cols) :
    data(_data), data_extended(_data_extended) {
  this->variable_names = variable_names;
  this->num_rows = num_rows;
  this->num_cols = num_cols;
  this->num_cols_no_snp = num_cols;
}

template class DataDense<Eigen::MatrixXd>;
template class DataDense<Eigen::Ref<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>>;

} // namespace ranger