#pragma once

#include <Eigen/Dense>

#include "Data.h"

namespace ranger
{

template<class MatrixType = Eigen::MatrixXd>
class DataDense : public Data
{
  public:
    DataDense() = default;

    DataDense(MatrixType &data, Eigen::MatrixXd& data_extended, std::vector<std::string> variable_names, size_t num_rows,
              size_t num_cols);

    DataDense(const DataDense &) = delete;
    DataDense &operator=(const DataDense &) = delete;

    virtual ~DataDense() override = default;

    double get(size_t row, size_t col) const override
    {
        // Use permuted data for corrected impurity importance
        size_t col_permuted = col;
        if (col >= num_cols)
        {
            col = getUnpermutedVarID(col);
            row = getPermutedSampleID(row);
        }
        if (col >= data.cols()) 
            return data_extended.coeff(row,col - data.cols());
        else 
            return data.coeff(row, col);
    }

    void reserveMemory() override
    {
    }

    void set(size_t col, size_t row, double value, bool &error) override
    {
        if (col >= data.cols())
            data_extended.coeffRef(row, col-data.cols()) = value;
        else 
            data.coeffRef(row, col) = value;
    }

    // void filterRows(const std::vector<size_t>& f) {
    //     data = std::move(data(f,Eigen::all)).eval();
    //     data_extended = std::move(data(f,Eigen::all)).eval();
    //     num_rows = f.size();
    // }

  public:
    MatrixType data;
    Eigen::MatrixXd& data_extended;
};

} // namespace ranger