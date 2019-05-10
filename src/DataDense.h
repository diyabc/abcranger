#pragma once

#include <Eigen/Dense>

#include "Data.h"

namespace ranger
{

class DataDense : public Data
{
  public:
    DataDense() = default;

    DataDense(Eigen::MatrixXd &data, std::vector<std::string> variable_names, size_t num_rows,
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
        return data.coeff(row, col);
    }

    void reserveMemory() override
    {
        data.resize(num_rows, num_cols);
    }

    void set(size_t col, size_t row, double value, bool &error) override
    {
        data.coeffRef(row, col) = value;
    }

    void filterRows(const std::vector<size_t>& f) {
        data = std::move(data(f,Eigen::all)).eval();
        num_rows = f.size();
    }

  private:
    Eigen::MatrixXd data;
};

} // namespace ranger