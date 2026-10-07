#include "storage.hpp"

namespace nsb_sample8 {

namespace { int destroyed_view_count = 0; }

Vector::~Vector() { ++destroyed_view_count; }

double Vector::Sum() const
{
    double result = 0.0;
    for (int i = 0; i < size_; ++i) {
        result += data_[i];
    }
    return result;
}

void Vector::Add(double value)
{
    for (int i = 0; i < size_; ++i) {
        data_[i] += value;
    }
}

DenseMatrix::~DenseMatrix() { ++destroyed_view_count; }

double DenseMatrix::Sum() const
{
    double result = 0.0;
    for (int col = 0; col < cols_; ++col) {
        for (int row = 0; row < rows_; ++row) {
            result += Get(row, col);
        }
    }
    return result;
}

int DestroyedViewCount() { return destroyed_view_count; }

}  // namespace nsb_sample8
