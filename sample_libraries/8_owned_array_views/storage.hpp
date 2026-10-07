#pragma once

namespace nsb_sample8 {

class Vector {
public:
    Vector(double *data, int size) : data_(data), size_(size) { }
    ~Vector();

    int Size() const { return size_; }
    double *GetData() const { return data_; }
    double Sum() const;
    void Add(double value);

private:
    double *data_;
    int size_;
};

class DenseMatrix {
public:
    DenseMatrix(double *data, int rows, int cols)
        : data_(data), rows_(rows), cols_(cols) { }
    ~DenseMatrix();

    int Height() const { return rows_; }
    int Width() const { return cols_; }
    double *GetData() const { return data_; }

    double Get(int row, int col) const { return data_[row + rows_ * col]; }
    void Set(int row, int col, double value)
    {
        data_[row + rows_ * col] = value;
    }
    double Sum() const;

private:
    double *data_;
    int rows_;
    int cols_;
};

int DestroyedViewCount();

}  // namespace nsb_sample8
