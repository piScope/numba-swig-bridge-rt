#pragma once
#include <memory>
#include <vector>
namespace nsb_sample5 {
class Vector {
public:
 explicit Vector(int n): data_(n) {}
 int Size() const { return static_cast<int>(data_.size()); }
 double *GetData() const { return const_cast<double*>(data_.data()); }
 double Get(int i) const { return data_.at(i); }
 void Set(int i, double v) { data_.at(i)=v; }
private: std::vector<double> data_;
};
class IntVector {
public:
 explicit IntVector(int n): data_(n) {}
 int Size() const { return static_cast<int>(data_.size()); }
 int *GetData() const { return const_cast<int*>(data_.data()); }
 int Get(int i) const { return data_.at(i); }
private: std::vector<int> data_;
};
class FloatVector {
public:
 explicit FloatVector(int n): data_(n) {}
 int Size() const { return static_cast<int>(data_.size()); }
 float *GetData() const { return const_cast<float*>(data_.data()); }
 float Get(int i) const { return data_.at(i); }
private: std::vector<float> data_;
};
class BoolVector {
public:
 explicit BoolVector(int n): size_(n), data_(std::make_unique<bool[]>(n)) {}
 int Size() const { return size_; }
 bool *GetData() const { return data_.get(); }
 bool Get(int i) const { return data_[i]; }
private: int size_; std::unique_ptr<bool[]> data_;
};
class DenseTensor { public: DenseTensor(int i,int j,int k): i_(i),j_(j),k_(k),data_(i*j*k) {} int SizeI() const{return i_;} int SizeJ() const{return j_;} int SizeK() const{return k_;} double *Data(){return data_.data();} private:int i_,j_,k_;std::vector<double> data_; };
class DenseMatrix {
public:
 DenseMatrix(int rows,int cols): rows_(rows),cols_(cols),data_(rows*cols) {}
 int Height() const { return rows_; }
 int Width() const { return cols_; }
 double *GetData() const { return const_cast<double*>(data_.data()); }
private: int rows_, cols_; std::vector<double> data_;
};
}
