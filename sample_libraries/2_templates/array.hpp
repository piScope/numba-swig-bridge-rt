#pragma once
#include <stdexcept>
#include <vector>

namespace nsb_sample2 {
// A small owning array. Python creates it; Numba borrows it.
template <typename T>
class Array {
public:
    explicit Array(int size) : values_(checked_size(size)) {}
    int Size() const { return static_cast<int>(values_.size()); }
    T& operator[](int index) { return values_.at(index); }
    const T& operator[](int index) const { return values_.at(index); }
    // Value-based access avoids exporting a C++ reference through the C ABI.
    T Get(int index) const { return (*this)[index]; }
    void Set(int index, T value) { (*this)[index] = value; }
    // Mutate another selected template instance through an ordinary T&
    // parameter. This exercises Numba bridge in/out object references.
    void CopyTo(Array<T>& target) const { target.values_ = values_; }
private:
    static int checked_size(int size) {
        if (size < 0) throw std::invalid_argument("negative array size");
        return size;
    }
    std::vector<T> values_;
};
} // namespace nsb_sample2
