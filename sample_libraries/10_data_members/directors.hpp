#pragma once
#include <cstdint>
namespace nsb_sample10 {
class Data {
public:
    enum Mode { FIRST, SECOND };
    typedef double real_t;
    int count = 2;
    std::int64_t wide = 3;
    float single = 1.5f;
    real_t weight = 2.5;
    bool enabled = true;
    Mode mode = FIRST;
    const int fixed = 7;
    int immutable = 8;
    int restricted = 9;
    int original = 10;
    int hidden = 11;
    int ignored = 12;
    Data() = default;
    virtual ~Data() = default;
    double Total() const { return count + wide + single + weight + enabled + mode; }
    void Increment() { ++count; }
    void TakeMutable(Data& other) { other.Increment(); }
    const Data& ConstView() const { return *this; }
};
class Opaque {
    int count_ = 1;
public:
    int Read() const { return count_; }
    void Write(int count) { count_ = count; }
    const Opaque& Self() const { return *this; }
};
class Unbridged {};
class Container {
    Data storage;
    Opaque opaque_storage;
public:
    Data* pointer = &storage;
    Data* null_pointer = nullptr;
    const Data* const_pointee = &storage;
    Data* const const_slot = &storage;
    const Data* const both_const = &storage;
    const Opaque* opaque = &opaque_storage;
    Unbridged* unsupported = nullptr;
    Container() = default;
    const Container& ConstView() const { return *this; }
    int ReadCount() const { return storage.count; }
};
class Callback {
public:
    Callback() = default;
    virtual ~Callback() = default;
    virtual double Apply(const Data& source, Data& target) = 0;
};
inline double consume(Callback& callback, const Data& source, Data& target) {
    return callback.Apply(source, target) + target.Total();
}
}
