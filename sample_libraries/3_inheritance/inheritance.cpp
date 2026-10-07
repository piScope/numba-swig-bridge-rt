#include "inheritance.hpp"
namespace nsb_sample3 {
Base::Base(int value) : value_(value) {}
int Base::BaseMethod() const { return value_; }
Derived::Derived(int value) : Base(value) {}
int Derived::DerivedMethod() const { return 2 * BaseMethod(); }
} // namespace nsb_sample3
