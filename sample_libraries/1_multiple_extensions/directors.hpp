#pragma once

#include "objects.hpp"

namespace nsb_sample1 {

class Arithmetic {
public:
    [[nodiscard]] double Transform(
        const Buffer& source, Buffer& destination, Counter* counter,
        double scale, int bias, std::int64_t count, bool reverse) const;
};

// Exercises overload selectors, object arguments, pointer/count recipes,
// and same-base director nesting.
class Evaluator {
public:
    Evaluator() = default;
    virtual ~Evaluator();
    [[nodiscard]] virtual double Bias(double value);
    [[nodiscard]] virtual double Evaluate(
        const Buffer& source, Buffer& destination, Counter* counter,
        double scale, int bias, std::int64_t count, bool reverse) = 0;
    [[nodiscard]] virtual double EvaluateValues(const double* values, std::int64_t count) = 0;
    [[nodiscard]] virtual double Probe(double value);
    [[nodiscard]] virtual double Probe(const double* values, std::int64_t count);
    [[nodiscard]] virtual double Unselected(double value) const noexcept;
};

// Three unrelated bases prove heterogeneous director fields in one Outer state.
class Inner {
public:
    Inner() = default;
    virtual ~Inner();
    [[nodiscard]] virtual double Apply(double value) = 0;
};

class Auxiliary {
public:
    Auxiliary() = default;
    virtual ~Auxiliary();
    [[nodiscard]] virtual double Adjust(double value) = 0;
};

class Outer {
public:
    Outer() = default;
    virtual ~Outer();
    [[nodiscard]] virtual double Combine(double value) = 0;
};

// Scalar callbacks use their native C++ virtual result types.
class ScalarReturns {
public:
    ScalarReturns() = default;
    virtual ~ScalarReturns();
    [[nodiscard]] virtual double DoubleValue(double value) = 0;
    [[nodiscard]] virtual int IntValue(int value) = 0;
    [[nodiscard]] virtual std::int64_t Int64Value(std::int64_t value) = 0;
    [[nodiscard]] virtual bool BoolValue(bool value) = 0;
};

// A polymorphic base whose non-virtual destructor makes delete Base* unsafe.
// NSB may own its concrete generated director, but must reject ownership
// release through the public director facade.
class NonVirtualCallback {
public:
    NonVirtualCallback() = default;
    ~NonVirtualCallback();
    [[nodiscard]] virtual double Apply(double value) = 0;
};

// A focused stand-in for libraries such as MFEM whose container assumes
// ownership of a director through a virtual base pointer.
class ExternalEvaluatorOwner {
public:
    ExternalEvaluatorOwner() = default;
    ~ExternalEvaluatorOwner();
    void Take(Evaluator* evaluator, bool fail = false);
    [[nodiscard]] bool HasEvaluator() const noexcept;
    [[nodiscard]] double InvokeBias(double value) const;
    void DeleteEvaluator() noexcept;

private:
    Evaluator* evaluator_ = nullptr;
};

[[nodiscard]] double invoke_evaluate(
    Evaluator& evaluator, const Buffer& source, Buffer& destination, Counter* counter,
    double scale, int bias, std::int64_t count, bool reverse);
[[nodiscard]] double invoke_evaluate_values(Evaluator& evaluator, const Buffer& values);
[[nodiscard]] double invoke_bias(Evaluator& evaluator, double value);
[[nodiscard]] double invoke_bias_repeated(Evaluator& evaluator, double value, int count);
[[nodiscard]] double invoke_probe(Evaluator& evaluator, double value);
[[nodiscard]] double invoke_probe_values(Evaluator& evaluator, const Buffer& values);
[[nodiscard]] double invoke_unselected(const Evaluator& evaluator, double value);

[[nodiscard]] double invoke_inner(Inner& inner, double value);
[[nodiscard]] double invoke_auxiliary(Auxiliary& auxiliary, double value);
[[nodiscard]] double invoke_outer(Outer& outer, double value);
[[nodiscard]] double invoke_double_value(ScalarReturns& director, double value);
[[nodiscard]] int invoke_int_value(ScalarReturns& director, int value);
[[nodiscard]] std::int64_t invoke_int64_value(ScalarReturns& director, std::int64_t value);
[[nodiscard]] bool invoke_bool_value(ScalarReturns& director, bool value);
[[nodiscard]] double invoke_nonvirtual(NonVirtualCallback& director, double value);
[[nodiscard]] unsigned long long evaluator_destruction_count() noexcept;
[[nodiscard]] unsigned long long nonvirtual_destruction_count() noexcept;

}  // namespace nsb_sample1
