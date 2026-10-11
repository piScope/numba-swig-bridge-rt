#pragma once
#include <cstdint>
#include "scalar_types.hpp"

namespace nsb_sample4 {


// Like MFEM FluxFunction, this base requires two immutable dimensions.
class Evaluator {
public:
    Evaluator(const int equations, const int dimension);
    virtual ~Evaluator() = default;
    int Equations() const { return equations_; }
    int Dimension() const { return dimension_; }
    virtual double Evaluate(double value) const = 0;
    virtual double Bias(double value) const { return value + equations_ + dimension_; }
private:
    const int equations_;
    const int dimension_;
};

double invoke_evaluate(const Evaluator& evaluator, double value);
double invoke_bias(const Evaluator& evaluator, double value);

class FloatEvaluator {
public:
    explicit FloatEvaluator(scalar_t gain) : gain_(gain) {}
    virtual ~FloatEvaluator() = default;
    scalar_t Gain() const { return gain_; }
    virtual scalar_t Apply(scalar_t value) const = 0;
    virtual scalar_t SumValues(const float* values, std::int64_t count) const = 0;
    virtual scalar_t Scale(scalar_t value) const { return gain_ * value; }
private:
    const scalar_t gain_;
};

class FloatOps {
public:
    scalar_t RoundTrip(scalar_t value) const { return value; }
    real64_t DoubleRoundTrip(real64_t value) const { return value; }
    int Precision(float value) const { return 32; }
    int Precision(double value) const { return 64; }
    scalar_t SumValues(const scalar_t* values, std::int64_t count) const;
};
float invoke_float(const FloatEvaluator& evaluator, float value);
float invoke_float_values(const FloatEvaluator& evaluator);

} // namespace nsb_sample4
