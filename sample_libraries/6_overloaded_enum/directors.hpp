#pragma once
#include <cstdint>

namespace nsb_sample6 {

class Evaluator {
public:
    enum Mode { MULTIPLY, OFFSET };

    Evaluator(int equations, int dimension, Mode mode);
    Evaluator(double scale, Mode mode);
    virtual ~Evaluator() = default;

    int Equations() const { return equations_; }
    int Dimension() const { return dimension_; }
    Mode GetMode() const { return mode_; }
    virtual double Evaluate(double value) const = 0;
    virtual double Bias(double value) const { return value + equations_ + dimension_; }
    virtual void Record(double value) const { }

private:
    int equations_;
    int dimension_;
    Mode mode_;
};

double invoke_evaluate(const Evaluator& evaluator, double value);
double invoke_bias(const Evaluator& evaluator, double value);
void invoke_record(const Evaluator& evaluator, double value);

// Protected constructors are legal for generated subclasses even when the
// base itself is concrete. The pointer overload is deliberately unsupported.
class ProtectedEvaluator {
public:
    enum Mode { ADD, MULTIPLY };
    virtual ~ProtectedEvaluator() = default;
    virtual double Evaluate(double value) const = 0;
    double Bias() const { return bias_; }
protected:
    ProtectedEvaluator(const int* unused = nullptr) : bias_(7.0) {}
    ProtectedEvaluator(int count, double scale, Mode mode, bool enabled,
                       std::int64_t offset)
        : bias_(enabled ? (mode == ADD ? count + scale : count * scale) + offset : 0.) {}
private:
    ProtectedEvaluator(double inaccessible) : bias_(inaccessible) {}
    double bias_;
};

class ProtectedConcrete {
protected:
    ProtectedConcrete(int value = 3) : value_(value) {}
public:
    virtual ~ProtectedConcrete() = default;
    virtual double Evaluate(double value) const { return value + value_; }
private:
    int value_;
};

inline double invoke_protected(const ProtectedEvaluator& evaluator, double value) {
    return evaluator.Evaluate(value);
}
inline double invoke_concrete(const ProtectedConcrete& evaluator, double value) {
    return evaluator.Evaluate(value);
}

} // namespace nsb_sample6
