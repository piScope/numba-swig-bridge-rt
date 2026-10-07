#include "directors.hpp"
#include <stdexcept>
namespace nsb_sample4 {
Evaluator::Evaluator(int equations, int dimension)
    : equations_(equations), dimension_(dimension) {
    if (equations <= 0 || dimension <= 0) {
        throw std::invalid_argument("dimensions must be positive");
    }
}
double invoke_evaluate(const Evaluator& evaluator, double value) {
    return evaluator.Evaluate(value);
}
double invoke_bias(const Evaluator& evaluator, double value) {
    return evaluator.Bias(value);
}
}

namespace nsb_sample4 {
float FloatOps::SumValues(const float* values, std::int64_t count) const {
    float total = 0.0f;
    for (std::int64_t i = 0; i < count; ++i) total += values[i];
    return total;
}
float invoke_float(const FloatEvaluator& evaluator, float value) {
    return evaluator.Apply(value);
}
float invoke_float_values(const FloatEvaluator& evaluator) {
    const float values[] = {1.25f, 2.5f, 3.75f};
    return evaluator.SumValues(values, 3);
}
}
