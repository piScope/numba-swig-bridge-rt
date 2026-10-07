#include "directors.hpp"
#include <stdexcept>

namespace nsb_sample6 {

Evaluator::Evaluator(int equations, int dimension, Mode mode)
    : equations_(equations), dimension_(dimension), mode_(mode) {
    if (equations <= 0 || dimension <= 0) {
        throw std::invalid_argument("dimensions must be positive");
    }
}

Evaluator::Evaluator(double scale, Mode mode)
    : Evaluator(static_cast<int>(scale), 1, mode) {}

double invoke_evaluate(const Evaluator& evaluator, double value) {
    return evaluator.Evaluate(value);
}

double invoke_bias(const Evaluator& evaluator, double value) {
    return evaluator.Bias(value);
}

void invoke_record(const Evaluator& evaluator, double value) {
    evaluator.Record(value);
}

} // namespace nsb_sample6
