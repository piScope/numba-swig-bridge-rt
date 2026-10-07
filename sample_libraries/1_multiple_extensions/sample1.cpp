#include "sample1.hpp"

#include <atomic>

namespace {
std::atomic<unsigned long long> evaluator_destructions{0};
std::atomic<unsigned long long> nonvirtual_destructions{0};
}

namespace nsb_sample1 {

Buffer::Buffer(int size) {
    if (size < 0) throw std::invalid_argument("Buffer size must not be negative");
    values_.resize(static_cast<std::size_t>(size), 0.0);
}
int Buffer::Size() const noexcept { return static_cast<int>(values_.size()); }
double Buffer::Get(int index) const {
    if (index < 0 || index >= Size()) throw std::out_of_range("Buffer index out of range");
    return values_[static_cast<std::size_t>(index)];
}
void Buffer::Set(int index, double value) {
    if (index < 0 || index >= Size()) throw std::out_of_range("Buffer index out of range");
    values_[static_cast<std::size_t>(index)] = value;
}
double Buffer::Sum() const noexcept {
    double total = 0.0;
    for (double value : values_) total += value;
    return total;
}

Counter::Counter(std::int64_t initial) noexcept : value_(initial) {}
void Counter::Add(std::int64_t amount) noexcept { value_ += amount; }
std::int64_t Counter::Value() const noexcept { return value_; }
bool Counter::AtLeast(std::int64_t value) const noexcept { return value_ >= value; }

double Arithmetic::Transform(
    const Buffer& source, Buffer& destination, Counter* counter,
    double scale, int bias, std::int64_t count, bool reverse) const {
    if (counter == nullptr || count < 0 || count > source.Size() || count > destination.Size()) {
        throw std::invalid_argument("invalid borrowed-object transform arguments");
    }
    for (std::int64_t position = 0; position < count; ++position) {
        const int index = reverse ? static_cast<int>(count - 1 - position) : static_cast<int>(position);
        destination.Set(index, source.Get(index) * scale + static_cast<double>(bias));
    }
    counter->Add(count);
    return destination.Sum();
}

Evaluator::~Evaluator() { evaluator_destructions.fetch_add(1, std::memory_order_relaxed); }
double Evaluator::Bias(double value) { return 100.0 + value; }
double Evaluator::Probe(double value) { return -100.0 + value; }
double Evaluator::Probe(const double* values, std::int64_t count) {
    if (values == nullptr || count < 0) throw std::invalid_argument("invalid Probe values");
    double total = -100.0;
    for (std::int64_t index = 0; index < count; ++index) total += values[index];
    return total;
}
double Evaluator::Unselected(double value) const noexcept { return -10.0 + value; }

Inner::~Inner() = default;
Auxiliary::~Auxiliary() = default;
Outer::~Outer() = default;
ScalarReturns::~ScalarReturns() = default;
NonVirtualCallback::~NonVirtualCallback() {
    nonvirtual_destructions.fetch_add(1, std::memory_order_relaxed);
}

ExternalEvaluatorOwner::~ExternalEvaluatorOwner() { DeleteEvaluator(); }
void ExternalEvaluatorOwner::Take(Evaluator* evaluator, bool fail) {
    if (fail) throw std::runtime_error("requested external-owner insertion failure");
    if (evaluator == nullptr || evaluator_ != nullptr) {
        throw std::invalid_argument("external owner requires one non-null evaluator");
    }
    evaluator_ = evaluator;
}
bool ExternalEvaluatorOwner::HasEvaluator() const noexcept { return evaluator_ != nullptr; }
double ExternalEvaluatorOwner::InvokeBias(double value) const {
    if (evaluator_ == nullptr) throw std::runtime_error("external owner has no evaluator");
    return evaluator_->Bias(value);
}
void ExternalEvaluatorOwner::DeleteEvaluator() noexcept {
    delete evaluator_;
    evaluator_ = nullptr;
}

double invoke_evaluate(
    Evaluator& evaluator, const Buffer& source, Buffer& destination, Counter* counter,
    double scale, int bias, std::int64_t count, bool reverse) {
    return evaluator.Evaluate(source, destination, counter, scale, bias, count, reverse);
}
double invoke_evaluate_values(Evaluator& evaluator, const Buffer& values) {
    const std::int64_t count = values.Size();
    std::vector<double> packed(static_cast<std::size_t>(count));
    for (std::int64_t index = 0; index < count; ++index) packed[static_cast<std::size_t>(index)] = values.Get(static_cast<int>(index));
    return evaluator.EvaluateValues(packed.data(), count);
}
double invoke_bias(Evaluator& evaluator, double value) { return evaluator.Bias(value); }

double invoke_bias_repeated(Evaluator& evaluator, double value, int count) {
    double total = 0.0;
    for (int i = 0; i < count; ++i) { total += evaluator.Bias(value); }
    return total;
}
double invoke_probe(Evaluator& evaluator, double value) { return evaluator.Probe(value); }
double invoke_probe_values(Evaluator& evaluator, const Buffer& values) {
    const std::int64_t count = values.Size();
    std::vector<double> packed(static_cast<std::size_t>(count));
    for (std::int64_t index = 0; index < count; ++index) packed[static_cast<std::size_t>(index)] = values.Get(static_cast<int>(index));
    return evaluator.Probe(packed.data(), count);
}
double invoke_unselected(const Evaluator& evaluator, double value) { return evaluator.Unselected(value); }

double invoke_inner(Inner& inner, double value) { return inner.Apply(value); }
double invoke_auxiliary(Auxiliary& auxiliary, double value) { return auxiliary.Adjust(value); }
double invoke_outer(Outer& outer, double value) { return outer.Combine(value); }
double invoke_double_value(ScalarReturns& director, double value) { return director.DoubleValue(value); }
int invoke_int_value(ScalarReturns& director, int value) { return director.IntValue(value); }
std::int64_t invoke_int64_value(ScalarReturns& director, std::int64_t value) { return director.Int64Value(value); }
bool invoke_bool_value(ScalarReturns& director, bool value) { return director.BoolValue(value); }
double invoke_nonvirtual(NonVirtualCallback& director, double value) { return director.Apply(value); }
unsigned long long evaluator_destruction_count() noexcept {
    return evaluator_destructions.load(std::memory_order_relaxed);
}
unsigned long long nonvirtual_destruction_count() noexcept {
    return nonvirtual_destructions.load(std::memory_order_relaxed);
}

}  // namespace nsb_sample1
