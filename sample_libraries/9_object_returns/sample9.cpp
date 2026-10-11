#include "directors.hpp"

namespace nsb_sample9 {

namespace {
int destroyed_parent_count = 0;
}

Parent::~Parent() { ++destroyed_parent_count; }

int ConsumeReference(Provider &provider) { return provider.GetChild().Value(); }

int ConsumePointer(Provider &provider, bool present)
{
    Child *child = provider.MaybeChild(present);
    return child == nullptr ? -1 : child->Value();
}

int DestroyedParentCount() { return destroyed_parent_count; }

namespace { int destroyed_gradient_count = 0; }
GradientBase::~GradientBase() { ++destroyed_gradient_count; }
Form::~Form() { delete gradient_; }
GradientBase& Form::GetGradient(int value) {
    delete gradient_;
    gradient_ = new Gradient(value);
    return *gradient_;
}
int ConsumeGradient(GradientConsumer& consumer, Form& form, int value) {
    consumer.SetOperator(form.GetGradient(value));
    return consumer.Read();
}
int DestroyedGradientCount() { return destroyed_gradient_count; }

}  // namespace nsb_sample9
