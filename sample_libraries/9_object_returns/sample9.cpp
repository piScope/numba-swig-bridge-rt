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

}  // namespace nsb_sample9
