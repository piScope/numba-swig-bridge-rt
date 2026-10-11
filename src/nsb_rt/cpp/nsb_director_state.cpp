#include "nsb_director_state.hpp"

#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace nsb_director_state {
namespace {

std::size_t checked_count(int count, const char* label) {
    if (count < 0) {
        throw std::invalid_argument(std::string(label) + " must not be negative");
    }
    return static_cast<std::size_t>(count);
}

}  // namespace

StateBlock::StateBlock(
    int double_slots, int int_slots, int int64_slots, int bool_slots, int object_slots, int float_slots, int array_slots)
    : doubles_(checked_count(double_slots, "double slot count"), 0.0),
      floats_(checked_count(float_slots, "float slot count"), 0.0f),
      ints_(checked_count(int_slots, "int slot count"), 0),
      int64s_(checked_count(int64_slots, "int64 slot count"), 0),
      bools_(checked_count(bool_slots, "bool slot count"), 0),
      objects_(checked_count(object_slots, "object slot count")),
      arrays_(checked_count(array_slots, "array slot count")) {}

StateBlock::~StateBlock() { Close(); }

bool StateBlock::IsAlive() const noexcept { return alive_; }

void StateBlock::SetReleaseMemInfo(unsigned long long address) noexcept {
    release_meminfo_ = reinterpret_cast<ReleaseMemInfo>(address);
}

void StateBlock::Close() noexcept {
    if (!alive_) {
        return;
    }
    alive_ = false;
    for (ObjectSlot& slot : objects_) {
        ReleaseObject(slot);
        ++slot.generation;
    }
    for (ArraySlot& slot : arrays_) {
        ReleaseArray(slot);
    }
}

void StateBlock::ReleaseObject(ObjectSlot& object) noexcept {
    if (object.meminfo != nullptr && release_meminfo_ != nullptr) {
        release_meminfo_(object.meminfo);
    }
    object.address = 0;
    object.meminfo = nullptr;
    object.ownership = ObjectOwnership::Empty;
}

void StateBlock::ReleaseArray(ArraySlot& array) noexcept {
    if (array.meminfo != nullptr && release_meminfo_ != nullptr) {
        release_meminfo_(array.meminfo);
    }
    array = ArraySlot{};
}

void StateBlock::RequireAlive() const {
    if (!alive_) {
        throw std::runtime_error("native state block is closed");
    }
}

std::size_t StateBlock::Index(int slot, std::size_t size, const char* label) {
    if (slot < 0 || static_cast<std::size_t>(slot) >= size) {
        throw std::out_of_range(std::string(label) + " state slot is out of range");
    }
    return static_cast<std::size_t>(slot);
}

std::size_t StateBlock::ObjectIndex(int slot) const { return Index(slot, objects_.size(), "object"); }

float StateBlock::Float(int slot) const { RequireAlive(); return floats_[Index(slot, floats_.size(), "float")]; }
void StateBlock::SetFloat(int slot, float value) { RequireAlive(); floats_[Index(slot, floats_.size(), "float")] = value; }
double StateBlock::Double(int slot) const { RequireAlive(); return doubles_[Index(slot, doubles_.size(), "double")]; }
void StateBlock::SetDouble(int slot, double value) { RequireAlive(); doubles_[Index(slot, doubles_.size(), "double")] = value; }
int StateBlock::Int(int slot) const { RequireAlive(); return ints_[Index(slot, ints_.size(), "int")]; }
void StateBlock::SetInt(int slot, int value) { RequireAlive(); ints_[Index(slot, ints_.size(), "int")] = value; }
std::int64_t StateBlock::Int64(int slot) const { RequireAlive(); return int64s_[Index(slot, int64s_.size(), "int64")]; }
void StateBlock::SetInt64(int slot, std::int64_t value) { RequireAlive(); int64s_[Index(slot, int64s_.size(), "int64")] = value; }
bool StateBlock::Bool(int slot) const { RequireAlive(); return bools_[Index(slot, bools_.size(), "bool")] != 0; }
void StateBlock::SetBool(int slot, bool value) { RequireAlive(); bools_[Index(slot, bools_.size(), "bool")] = value ? 1 : 0; }

void StateBlock::SetObjectFromPython(int slot, std::uintptr_t address) {
    RequireAlive();
    if (address == 0) {
        throw std::invalid_argument("Python-assigned object address must not be zero");
    }
    ObjectSlot& object = objects_[ObjectIndex(slot)];
    ReleaseObject(object);
    object.address = address;
    object.ownership = ObjectOwnership::PythonAssigned;
    ++object.generation;
}

void StateBlock::ReplaceObjectFromCpp(int slot, std::uintptr_t address) {
    RequireAlive();
    if (address == 0) {
        throw std::invalid_argument("C++-assigned object address must not be zero");
    }
    ObjectSlot& object = objects_[ObjectIndex(slot)];
    ReleaseObject(object);
    object.address = address;
    object.ownership = ObjectOwnership::CppOwned;
    ++object.generation;
}

void StateBlock::SetObjectFromNative(int slot, std::uintptr_t address, void* meminfo) {
    RequireAlive();
    if (address == 0 || meminfo == nullptr || release_meminfo_ == nullptr) {
        throw std::invalid_argument("native object state requires an address, NRT token, and release helper");
    }
    ObjectSlot& object = objects_[ObjectIndex(slot)];
    ObjectSlot previous = object;
    object.address = address;
    object.meminfo = meminfo;
    object.ownership = ObjectOwnership::NativeOwned;
    ++object.generation;
    ReleaseObject(previous);
}

void StateBlock::SetObjectBorrowed(int slot, std::uintptr_t address) {
    RequireAlive();
    ObjectSlot& object = objects_[ObjectIndex(slot)];
    ReleaseObject(object);
    object.address = address;
    object.ownership = address == 0 ? ObjectOwnership::Empty : ObjectOwnership::Borrowed;
    ++object.generation;
}

bool StateBlock::ClearObjectIf(int slot, std::uintptr_t expected_address) noexcept {
    if (!alive_ || slot < 0 || static_cast<std::size_t>(slot) >= objects_.size()) {
        return false;
    }
    ObjectSlot& object = objects_[static_cast<std::size_t>(slot)];
    if (object.address != expected_address) {
        return false;
    }
    ReleaseObject(object);
    ++object.generation;
    return true;
}

std::uintptr_t StateBlock::ObjectAddress(int slot) const { RequireAlive(); return objects_[ObjectIndex(slot)].address; }
std::uintptr_t StateBlock::ObjectMemInfo(int slot) const { RequireAlive(); return reinterpret_cast<std::uintptr_t>(objects_[ObjectIndex(slot)].meminfo); }
std::uint64_t StateBlock::ObjectGeneration(int slot) const { RequireAlive(); return objects_[ObjectIndex(slot)].generation; }
int StateBlock::ObjectOwnershipCode(int slot) const { RequireAlive(); return static_cast<int>(objects_[ObjectIndex(slot)].ownership); }
void StateBlock::SetArrayFromPython(int slot, std::uintptr_t data, int ndim, std::int64_t size0, std::int64_t size1, std::int64_t size2, std::int64_t stride0, std::int64_t stride1, std::int64_t stride2) {
    RequireAlive();
    if (data == 0 || ndim < 1 || ndim > 3 || size0 < 0 || size1 < 0 || size2 < 0) throw std::invalid_argument("array state requires valid C-contiguous dimensions");
    ArraySlot& array = arrays_[Index(slot, arrays_.size(), "array")];
    ReleaseArray(array);
    array.data = data; array.ndim = ndim; array.shape[0] = size0; array.shape[1] = size1; array.shape[2] = size2; array.strides[0] = stride0; array.strides[1] = stride1; array.strides[2] = stride2;
}

void StateBlock::SetArrayFromNative(int slot, std::uintptr_t data, void* meminfo, int ndim, std::int64_t size0, std::int64_t size1, std::int64_t size2, std::int64_t stride0, std::int64_t stride1, std::int64_t stride2) {
    RequireAlive();
    if (data == 0 || meminfo == nullptr || release_meminfo_ == nullptr || ndim < 1 || ndim > 3 || size0 < 0 || size1 < 0 || size2 < 0) {
        throw std::invalid_argument("native array state requires valid metadata, NRT token, and release helper");
    }
    ArraySlot& array = arrays_[Index(slot, arrays_.size(), "array")];
    ArraySlot previous = array;
    array.data = data; array.meminfo = meminfo; array.ndim = ndim;
    array.shape[0] = size0; array.shape[1] = size1; array.shape[2] = size2;
    array.strides[0] = stride0; array.strides[1] = stride1; array.strides[2] = stride2;
    ReleaseArray(previous);
}
std::uintptr_t StateBlock::ArrayData(int slot) const { RequireAlive(); return arrays_[Index(slot, arrays_.size(), "array")].data; }
std::int64_t StateBlock::ArrayShape(int slot, int dimension) const { RequireAlive(); const ArraySlot& array = arrays_[Index(slot, arrays_.size(), "array")]; if (dimension < 0 || dimension >= array.ndim) throw std::out_of_range("array dimension"); return array.shape[dimension]; }
std::int64_t StateBlock::ArrayStride(int slot, int dimension) const { RequireAlive(); const ArraySlot& array = arrays_[Index(slot, arrays_.size(), "array")]; if (dimension < 0 || dimension >= array.ndim) throw std::out_of_range("array dimension"); return array.strides[dimension]; }

StateHandle::StateHandle(int doubles, int ints, int int64s, int bools, int objects, int floats, int arrays, unsigned long long release_meminfo)
    : state_(std::make_shared<StateBlock>(doubles, ints, int64s, bools, objects, floats, arrays)) {
    state_->SetReleaseMemInfo(release_meminfo);
}

bool StateHandle::IsAlive() const noexcept { return state_ != nullptr && state_->IsAlive(); }
void StateHandle::Close() noexcept { if (state_ != nullptr) { state_->Close(); } }

#define NUMBA_SWIG_BRIDGE_STATE_HANDLE_ACCESSORS(NAME, TYPE) \
TYPE StateHandle::NAME(int slot) const { \
    if (state_ == nullptr) throw std::runtime_error("native state handle has no state block"); \
    return state_->NAME(slot); \
} \
void StateHandle::Set##NAME(int slot, TYPE value) { \
    if (state_ == nullptr) throw std::runtime_error("native state handle has no state block"); \
    state_->Set##NAME(slot, value); \
}

NUMBA_SWIG_BRIDGE_STATE_HANDLE_ACCESSORS(Float, float)
NUMBA_SWIG_BRIDGE_STATE_HANDLE_ACCESSORS(Double, double)
NUMBA_SWIG_BRIDGE_STATE_HANDLE_ACCESSORS(Int, int)
NUMBA_SWIG_BRIDGE_STATE_HANDLE_ACCESSORS(Int64, std::int64_t)
NUMBA_SWIG_BRIDGE_STATE_HANDLE_ACCESSORS(Bool, bool)
#undef NUMBA_SWIG_BRIDGE_STATE_HANDLE_ACCESSORS

void StateHandle::SetObjectFromPython(int slot, unsigned long long address) {
    if (state_ == nullptr) throw std::runtime_error("native state handle has no state block");
    state_->SetObjectFromPython(slot, static_cast<std::uintptr_t>(address));
}
void StateHandle::ReplaceObjectFromCpp(int slot, unsigned long long address) {
    if (state_ == nullptr) throw std::runtime_error("native state handle has no state block");
    state_->ReplaceObjectFromCpp(slot, static_cast<std::uintptr_t>(address));
}
void StateHandle::SetObjectBorrowed(int slot, unsigned long long address) {
    state_->SetObjectBorrowed(slot, static_cast<std::uintptr_t>(address));
}

void StateHandle::SetObjectFromNative(int slot, unsigned long long address, unsigned long long meminfo) {
    if (state_ == nullptr) throw std::runtime_error("native state handle has no state block");
    state_->SetObjectFromNative(slot, static_cast<std::uintptr_t>(address), reinterpret_cast<void*>(meminfo));
}
unsigned long long StateHandle::ObjectAddress(int slot) const {
    if (state_ == nullptr) throw std::runtime_error("native state handle has no state block");
    return static_cast<unsigned long long>(state_->ObjectAddress(slot));
}
unsigned long long StateHandle::ObjectGeneration(int slot) const {
    if (state_ == nullptr) throw std::runtime_error("native state handle has no state block");
    return static_cast<unsigned long long>(state_->ObjectGeneration(slot));
}
int StateHandle::ObjectOwnershipCode(int slot) const {
    if (state_ == nullptr) throw std::runtime_error("native state handle has no state block");
    return state_->ObjectOwnershipCode(slot);
}
void StateHandle::SetArrayFromPython(int slot, unsigned long long data, int ndim, std::int64_t size0, std::int64_t size1, std::int64_t size2, std::int64_t stride0, std::int64_t stride1, std::int64_t stride2) {
    if (state_ == nullptr) throw std::runtime_error("native state handle has no state block");
    state_->SetArrayFromPython(slot, static_cast<std::uintptr_t>(data), ndim, size0, size1, size2, stride0, stride1, stride2);
}
void StateHandle::SetArrayFromNative(int slot, unsigned long long data, unsigned long long meminfo, int ndim, std::int64_t size0, std::int64_t size1, std::int64_t size2, std::int64_t stride0, std::int64_t stride1, std::int64_t stride2) {
    if (state_ == nullptr) throw std::runtime_error("native state handle has no state block");
    state_->SetArrayFromNative(slot, static_cast<std::uintptr_t>(data), reinterpret_cast<void*>(meminfo), ndim, size0, size1, size2, stride0, stride1, stride2);
}
unsigned long long StateHandle::NativeAddress() const noexcept {
    return IsAlive() ? reinterpret_cast<unsigned long long>(state_.get()) : 0;
}
std::shared_ptr<StateBlock> StateHandle::SharedState() const noexcept { return state_; }
unsigned long long StateHandle::AcquireLease() const noexcept {
    if (state_ == nullptr) return 0;
    try { return reinterpret_cast<unsigned long long>(new std::shared_ptr<StateBlock>(state_)); }
    catch (...) { return 0; }
}

extern "C" float nsb_director_state_float_at(void* state, int slot) noexcept {
    try { return state == nullptr ? std::numeric_limits<float>::quiet_NaN() : static_cast<StateBlock*>(state)->Float(slot); }
    catch (...) { return std::numeric_limits<float>::quiet_NaN(); }
}
extern "C" double nsb_director_state_double_at(void* state, int slot) noexcept {
    try { return state == nullptr ? std::numeric_limits<double>::quiet_NaN() : static_cast<StateBlock*>(state)->Double(slot); }
    catch (...) { return std::numeric_limits<double>::quiet_NaN(); }
}
extern "C" int nsb_director_state_int_at(void* state, int slot) noexcept {
    try { return state == nullptr ? 0 : static_cast<StateBlock*>(state)->Int(slot); }
    catch (...) { return 0; }
}
extern "C" std::int64_t nsb_director_state_int64_at(void* state, int slot) noexcept {
    try { return state == nullptr ? 0 : static_cast<StateBlock*>(state)->Int64(slot); }
    catch (...) { return 0; }
}
extern "C" int nsb_director_state_bool_at(void* state, int slot) noexcept {
    try { return state != nullptr && static_cast<StateBlock*>(state)->Bool(slot) ? 1 : 0; }
    catch (...) { return 0; }
}
extern "C" std::uintptr_t nsb_director_state_object_address_at(void* state, int slot) noexcept {
    try { return state == nullptr ? 0 : static_cast<StateBlock*>(state)->ObjectAddress(slot); }
    catch (...) { return 0; }
}
extern "C" std::uintptr_t nsb_director_state_object_meminfo_at(void* state, int slot) noexcept {
    try { return state == nullptr ? 0 : static_cast<StateBlock*>(state)->ObjectMemInfo(slot); }
    catch (...) { return 0; }
}
extern "C" int nsb_director_state_set_object_native(void* state, int slot, std::uintptr_t address, void* meminfo) noexcept {
    try {
        if (state == nullptr) return 1;
        static_cast<StateBlock*>(state)->SetObjectFromNative(slot, address, meminfo);
        return 0;
    } catch (...) { return 1; }
}
extern "C" std::uintptr_t nsb_director_state_array_data_at(void* state, int slot) noexcept {
    try { return state == nullptr ? 0 : static_cast<StateBlock*>(state)->ArrayData(slot); }
    catch (...) { return 0; }
}
extern "C" int nsb_director_state_set_object_borrowed(void* state, int slot, std::uintptr_t address) noexcept {
    try {
        if (state == nullptr) return 1;
        static_cast<StateBlock*>(state)->SetObjectBorrowed(slot, address);
        return 0;
    } catch (...) { return 1; }
}
extern "C" std::int64_t nsb_director_state_array_shape_at(void* state, int slot, int dimension) noexcept {
    try { return state == nullptr ? 0 : static_cast<StateBlock*>(state)->ArrayShape(slot, dimension); }
    catch (...) { return 0; }
}
extern "C" std::int64_t nsb_director_state_array_stride_at(void* state, int slot, int dimension) noexcept {
    try { return state == nullptr ? 0 : static_cast<StateBlock*>(state)->ArrayStride(slot, dimension); }
    catch (...) { return 0; }
}
extern "C" int nsb_director_state_set_array_native(void* state, int slot, std::uintptr_t data, void* meminfo, int ndim, std::int64_t size0, std::int64_t size1, std::int64_t size2, std::int64_t stride0, std::int64_t stride1, std::int64_t stride2) noexcept {
    try {
        if (state == nullptr) return 1;
        static_cast<StateBlock*>(state)->SetArrayFromNative(slot, data, meminfo, ndim, size0, size1, size2, stride0, stride1, stride2);
        return 0;
    } catch (...) { return 1; }
}
extern "C" void* nsb_director_state_lease_address(void* lease) noexcept {
    const auto* shared = static_cast<std::shared_ptr<StateBlock>*>(lease);
    return shared == nullptr ? nullptr : shared->get();
}
extern "C" int nsb_director_state_lease_is_alive(void* lease) noexcept {
    const auto* shared = static_cast<std::shared_ptr<StateBlock>*>(lease);
    return shared != nullptr && *shared != nullptr && (*shared)->IsAlive() ? 1 : 0;
}
extern "C" void nsb_director_state_lease_close(void* lease) noexcept {
    const auto* shared = static_cast<std::shared_ptr<StateBlock>*>(lease);
    if (shared != nullptr && *shared != nullptr) (*shared)->Close();
}
extern "C" void nsb_director_state_lease_release(void* lease) noexcept {
    delete static_cast<std::shared_ptr<StateBlock>*>(lease);
}

unsigned long long nsb_director_state_function_address(const char* name) noexcept {
    if (name == nullptr) return 0;
    if (std::strcmp(name, "nsb_director_state_float_at") == 0) return reinterpret_cast<unsigned long long>(&nsb_director_state_float_at);
    if (std::strcmp(name, "nsb_director_state_double_at") == 0) return reinterpret_cast<unsigned long long>(&nsb_director_state_double_at);
    if (std::strcmp(name, "nsb_director_state_int_at") == 0) return reinterpret_cast<unsigned long long>(&nsb_director_state_int_at);
    if (std::strcmp(name, "nsb_director_state_int64_at") == 0) return reinterpret_cast<unsigned long long>(&nsb_director_state_int64_at);
    if (std::strcmp(name, "nsb_director_state_bool_at") == 0) return reinterpret_cast<unsigned long long>(&nsb_director_state_bool_at);
    if (std::strcmp(name, "nsb_director_state_object_address_at") == 0) return reinterpret_cast<unsigned long long>(&nsb_director_state_object_address_at);
    if (std::strcmp(name, "nsb_director_state_object_meminfo_at") == 0) return reinterpret_cast<unsigned long long>(&nsb_director_state_object_meminfo_at);
    if (std::strcmp(name, "nsb_director_state_set_object_native") == 0) return reinterpret_cast<unsigned long long>(&nsb_director_state_set_object_native);
    if (std::strcmp(name, "nsb_director_state_set_object_borrowed") == 0) return reinterpret_cast<unsigned long long>(&nsb_director_state_set_object_borrowed);
    if (std::strcmp(name, "nsb_director_state_array_data_at") == 0) return reinterpret_cast<unsigned long long>(&nsb_director_state_array_data_at);
    if (std::strcmp(name, "nsb_director_state_array_shape_at") == 0) return reinterpret_cast<unsigned long long>(&nsb_director_state_array_shape_at);
    if (std::strcmp(name, "nsb_director_state_array_stride_at") == 0) return reinterpret_cast<unsigned long long>(&nsb_director_state_array_stride_at);
    if (std::strcmp(name, "nsb_director_state_set_array_native") == 0) return reinterpret_cast<unsigned long long>(&nsb_director_state_set_array_native);
    if (std::strcmp(name, "nsb_director_state_lease_address") == 0) return reinterpret_cast<unsigned long long>(&nsb_director_state_lease_address);
    if (std::strcmp(name, "nsb_director_state_lease_is_alive") == 0) return reinterpret_cast<unsigned long long>(&nsb_director_state_lease_is_alive);
    if (std::strcmp(name, "nsb_director_state_lease_close") == 0) return reinterpret_cast<unsigned long long>(&nsb_director_state_lease_close);
    if (std::strcmp(name, "nsb_director_state_lease_release") == 0) return reinterpret_cast<unsigned long long>(&nsb_director_state_lease_release);
    return 0;
}

}  // namespace nsb_director_state
