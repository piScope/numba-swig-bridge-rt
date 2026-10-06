#pragma once

#define NUMBA_SWIG_BRIDGE_STATE_ABI_VERSION 6

#include <cstdint>
#include <memory>
#include <vector>

namespace nsb_director_state {

enum class ObjectOwnership : int {
    Empty = 0,
    PythonAssigned = 1,
    CppOwned = 2,
    NativeOwned = 3,
};

// Native state owns values, never the C++ objects addressed by object slots.
// Each object assignment increments its generation so the Python proxy cache
// can discard a stale borrowed proxy after a C++-side replacement.
class StateBlock {
public:
    StateBlock(
        int double_slots,
        int int_slots,
        int int64_slots,
        int bool_slots,
        int object_slots, int float_slots = 0, int array_slots = 0);
    ~StateBlock();

    [[nodiscard]] bool IsAlive() const noexcept;
    void Close() noexcept;
    void SetReleaseMemInfo(unsigned long long address) noexcept;

    [[nodiscard]] float Float(int slot) const;
    void SetFloat(int slot, float value);
    [[nodiscard]] double Double(int slot) const;
    void SetDouble(int slot, double value);
    [[nodiscard]] int Int(int slot) const;
    void SetInt(int slot, int value);
    [[nodiscard]] std::int64_t Int64(int slot) const;
    void SetInt64(int slot, std::int64_t value);
    [[nodiscard]] bool Bool(int slot) const;
    void SetBool(int slot, bool value);

    void SetObjectFromPython(int slot, std::uintptr_t address);
    void ReplaceObjectFromCpp(int slot, std::uintptr_t address);
    void SetObjectFromNative(int slot, std::uintptr_t address, void* meminfo);
    [[nodiscard]] bool ClearObjectIf(int slot, std::uintptr_t expected_address) noexcept;
    [[nodiscard]] std::uintptr_t ObjectAddress(int slot) const;
    [[nodiscard]] std::uintptr_t ObjectMemInfo(int slot) const;
    [[nodiscard]] std::uint64_t ObjectGeneration(int slot) const;
    [[nodiscard]] int ObjectOwnershipCode(int slot) const;
    void SetArrayFromPython(int slot, std::uintptr_t data, int ndim, std::int64_t size0, std::int64_t size1, std::int64_t size2, std::int64_t stride0, std::int64_t stride1, std::int64_t stride2);
    void SetArrayFromNative(int slot, std::uintptr_t data, void* meminfo, int ndim, std::int64_t size0, std::int64_t size1, std::int64_t size2, std::int64_t stride0, std::int64_t stride1, std::int64_t stride2);
    [[nodiscard]] std::uintptr_t ArrayData(int slot) const;
    [[nodiscard]] std::int64_t ArrayShape(int slot, int dimension) const;
    [[nodiscard]] std::int64_t ArrayStride(int slot, int dimension) const;

private:
    struct ObjectSlot {
        std::uintptr_t address = 0;
        void* meminfo = nullptr;
        std::uint64_t generation = 0;
        ObjectOwnership ownership = ObjectOwnership::Empty;
    };
    struct ArraySlot { std::uintptr_t data = 0; void* meminfo = nullptr; int ndim = 0; std::int64_t shape[3] = {0, 0, 0}; std::int64_t strides[3] = {0, 0, 0}; };

    using ReleaseMemInfo = void (*)(void*);
    void ReleaseObject(ObjectSlot& object) noexcept;
    void ReleaseArray(ArraySlot& array) noexcept;

    void RequireAlive() const;
    [[nodiscard]] static std::size_t Index(int slot, std::size_t size, const char* label);
    [[nodiscard]] std::size_t ObjectIndex(int slot) const;

    bool alive_ = true;
    std::vector<double> doubles_;
    std::vector<float> floats_;
    std::vector<int> ints_;
    std::vector<std::int64_t> int64s_;
    std::vector<std::uint8_t> bools_;
    std::vector<ObjectSlot> objects_;
    std::vector<ArraySlot> arrays_;
    ReleaseMemInfo release_meminfo_ = nullptr;
};

class StateHandle {
public:
    StateHandle(int double_slots, int int_slots, int int64_slots, int bool_slots, int object_slots, int float_slots = 0, int array_slots = 0, unsigned long long release_meminfo = 0);

    [[nodiscard]] bool IsAlive() const noexcept;
    void Close() noexcept;

    [[nodiscard]] float Float(int slot) const;
    void SetFloat(int slot, float value);
    [[nodiscard]] double Double(int slot) const;
    void SetDouble(int slot, double value);
    [[nodiscard]] int Int(int slot) const;
    void SetInt(int slot, int value);
    [[nodiscard]] std::int64_t Int64(int slot) const;
    void SetInt64(int slot, std::int64_t value);
    [[nodiscard]] bool Bool(int slot) const;
    void SetBool(int slot, bool value);

    void SetObjectFromPython(int slot, unsigned long long address);
    void ReplaceObjectFromCpp(int slot, unsigned long long address);
    void SetObjectFromNative(int slot, unsigned long long address, unsigned long long meminfo);
    [[nodiscard]] unsigned long long ObjectAddress(int slot) const;
    [[nodiscard]] unsigned long long ObjectGeneration(int slot) const;
    [[nodiscard]] int ObjectOwnershipCode(int slot) const;
    void SetArrayFromPython(int slot, unsigned long long data, int ndim, std::int64_t size0, std::int64_t size1, std::int64_t size2, std::int64_t stride0, std::int64_t stride1, std::int64_t stride2);
    void SetArrayFromNative(int slot, unsigned long long data, unsigned long long meminfo, int ndim, std::int64_t size0, std::int64_t size1, std::int64_t size2, std::int64_t stride0, std::int64_t stride1, std::int64_t stride2);
    [[nodiscard]] unsigned long long NativeAddress() const noexcept;
    [[nodiscard]] std::shared_ptr<StateBlock> SharedState() const noexcept;
    // Transfer one independently owned opaque native lease to a generated
    // director.  The recipient must release it with
    // nsb_director_state_lease_release; it must not inspect its C++ type.
    [[nodiscard]] unsigned long long AcquireLease() const noexcept;

private:
    std::shared_ptr<StateBlock> state_;
};

extern "C" float nsb_director_state_float_at(void* opaque_state, int slot) noexcept;
extern "C" double nsb_director_state_double_at(void* opaque_state, int slot) noexcept;
extern "C" int nsb_director_state_int_at(void* opaque_state, int slot) noexcept;
extern "C" std::int64_t nsb_director_state_int64_at(void* opaque_state, int slot) noexcept;
extern "C" int nsb_director_state_bool_at(void* opaque_state, int slot) noexcept;
extern "C" std::uintptr_t nsb_director_state_object_address_at(void* opaque_state, int slot) noexcept;
extern "C" std::uintptr_t nsb_director_state_object_meminfo_at(void* opaque_state, int slot) noexcept;
extern "C" int nsb_director_state_set_object_native(void* opaque_state, int slot, std::uintptr_t address, void* meminfo) noexcept;
extern "C" std::uintptr_t nsb_director_state_array_data_at(void* opaque_state, int slot) noexcept;
extern "C" std::int64_t nsb_director_state_array_shape_at(void* opaque_state, int slot, int dimension) noexcept;
extern "C" std::int64_t nsb_director_state_array_stride_at(void* opaque_state, int slot, int dimension) noexcept;
extern "C" int nsb_director_state_set_array_native(void* opaque_state, int slot, std::uintptr_t data, void* meminfo, int ndim, std::int64_t size0, std::int64_t size1, std::int64_t size2, std::int64_t stride0, std::int64_t stride1, std::int64_t stride2) noexcept;
extern "C" void* nsb_director_state_lease_address(void* lease) noexcept;
extern "C" int nsb_director_state_lease_is_alive(void* lease) noexcept;
extern "C" void nsb_director_state_lease_close(void* lease) noexcept;
extern "C" void nsb_director_state_lease_release(void* lease) noexcept;
unsigned long long nsb_director_state_function_address(const char* name) noexcept;

}  // namespace nsb_director_state
