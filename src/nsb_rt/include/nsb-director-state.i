%module nsb_director_state_bindings
%feature("autodoc", "1");
%include "stdint.i"
%constant int NUMBA_SWIG_BRIDGE_STATE_ABI_VERSION = NUMBA_SWIG_BRIDGE_STATE_ABI_VERSION;
%{
#include "nsb_director_state.hpp"
#include <stdexcept>
%}
%exception {
    try { $action }
    catch (const std::invalid_argument &error) { SWIG_exception_fail(SWIG_ValueError, error.what()); }
    catch (const std::out_of_range &error) { SWIG_exception_fail(SWIG_IndexError, error.what()); }
    catch (const std::exception &error) { SWIG_exception_fail(SWIG_RuntimeError, error.what()); }
}
namespace nsb_director_state {
%nodefaultctor StateHandle;
class StateHandle {
public:
    StateHandle(int double_slots, int int_slots, int int64_slots, int bool_slots, int object_slots, int float_slots = 0, int array_slots = 0, unsigned long long release_meminfo = 0);
    bool IsAlive() const;
    void Close();
    float Float(int slot) const;
    void SetFloat(int slot, float value);
    double Double(int slot) const;
    void SetDouble(int slot, double value);
    int Int(int slot) const;
    void SetInt(int slot, int value);
    std::int64_t Int64(int slot) const;
    void SetInt64(int slot, std::int64_t value);
    bool Bool(int slot) const;
    void SetBool(int slot, bool value);
    void SetObjectFromPython(int slot, unsigned long long address);
    void SetObjectBorrowed(int slot, unsigned long long address);
    void ReplaceObjectFromCpp(int slot, unsigned long long address);
    void SetObjectFromNative(int slot, unsigned long long address, unsigned long long meminfo);
    unsigned long long ObjectAddress(int slot) const;
    unsigned long long ObjectGeneration(int slot) const;
    int ObjectOwnershipCode(int slot) const;
    void SetArrayFromPython(int slot, unsigned long long data, int ndim, std::int64_t size0, std::int64_t size1, std::int64_t size2, std::int64_t stride0, std::int64_t stride1, std::int64_t stride2);
    void SetArrayFromNative(int slot, unsigned long long data, unsigned long long meminfo, int ndim, std::int64_t size0, std::int64_t size1, std::int64_t size2, std::int64_t stride0, std::int64_t stride1, std::int64_t stride2);
    unsigned long long NativeAddress() const;
    unsigned long long AcquireLease() const;
};
unsigned long long nsb_director_state_function_address(const char* name);
}  // namespace nsb_director_state
