// Common SWIG setup for numba-swig-bridge annotations and lowering recipes.
// Include before declaring the classes and methods selected for the bridge.
#ifndef NUMBA_SWIG_BRIDGE_COMMON_I
#define NUMBA_SWIG_BRIDGE_COMMON_I

%{
#include <stdexcept>
%}

%feature("autodoc", "1");
%exception {
    try { $action }
    catch (const std::invalid_argument &error) { SWIG_exception_fail(SWIG_ValueError, error.what()); }
    catch (const std::out_of_range &error) { SWIG_exception_fail(SWIG_IndexError, error.what()); }
    catch (const std::exception &error) { SWIG_exception_fail(SWIG_RuntimeError, error.what()); }
}
%include "stdint.i"
%include "numba-swig-bridge.i"
%include "numba-swig-bridge-lowering.i"

#endif  // NUMBA_SWIG_BRIDGE_COMMON_I
