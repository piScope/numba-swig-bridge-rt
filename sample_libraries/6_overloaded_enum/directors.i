%module nsb_sample6_directors
%{
#include "directors.hpp"
%}
%include "exception.i"
%exception {
    try { $action }
    catch (const std::exception& error) { SWIG_exception_fail(SWIG_RuntimeError, error.what()); }
}
%include "numba-swig-bridge-common.i"
// Without named selections, nsb-director exposes every lowerable virtual.
%feature("nsb-director", "nsb_Evaluator") nsb_sample6::Evaluator;
%feature("nsb-director", "nsb_ProtectedEvaluator") nsb_sample6::ProtectedEvaluator;
%feature("nsb-director", "nsb_ProtectedConcrete") nsb_sample6::ProtectedConcrete;
%include "directors.hpp"
#ifdef NUMBA_SWIG_BRIDGE_GENERATED
%include "generated/nsb_sample6_directors/numba-swig-bridge.i"
#endif
