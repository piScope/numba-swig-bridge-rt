%module nsb_sample4_directors
%{
#include "directors.hpp"
%}
%include "exception.i"
%exception {
    try { $action }
    catch (const std::exception& error) { SWIG_exception_fail(SWIG_RuntimeError, error.what()); }
}
%include "numba-swig-bridge-common.i"
%include "scalar_types.hpp"
%feature("nsb-director", "nsb_Evaluator") nsb_sample4::Evaluator;
NUMBA_SWIG_BRIDGE_DIRECTOR_OVERRIDE(nsb_sample4::Evaluator::Evaluate(double) const)
NUMBA_SWIG_BRIDGE_DIRECTOR_OVERRIDE(nsb_sample4::Evaluator::Bias(double) const)
%feature("nsb-director", "nsb_FloatEvaluator") nsb_sample4::FloatEvaluator;
namespace nsb_sample4 {
NUMBA_SWIG_BRIDGE_DIRECTOR_OVERRIDE(FloatEvaluator::Apply(scalar_t) const)
NUMBA_SWIG_BRIDGE_DIRECTOR_OVERRIDE(FloatEvaluator::SumValues(const float*, std::int64_t) const)
NUMBA_SWIG_BRIDGE_DIRECTOR_OVERRIDE(FloatEvaluator::Scale(scalar_t) const)
NUMBA_SWIG_BRIDGE_LOWERING(
    FloatEvaluator::SumValues(const float*, std::int64_t) const,
    "borrowed-contiguous-array-1d-f32-input", "pointer=0;count=1"
)
NUMBA_SWIG_BRIDGE_LOWERING(
    FloatOps::SumValues(const scalar_t*, std::int64_t) const,
    "borrowed-contiguous-array-1d-f32-input", "pointer=0;count=1"
)
}
%include "directors.hpp"
NUMBA_SWIG_BRIDGE_CLASS(nsb_sample4::FloatOps)
#ifdef NUMBA_SWIG_BRIDGE_GENERATED
%include "generated/nsb_sample4_directors/numba-swig-bridge.i"
#endif
