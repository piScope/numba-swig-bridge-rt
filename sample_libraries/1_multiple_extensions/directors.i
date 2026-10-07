%module nsb_sample1_directors
%{
#include "directors.hpp"
%}
%include "exception.i"
%exception {
    try { $action }
    catch (const std::exception& error) {
        SWIG_exception_fail(SWIG_RuntimeError, error.what());
    }
}
%include "numba-swig-bridge-common.i"

%import "objects.i"

%feature("nsb-director", "nsb_Evaluator") nsb_sample1::Evaluator;
NUMBA_SWIG_BRIDGE_DIRECTOR_OVERRIDE(nsb_sample1::Evaluator::Bias(double))
namespace nsb_sample1 {
NUMBA_SWIG_BRIDGE_DIRECTOR_OVERRIDE(
    Evaluator::Evaluate(Buffer const&, Buffer&, Counter*, double, int, std::int64_t, bool)
)
}
NUMBA_SWIG_BRIDGE_DIRECTOR_OVERRIDE(nsb_sample1::Evaluator::EvaluateValues(const double *, std::int64_t))
NUMBA_SWIG_BRIDGE_LOWERING(
    nsb_sample1::Evaluator::EvaluateValues(const double *, std::int64_t),
    "borrowed-contiguous-array-1d-f64-input", "pointer=0;count=1"
)
NUMBA_SWIG_BRIDGE_DIRECTOR_OVERRIDE(nsb_sample1::Evaluator::Probe(double))
NUMBA_SWIG_BRIDGE_DIRECTOR_OVERRIDE(nsb_sample1::Evaluator::Probe(const double *, std::int64_t))
NUMBA_SWIG_BRIDGE_LOWERING(
    nsb_sample1::Evaluator::Probe(const double *, std::int64_t),
    "borrowed-contiguous-array-1d-f64-input", "pointer=0;count=1"
)

%feature("nsb-director", "nsb_Inner") nsb_sample1::Inner;
%feature("nsb-director", "nsb_Auxiliary") nsb_sample1::Auxiliary;
%feature("nsb-director", "nsb_Outer") nsb_sample1::Outer;
%feature("nsb-director", "nsb_ScalarReturns") nsb_sample1::ScalarReturns;
%feature("nsb-director", "nsb_NonVirtualCallback") nsb_sample1::NonVirtualCallback;
NUMBA_SWIG_BRIDGE_DIRECTOR_OVERRIDE(nsb_sample1::Inner::Apply(double))
NUMBA_SWIG_BRIDGE_DIRECTOR_OVERRIDE(nsb_sample1::Auxiliary::Adjust(double))
NUMBA_SWIG_BRIDGE_DIRECTOR_OVERRIDE(nsb_sample1::Outer::Combine(double))
NUMBA_SWIG_BRIDGE_DIRECTOR_OVERRIDE(nsb_sample1::ScalarReturns::DoubleValue(double))
NUMBA_SWIG_BRIDGE_DIRECTOR_OVERRIDE(nsb_sample1::ScalarReturns::IntValue(int))
NUMBA_SWIG_BRIDGE_DIRECTOR_OVERRIDE(nsb_sample1::ScalarReturns::Int64Value(std::int64_t))
NUMBA_SWIG_BRIDGE_DIRECTOR_OVERRIDE(nsb_sample1::ScalarReturns::BoolValue(bool))
NUMBA_SWIG_BRIDGE_DIRECTOR_OVERRIDE(nsb_sample1::NonVirtualCallback::Apply(double))

%include "directors.hpp"

NUMBA_SWIG_BRIDGE_CLASS(nsb_sample1::Arithmetic)

#ifdef NUMBA_SWIG_BRIDGE_GENERATED
%include "generated/nsb_sample1_directors/numba-swig-bridge.i"
#endif
