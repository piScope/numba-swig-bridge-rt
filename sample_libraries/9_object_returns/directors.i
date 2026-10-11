%module nsb_sample9_api
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

%feature("nsb-director", "nsb_Provider") nsb_sample9::Provider;
NUMBA_SWIG_BRIDGE_DIRECTOR_OVERRIDE(nsb_sample9::Provider::GetChild())
NUMBA_SWIG_BRIDGE_DIRECTOR_OVERRIDE(nsb_sample9::Provider::MaybeChild(bool))
%feature("nsb-director", "nsb_GradientConsumer") nsb_sample9::GradientConsumer;
NUMBA_SWIG_BRIDGE_DIRECTOR_OVERRIDE(nsb_sample9::GradientConsumer::SetOperator(const nsb_sample9::GradientBase&))
NUMBA_SWIG_BRIDGE_DIRECTOR_OVERRIDE(nsb_sample9::GradientConsumer::Read())

%include "directors.hpp"
NUMBA_SWIG_BRIDGE_CLASS(nsb_sample9::Child)
NUMBA_SWIG_BRIDGE_CLASS(nsb_sample9::Parent)
NUMBA_SWIG_BRIDGE_CLASS(nsb_sample9::Provider)
NUMBA_SWIG_BRIDGE_CLASS(nsb_sample9::GradientBase)
NUMBA_SWIG_BRIDGE_CLASS(nsb_sample9::Gradient)
NUMBA_SWIG_BRIDGE_CLASS(nsb_sample9::Form)
NUMBA_SWIG_BRIDGE_CLASS(nsb_sample9::GradientConsumer)

#ifdef NUMBA_SWIG_BRIDGE_GENERATED
%include "generated/nsb_sample9_api/numba-swig-bridge.i"
#endif
