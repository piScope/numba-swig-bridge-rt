%module nsb_sample7_directors
%{
#include "directors.hpp"
%}
%include "numba-swig-bridge-common.i"
%feature("nsb-director", "nsb_Accumulator") nsb_sample7::Accumulator;
NUMBA_SWIG_BRIDGE_DIRECTOR_OVERRIDE(nsb_sample7::Accumulator::Accumulate(double) const)
%include "directors.hpp"
#ifdef NUMBA_SWIG_BRIDGE_GENERATED
%include "generated/nsb_sample7_directors/numba-swig-bridge.i"
#endif
