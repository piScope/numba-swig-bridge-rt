%module nsb_sample1_objects
%{
#include "objects.hpp"
%}
%include "numba-swig-bridge-common.i"

%include "objects.hpp"

NUMBA_SWIG_BRIDGE_CLASS(nsb_sample1::Buffer)
NUMBA_SWIG_BRIDGE_CLASS(nsb_sample1::Counter)

#ifdef NUMBA_SWIG_BRIDGE_GENERATED
%include "generated/nsb_sample1_objects/numba-swig-bridge.i"
#endif
