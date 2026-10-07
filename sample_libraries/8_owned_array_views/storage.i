%module nsb_sample8_storage
%{ #include "storage.hpp" %}
%include "numba-swig-bridge-common.i"
%include "storage.hpp"
NUMBA_SWIG_BRIDGE_CLASS(nsb_sample8::Vector)
NUMBA_SWIG_BRIDGE_CLASS(nsb_sample8::DenseMatrix)
#ifdef NUMBA_SWIG_BRIDGE_GENERATED
%include "generated/nsb_sample8_storage/numba-swig-bridge.i"
#endif
