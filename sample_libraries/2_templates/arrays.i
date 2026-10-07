%module nsb_sample2_arrays
%{
#include "array.hpp"
%}
%include "numba-swig-bridge-common.i"
// SWIG and Numba expose indexing through scalar Get/Set adapters.
%ignore nsb_sample2::Array::operator[];
%rename(__getitem__) nsb_sample2::Array::Get;
%rename(__setitem__) nsb_sample2::Array::Set;
%include "array.hpp"
NUMBA_SWIG_BRIDGE_CLASS(nsb_sample2::Array<int>)
NUMBA_SWIG_BRIDGE_CLASS(nsb_sample2::Array<double>)
%template(IntArray) nsb_sample2::Array<int>;
%template(DoubleArray) nsb_sample2::Array<double>;
#ifdef NUMBA_SWIG_BRIDGE_GENERATED
%include "generated/nsb_sample2_arrays/numba-swig-bridge.i"
#endif
