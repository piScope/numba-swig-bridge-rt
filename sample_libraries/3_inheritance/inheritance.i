%module nsb_sample3_inheritance
%{
#include "inheritance.hpp"
%}
%include "numba-swig-bridge-common.i"
%include "inheritance.hpp"
NUMBA_SWIG_BRIDGE_CLASS(nsb_sample3::Base)
NUMBA_SWIG_BRIDGE_CLASS(nsb_sample3::Derived)
NUMBA_SWIG_BRIDGE_CLASS(nsb_sample3::GrandDerived)
NUMBA_SWIG_BRIDGE_CLASS(nsb_sample3::Side)
NUMBA_SWIG_BRIDGE_CLASS(nsb_sample3::Multiple)
NUMBA_SWIG_BRIDGE_CLASS(nsb_sample3::VirtualDerived)
NUMBA_SWIG_BRIDGE_CLASS(nsb_sample3::Alternate)
NUMBA_SWIG_BRIDGE_CLASS(nsb_sample3::Ambiguous)
NUMBA_SWIG_BRIDGE_CLASS(nsb_sample3::Hiding)
NUMBA_SWIG_BRIDGE_CLASS(nsb_sample3::Consumer)
NUMBA_SWIG_BRIDGE_CLASS(nsb_sample3::Plain)
NUMBA_SWIG_BRIDGE_CLASS(nsb_sample3::PlainDerived)
NUMBA_SWIG_BRIDGE_CLASS(nsb_sample3::Private)
#ifdef NUMBA_SWIG_BRIDGE_GENERATED
%include "generated/nsb_sample3_inheritance/numba-swig-bridge.i"
#endif
