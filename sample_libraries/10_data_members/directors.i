%module nsb_sample10_directors
%{
#include "directors.hpp"
%}
%include "numba-swig-bridge-common.i"
%immutable nsb_sample10::Data::immutable;
%rename(renamed) nsb_sample10::Data::original;
%feature("nsb-director", "nsb_Callback") nsb_sample10::Callback;
NUMBA_SWIG_BRIDGE_MEMBER(nsb_sample10::Data::restricted, "readonly")
NUMBA_SWIG_BRIDGE_MEMBER(nsb_sample10::Data::hidden, "exclude")
%ignore nsb_sample10::Data::ignored;
%include "directors.hpp"
NUMBA_SWIG_BRIDGE_CLASS(nsb_sample10::Data)
NUMBA_SWIG_BRIDGE_CLASS(nsb_sample10::Container)
NUMBA_SWIG_BRIDGE_CLASS(nsb_sample10::Opaque)
#ifdef NUMBA_SWIG_BRIDGE_GENERATED
%include "generated/nsb_sample10_directors/numba-swig-bridge.i"
#endif
