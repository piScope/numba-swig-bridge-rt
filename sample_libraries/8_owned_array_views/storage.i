%module nsb_sample8_storage
%{ #include "storage.hpp" %}
%include "numba-swig-bridge-common.i"
%feature("nsb-director", "nsb_ViewCallback") nsb_sample8::ViewCallback;
NUMBA_SWIG_BRIDGE_DIRECTOR_OVERRIDE(nsb_sample8::ViewCallback::Apply(const nsb_sample8::Vector&, nsb_sample8::Vector&))
%include "storage.hpp"
NUMBA_SWIG_BRIDGE_CLASS(nsb_sample8::Vector)
NUMBA_SWIG_BRIDGE_CLASS(nsb_sample8::DenseMatrix)
NUMBA_SWIG_BRIDGE_CLASS(nsb_sample8::ViewCallback)
%extend nsb_sample8::Vector {
  NUMBA_SWIG_BRIDGE_ARRAY_VIEW_1D(GetDataArray, "dtype=float64;data=GetData;size=Size;mutable=true")
  PyObject* GetDataArray() const { return nullptr; }
}
#ifdef NUMBA_SWIG_BRIDGE_GENERATED
%include "generated/nsb_sample8_storage/numba-swig-bridge.i"
#endif
