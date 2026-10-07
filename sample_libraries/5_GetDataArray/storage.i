%module nsb_sample5_storage
%{ #include "storage.hpp" %}
%include "numba-swig-bridge-common.i"
%include "storage.hpp"
NUMBA_SWIG_BRIDGE_CLASS(nsb_sample5::Vector)
NUMBA_SWIG_BRIDGE_CLASS(nsb_sample5::IntVector)
NUMBA_SWIG_BRIDGE_CLASS(nsb_sample5::FloatVector)
NUMBA_SWIG_BRIDGE_CLASS(nsb_sample5::BoolVector)
NUMBA_SWIG_BRIDGE_CLASS(nsb_sample5::DenseMatrix)
NUMBA_SWIG_BRIDGE_CLASS(nsb_sample5::DenseTensor)
%extend nsb_sample5::Vector {
  NUMBA_SWIG_BRIDGE_ARRAY_VIEW_1D(GetDataArray, "dtype=float64;data=GetData;size=Size;mutable=true")
  PyObject* GetDataArray() { return nullptr; }
}
%extend nsb_sample5::IntVector {
  NUMBA_SWIG_BRIDGE_ARRAY_VIEW_1D(GetDataArray, "dtype=int32;data=GetData;size=Size;mutable=true")
  PyObject* GetDataArray() { return nullptr; }
}
%extend nsb_sample5::FloatVector {
  NUMBA_SWIG_BRIDGE_ARRAY_VIEW_1D(GetDataArray, "dtype=float32;data=GetData;size=Size;mutable=true")
  PyObject* GetDataArray() { return nullptr; }
}
%extend nsb_sample5::BoolVector {
  NUMBA_SWIG_BRIDGE_ARRAY_VIEW_1D(GetDataArray, "dtype=bool;data=GetData;size=Size;mutable=true")
  PyObject* GetDataArray() { return nullptr; }
}
%extend nsb_sample5::DenseMatrix {
  NUMBA_SWIG_BRIDGE_ARRAY_VIEW_2D(GetDataArray, "dtype=float64;data=GetData;rows=Height;cols=Width;layout=F;mutable=true")
  PyObject* GetDataArray() { return nullptr; }
}
%extend nsb_sample5::DenseTensor {
 NUMBA_SWIG_BRIDGE_ARRAY_VIEW_3D(GetDataArray, "dtype=float64;data=Data;depth=SizeK;rows=SizeI;cols=SizeJ;layout=Kij;mutable=true")
 PyObject* GetDataArray() { return nullptr; }
}
#ifdef NUMBA_SWIG_BRIDGE_GENERATED
%include "generated/nsb_sample5_storage/numba-swig-bridge.i"
#endif
