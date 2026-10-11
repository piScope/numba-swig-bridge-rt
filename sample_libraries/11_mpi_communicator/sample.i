%module nsb_sample11_api
%{
#include "sample.hpp"
#include <mpi4py/mpi4py.h>
%}
%include "numba-swig-bridge-common.i"
// Preserve the external name in SWIG; its native representation is MPI-owned.
typedef int MPI_Comm;
%typemap(in) MPI_Comm {
    if (import_mpi4py() < 0) { SWIG_fail; }
    MPI_Comm *handle = PyMPIComm_Get($input);
    if (!handle) { SWIG_fail; }
    $1 = *handle;
}
%typemap(out) MPI_Comm {
    PyErr_SetString(PyExc_TypeError, "native communicator returns to Python are unsupported");
    SWIG_fail;
}
NUMBA_SWIG_BRIDGE_CLASS(CommConsumer)
%feature("nsb-director", "nsb_CommCallback") CommCallback;
%include "sample.hpp"
#ifdef NUMBA_SWIG_BRIDGE_GENERATED
%include "generated/nsb_sample11_api/numba-swig-bridge.i"
#endif
