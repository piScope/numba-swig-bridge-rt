// Reusable opt-ins for explicit typemap-like Numba lowering recipes.
//
// This include defines annotation syntax only. Recipe implementations are a
// checked Python registry in lowering_recipes.py, so target interfaces never
// contain arbitrary Python, LLVM, or C++ source. `pointer` and `count` are
// zero-based C++ parameter indices.
%define NUMBA_SWIG_BRIDGE_LOWERING(METHOD, RECIPE, OPTIONS)
%feature("numba-swig-bridge-lowering", RECIPE "|" OPTIONS) METHOD;
%enddef
