// Reusable target-wrapper opt-in. The generator discovers these helpers
// through SWIG XML; target-specific methods and ABI shims are generated later.
%define NUMBA_SWIG_BRIDGE_CLASS(TYPE)
%extend TYPE {
    unsigned long long _nsb_pointer() {
        return reinterpret_cast<unsigned long long>($self);
    }
}
%enddef

// A class-level ``nsb-director`` feature selects every virtual declaration
// that the generator can lower safely. Adding one or more named overrides
// switches that class to strict, explicit selection instead.
%define NUMBA_SWIG_BRIDGE_DIRECTOR_OVERRIDE(METHOD)
%feature("nsb-director-override", "1") METHOD;
%enddef

// Mark a Python-visible %extend method as a non-owning native array view.
// SPEC is deliberately explicit, e.g.
// "dtype=float32|float64|int32|bool;data=GetData;size=Size;mutable".
%define NUMBA_SWIG_BRIDGE_ARRAY_VIEW_1D(METHOD, SPEC)
%feature("nsb-array-view-1d", SPEC) METHOD;
%enddef

%define NUMBA_SWIG_BRIDGE_ARRAY_VIEW_2D(METHOD, SPEC)
%feature("nsb-array-view-2d", SPEC) METHOD;
%enddef

%define NUMBA_SWIG_BRIDGE_ARRAY_VIEW_3D(METHOD, SPEC)
%feature("nsb-array-view-3d", SPEC) METHOD;
%enddef

// Restrict automatic member exposure. MODE: readonly or exclude.
%define NUMBA_SWIG_BRIDGE_MEMBER(MEMBER, MODE)
%feature("nsb-member", MODE) MEMBER;
%enddef
