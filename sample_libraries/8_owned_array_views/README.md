# Sample 8: owned array views

This compact C++ fixture is independent of MFEM. `Vector(double *, int)` and
`DenseMatrix(double *, int, int)` are non-owning views over caller-provided
double storage. It demonstrates raw `T*` returns and frame-owned constructor
lowering, including an assertion that both generated C++ view destructors run.
