"""Borrowed object pointer and reference results from a native director."""

from numba import types
import nsb_rt as nsb
import nsb_sample9
import nsb_sample9.registration  # noqa: F401


@nsb.director(
    base=nsb_sample9.Provider,
    binding=nsb_sample9.registration.director_binding("nsb_sample9::Provider"),
    state={"parent": nsb_sample9.Parent, "offset": types.intc},
)
class ParentProvider:
    def __init__(self, parent, offset):
        self.parent = parent
        self.offset = offset

    @nsb.override("GetChild")
    def get_child(self):
        return self.parent.GetChild()

    @nsb.override("MaybeChild", "bool")
    def maybe_child(self, present):
        return self.parent.MaybeChild(present)


provider = ParentProvider(nsb_sample9.Parent(12), 0)
try:
    base = provider.as_base_proxy()
    assert nsb_sample9.ConsumeReference(base) == 12
    assert nsb_sample9.ConsumePointer(base, True) == 12
    assert nsb_sample9.ConsumePointer(base, False) == -1
finally:
    provider.close()

print("object-returning directors: Normal End")
