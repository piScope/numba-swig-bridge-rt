"""Lazily construct an NSB-owned C++ object in director state."""

import nsb_rt as nsb
import nsb_sample9
import nsb_sample9.registration  # noqa: F401


@nsb.director(
    base=nsb_sample9.Provider,
    binding=nsb_sample9.registration.director_binding("nsb_sample9::Provider"),
    state={"parent": nsb_sample9.Parent},
)
class LazyProvider:
    def __init__(self):
        # ``parent`` deliberately starts empty. The compiled overrides create
        # it on the first C++ callback and StateBlock retains its NRT ownership.
        pass

    @nsb.override("GetChild")
    def get_child(self):
        if not nsb.is_slot_occupied(self, "parent"):
            self.parent = nsb_sample9.Parent(41)
        return self.parent.GetChild()

    @nsb.override("MaybeChild", "bool")
    def maybe_child(self, present):
        if not nsb.is_slot_occupied(self, "parent"):
            self.parent = nsb_sample9.Parent(41)
        elif present:
            # Replacing a native state value releases the earlier Parent.
            self.parent = nsb_sample9.Parent(42)
        return self.parent.MaybeChild(present)


before = nsb_sample9.DestroyedParentCount()
provider = LazyProvider()
try:
    base = provider.as_base_proxy()
    assert nsb_sample9.ConsumeReference(base) == 41
    assert nsb_sample9.ConsumePointer(base, True) == 42
    # The first Parent was released when the replacement was installed.
    assert nsb_sample9.DestroyedParentCount() == before + 1
    assert nsb_sample9.ConsumePointer(base, False) == -1
finally:
    provider.close()
assert nsb_sample9.DestroyedParentCount() == before + 2

print("native-owned director state: Normal End")
