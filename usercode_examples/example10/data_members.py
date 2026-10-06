"""Scalar/enum member access and const rejection through installed bridges."""
from numba import njit, types
from numba.core.errors import TypingError
import nsb_rt as nsb
import nsb_sample10 as sample
import nsb_sample10.registration  # noqa: F401


@njit
def change(data):
    data.count = 12
    data.wide = 2**40
    data.single = 3.25
    data.weight = 4.5
    data.enabled = False
    data.mode = 1
    data.renamed = 13
    return data.fixed + data.immutable + data.restricted


@njit
def owned():
    data = sample.Data()
    data.count = 15
    return data.count + data.weight


@nsb.director(state={"unused": types.intc})
class Callback(sample.Callback):
    def __init__(self):
        super().__init__()
        self.unused = 0

    @nsb.override
    def Apply(self, source, target):
        target.count = source.count + 1
        target.weight = source.weight * 2
        return source.fixed + target.weight


def rejects(function, *args):
    try:
        njit(function)(*args)
    except TypingError:
        return
    raise AssertionError("unsupported member operation compiled")


def run_test():
    data = sample.Data()
    assert change(data) == 24
    assert data.count == 12 and data.wide == 2**40
    assert data.single == 3.25 and data.weight == 4.5
    assert data.enabled is False and data.mode == 1 and data.renamed == 13
    assert owned() == 17.5
    callback = Callback()
    source, target = sample.Data(), sample.Data()
    try:
        assert sample.consume(callback, source, target) == 25.5
        assert target.count == 3 and target.weight == 5
    finally:
        callback.close()

    for name in ('fixed', 'immutable', 'restricted', 'hidden', 'ignored'):
        namespace = {}
        exec(f'def write(data):\n    data.{name} = 0\n', namespace)
        rejects(namespace['write'], data)

    @njit
    def read_const(data):
        view = data.ConstView()
        return view.count + view.weight + view.Total()
    assert read_const(data) == data.count + data.weight + data.Total()

    def write_const(data):
        view = data.ConstView()
        view.count = 0
    rejects(write_const, data)

    def call_mutable(data):
        data.ConstView().Increment()
    rejects(call_mutable, data)

    def pass_mutable(data):
        data.TakeMutable(data.ConstView())
    rejects(pass_mutable, data)

    def cast_write(data):
        view = nsb.upcast(data.ConstView(), sample.Data)
        view.count = 0
    rejects(cast_write, data)

    def escape(data):
        return data.ConstView()
    rejects(escape, data)

    try:
        @nsb.director(state={"unused": types.intc})
        class BadCallback(sample.Callback):
            @nsb.override
            def Apply(self, source, target):
                source.count = 1
                return 0.0
    except TypingError:
        pass
    else:
        raise AssertionError("const callback member assignment compiled")
    container = sample.Container()

    @njit
    def pointer_access(container):
        assert container.null_pointer is None
        child = container.pointer
        assert child is not None
        child.count = 21
        const_child = container.const_pointee
        assert const_child is not None
        slot_child = container.ConstView().const_slot
        assert slot_child is not None
        slot_child.weight = 6.0
        return const_child.count + slot_child.weight
    assert pointer_access(container) == 27
    assert container.ReadCount() == 21

    @nsb.borrowed_from(owner_arg="container")
    @njit
    def retain_python_owner(container):
        return container.pointer
    retained = retain_python_owner(container)
    del container
    import gc
    gc.collect()
    assert retained.count == 21

    @njit
    def retain_native_owner():
        owner = sample.Container()
        child = owner.pointer
        assert child is not None
        child.count = 31
        return nsb.borrowed_from(child, owner)
    retained_native = retain_native_owner()
    gc.collect()
    assert retained_native.count == 31

    container = sample.Container()
    for field in ('pointer', 'const_pointee', 'const_slot', 'both_const', 'null_pointer'):
        namespace = {}
        exec(f'def write(obj):\n    obj.{field} = obj.pointer\n', namespace)
        rejects(namespace['write'], container)
    for field in ('const_pointee', 'both_const'):
        namespace = {}
        exec(f'def write(obj):\n    child = obj.{field}\n    if child is not None:\n        child.count = 0\n', namespace)
        rejects(namespace['write'], container)

    def opaque_write(container):
        child = container.opaque
        if child is not None:
            child.Write(9)
    rejects(opaque_write, container)

    def opaque_self_write(container):
        child = container.opaque
        if child is not None:
            child.Self().Write(9)
    rejects(opaque_self_write, container)

    def unbridged(container):
        return container.unsupported
    rejects(unbridged, container)
    print("data members: Normal End")


if __name__ == '__main__':
    run_test()
