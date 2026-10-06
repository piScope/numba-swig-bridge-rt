"""Forward C++ constructor arguments and dispatch const virtual overrides."""
import nsb_rt as nsb
from numba import njit, types
from nsb_sample4 import Evaluator, invoke_evaluate, invoke_bias
import nsb_sample4.registration as bridge

original_base_initializer = Evaluator.__init__


@nsb.director(binding=bridge.director_binding("nsb_sample4::Evaluator"),
              state={"scale": types.float64}, fallback="silent")
class ScaledEvaluator(Evaluator):
    def __init__(self, equations, dimension, scale):
        super().__init__(equations, dimension)
        self.scale = scale

    @nsb.override("Evaluate", "double")
    def evaluate(self, value):
        return self.scale * value

    def dimensions(self):
        return self.Equations(), self.Dimension()


@nsb.director(binding=bridge.director_binding("nsb_sample4::Evaluator"),
              state={"offset": types.float64}, fallback="silent")
class ExplicitEvaluator(Evaluator):
    def __init__(self, equations, dimension):
        super(ExplicitEvaluator, self).__init__(equations, dimension)
        self.offset = 1.0

    @nsb.override("Evaluate", "double")
    def evaluate(self, value):
        return value + self.offset


@nsb.director(binding=bridge.director_binding("nsb_sample4::Evaluator"),
              state={"offset": types.float64}, fallback="silent")
class AutomaticEvaluator(Evaluator):
    @nsb.override("Evaluate", "double")
    def evaluate(self, value):
        return value + self.offset


@nsb.director(binding=bridge.director_binding("nsb_sample4::Evaluator"),
              state={"offset": types.float64}, fallback="silent")
class MissingBaseInitializer(Evaluator):
    def __init__(self):
        pass

    @nsb.override("Evaluate", "double")
    def evaluate(self, value):
        return value + self.offset


@nsb.director(binding=bridge.director_binding("nsb_sample4::Evaluator"),
              state={"offset": types.float64}, fallback="silent")
class RepeatedBaseInitializer(Evaluator):
    def __init__(self):
        super().__init__(3, 2)
        super().__init__(3, 2)

    @nsb.override("Evaluate", "double")
    def evaluate(self, value):
        return value + self.offset


@njit
def native_call(proxy, value):
    return proxy.Evaluate(value) + proxy.Equations() + proxy.Dimension()


def main():
    assert Evaluator.__init__ is original_base_initializer
    evaluator = ScaledEvaluator(3, 2, 2.5)
    try:
        proxy = evaluator
        assert isinstance(evaluator, Evaluator)
        assert evaluator.as_base_proxy() is evaluator
        assert proxy.Equations() == 3
        assert proxy.Dimension() == 2
        assert evaluator.dimensions() == (3, 2)
        assert evaluator.thisown
        evaluator.thisown = True  # Repeating the current ownership state is a no-op.
        # These C++ functions dispatch through a const base reference.
        assert invoke_evaluate(proxy, 4.0) == 10.0
        assert invoke_bias(proxy, 4.0) == 9.0  # inherited const implementation
        assert native_call(proxy, 4.0) == 15.0
        evaluator.scale = 3.0
        assert invoke_evaluate(proxy, 4.0) == 12.0
    finally:
        evaluator.close()

    for operation in (lambda: evaluator.Equations(),
                      lambda: invoke_evaluate(evaluator, 1.0),
                      lambda: native_call(evaluator, 1.0)):
        try:
            operation()
        except (RuntimeError, TypeError):
            pass
        else:
            raise AssertionError("closed director was accepted")
    evaluator.close()  # repeated close is harmless

    for cls in (ExplicitEvaluator, AutomaticEvaluator):
        instance = cls(equations=3, dimension=2)
        try:
            assert instance.Equations() == 3
            assert invoke_evaluate(instance, 4.0) == (5.0 if cls is ExplicitEvaluator else 4.0)
        finally:
            instance.close()
    for cls in (MissingBaseInitializer, RepeatedBaseInitializer):
        try:
            instance = cls()
        except RuntimeError:
            pass
        else:
            instance.close()
            raise AssertionError("invalid base initialization was accepted")

    for arguments in ((), (3,), (3, 2, 1), ("bad", 2)):
        try:
            instance = ScaledEvaluator(*arguments, scale=2.5)
        except TypeError:
            pass
        else:
            instance.close()
            raise AssertionError("invalid constructor arguments were accepted")
    try:
        instance = ScaledEvaluator(0, 2, 2.5)
    except RuntimeError:
        pass
    else:
        instance.close()
        raise AssertionError("failed C++ construction was accepted")
    print("Constructor forwarding and const virtual dispatch passed")


if __name__ == "__main__":
    main()
