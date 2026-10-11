"""Protected constructor forwarding with runtime-compiled user overrides."""

from numba import types
import nsb_rt as nsb
import nsb_sample6 as sample
import nsb_sample6.registration  # noqa: F401


@nsb.director(state={"bias": types.float64})
class Evaluator(sample.ProtectedEvaluator):
    def __init__(self, *args):
        super().__init__(*args)
        self.bias = self.Bias()

    @nsb.override
    def Evaluate(self, value):
        return value + self.bias


@nsb.director(state={"factor": types.float64})
class Concrete(sample.ProtectedConcrete):
    def __init__(self, value=3):
        super().__init__(value)
        self.factor = value

    @nsb.override
    def Evaluate(self, value):
        return value * self.factor


@nsb.director(state={"unused": types.intc}, fallback="silent")
class DefaultConcrete(sample.ProtectedConcrete):
    def __init__(self):
        super().__init__()


def run_test():
    instances = [
        Evaluator(),
        Evaluator(2, 1.5, sample.ProtectedEvaluator.ADD, True, 4),
        Evaluator(2, 1.5, sample.ProtectedEvaluator.MULTIPLY, False, 4),
        Concrete(5),
        DefaultConcrete(),
    ]
    try:
        assert sample.invoke_protected(instances[0], 2.0) == 9.0
        assert sample.invoke_protected(instances[1], 2.0) == 9.5
        assert sample.invoke_protected(instances[2], 2.0) == 2.0
        assert sample.invoke_concrete(instances[3], 2.0) == 10.0
        assert sample.invoke_concrete(instances[4], 2.0) == 5.0
        try:

            @nsb.director(state={"unused": types.intc})
            class Missing(sample.ProtectedEvaluator):
                pass

        except TypeError as error:
            assert "override" in str(error)
        else:
            raise AssertionError("missing pure virtual override accepted")
        for base in (sample.ProtectedEvaluator, sample.ProtectedConcrete):
            try:
                base()
            except (TypeError, AttributeError):
                pass
            else:
                raise AssertionError("protected base constructor exposed")
        print("protected constructors: Normal End")
    finally:
        for instance in instances:
            instance.close()


if __name__ == "__main__":
    run_test()
