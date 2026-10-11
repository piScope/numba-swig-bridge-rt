"""Resolve director object-field policy before compiling native callbacks."""

from numba import types
from numba.core import compiler, ir, typed_passes
from numba.core.inline_closurecall import inline_closure_call
from numba.core.errors import TypingError
from numba.core.registry import cpu_target
from numba.core.datamodel import models
from numba.extending import register_model
from numba.core.typing.templates import AttributeTemplate, infer_getattr

from . import ordinary, director_state


class InferenceState(types.Type):
    def __init__(self, fields):
        self.fields = dict(fields)
        super().__init__(name=f"nsb.state_inference[{tuple(fields.items())!r}]")

    @property
    def key(self):
        return tuple(self.fields.items())


@register_model(InferenceState)
class InferenceModel(models.OpaqueModel):
    """Opaque placeholder permits typing-only intrinsic helpers such as occupancy.

    No attribute lowering is installed; state mutation in separately compiled
    helpers therefore cannot silently bypass constraint collection.
    """


@infer_getattr
class StateAttributes(AttributeTemplate):
    key = InferenceState

    def generic_resolve(self, state, field):
        return state.fields.get(field)


def _inline_requested_helpers(function_ir):
    """Include always-inlined helper assignments in the analyzed IR."""
    def resolve(variable):
        definition = function_ir.get_definition(variable)
        if isinstance(definition, (ir.Global, ir.FreeVar)):
            return definition.value
        if isinstance(definition, ir.Expr) and definition.op == "getattr":
            return getattr(resolve(definition.value), definition.attr)
        return None

    for _ in range(100):
        changed = False
        for block in list(function_ir.blocks.values()):
            for index, node in enumerate(block.body):
                if not (isinstance(node, ir.Assign) and isinstance(node.value, ir.Expr)
                        and node.value.op == "call"):
                    continue
                try:
                    helper = resolve(node.value.func)
                except (KeyError, AttributeError):
                    continue
                if getattr(helper, "targetoptions", {}).get("inline") != "always":
                    continue
                inline_closure_call(function_ir, helper.py_func.__globals__, block,
                                    index, helper.py_func)
                changed = True
                break
            if changed:
                break
        if not changed:
            return
    raise TypingError("state inference exceeded the inlined helper limit (possible recursion)")


def infer_object_fields(binding, overrides, scalar_fields, object_fields, array_fields):
    """Inspect uncoerced typed SetAttr RHS values and iterate const constraints.

    The analysis type has no native representation: passing it to a separately
    compiled helper is unsupported rather than silently omitting its effects.
    """
    policies = {}
    context = cpu_target.typing_context
    context.refresh()
    for iteration in range(len(object_fields) + 2):
        fields = {name: director_state.scalar_numba_type(kind)
                  for name, kind in scalar_fields.items()}
        fields.update(array_fields)
        for name, bridge in object_fields.items():
            mode, const = policies.get(name, (None, False))
            if mode == "borrowed":
                fields[name] = ordinary.bridge_object_type(
                    binding.bridge_id, bridge.cxx_type, const=const, force_const=True,
                )
            else:
                fields[name] = ordinary.BorrowedReturnBridgeObjectType(
                    binding.bridge_id, bridge.cxx_type,
                )
        state = InferenceState(fields)
        found = dict(policies)
        errors = []
        for method in binding.methods:
            function = overrides.get(method.identity)
            if function is None:
                continue
            arguments = []
            for parameter in method.parameters:
                kind = parameter["kind"]
                if kind == "object":
                    argument = ordinary.bridge_object_type(
                        binding.bridge_id, parameter["cxx_type"],
                        const=parameter.get("const", False), force_const=True,
                    )
                elif kind == "pointer":
                    argument = types.CPointer({"float32": types.float32,
                        "float64": types.float64, "int32": types.int32,
                        "bool": types.boolean}[parameter["numpy_dtype"]])
                else:
                    argument = {"float": types.float32, "double": types.float64,
                        "int": types.intc, "int64": types.int64,
                        "bool": types.boolean, "enum": types.intc}[kind]
                arguments.append(argument)
            function_ir = compiler.run_frontend(function)
            _inline_requested_helpers(function_ir)
            try:
                result = typed_passes.type_inference_stage(
                    context, cpu_target.target_context, function_ir,
                    (state, *arguments), None,
                )
            except Exception as error:
                errors.append((method.identity, error))
                continue
            for block in function_ir.blocks.values():
                for node in block.body:
                    if not isinstance(node, ir.SetAttr) or node.attr not in object_fields:
                        continue
                    if result.typemap[node.target.name] != state:
                        continue
                    value = result.typemap[node.value.name]
                    if isinstance(value, ordinary.OwnedReturnBridgeObjectType):
                        raise TypingError(f"state field {node.attr!r}: own_this is a Python-bound transfer")
                    if isinstance(value, ordinary.OwnedBridgeObjectType):
                        mode, const = "owned", False
                    elif isinstance(value, ordinary.BorrowedReturnBridgeObjectType):
                        # A state read may carry either a Python or NRT owner;
                        # it is not an explicit externally guaranteed borrow.
                        errors.append((method.identity, TypingError(
                            f"cannot determine ownership of state field {node.attr!r} "
                            "from an ownership-bearing state read")))
                        continue
                    elif isinstance(value, ordinary.BridgeObjectType):
                        mode = "borrowed"
                        const = isinstance(value, ordinary.ConstBridgeObjectType)
                    else:
                        raise TypingError(f"cannot determine ownership of state field {node.attr!r}: "
                                          "narrow nullable values with require_not_none before assignment")
                    bridge = object_fields[node.attr]
                    if value.bridge_id != binding.bridge_id or value.cxx_type != bridge.cxx_type:
                        raise TypingError(f"state field {node.attr!r} requires {bridge.cxx_type}; "
                                          "use an explicit bridge cast")
                    previous = found.get(node.attr)
                    if previous and previous[0] != mode:
                        raise TypingError(f"state field {node.attr!r} receives both borrowed and owned values")
                    found[node.attr] = (mode, const or bool(previous and previous[1]))
        if found == policies:
            if errors:
                identity, error = errors[0]
                raise TypingError(f"state ownership inference failed in {identity}: {error}") from error
            return policies
        policies = found
    raise TypingError("state ownership/const inference did not converge")
