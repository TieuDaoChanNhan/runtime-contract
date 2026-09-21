import inspect
import textwrap
from dataclasses import dataclass
from types import UnionType
from typing import Any, Protocol, Union, get_args, get_origin

ALLOWED_TYPES = (
    list[dict[str, int | list[int]]]
    | dict[int, list[int]]
    | dict[str, Any]
    | str
    | bool
    | int
    | float
    | dict[str, str]
    | dict
    | list[int]
    | dict[str, bool | str | int]
    | dict[str, bool | str | int | None]
    | dict[str, int | list[str] | dict | None]
)


@dataclass(frozen=True)
class Arg:
    type: str | bool | int
    doc: str
    is_required: bool
    default_value: ALLOWED_TYPES | None

    def python_annotation(self) -> str:
        parts: list[str] = []
        if not self.is_required and self.default_value is None:
            parts.append(f"{self.type.__name__} | None")  # type: ignore
        else:
            parts.append(f"{self.type.__name__}")  # type: ignore

        if not self.is_required:
            parts.append(f" = {self.default_value}")
        return "".join(parts)


class ToolException(Exception): ...


class ToolInvocationException(ToolException):
    def __init__(self, message: str):
        super().__init__()
        self.message = message

    def __str__(self):
        return "ToolInvocationException:\n" + self.message


class ToolRuntimeException(ToolException):
    def __init__(self, message: str):
        super().__init__()
        self.message = message

    def __str__(self):
        return f"ToolRuntimeException: {self.message}"


class Tool(Protocol):
    name: str
    doc: str
    arg_doc: dict[str, str]
    inputs: dict[str, Arg]
    output_type: Any

    def __init__(self):
        self.inputs = {}

        allowed_types_set = get_args(ALLOWED_TYPES)

        sig = inspect.signature(self.run)
        func_params = {name: param for name, param in sig.parameters.items()}
        for name, param in func_params.items():
            if param.annotation == inspect.Parameter.empty:
                raise ValueError(f"Argument {name} of `run` is not type annotated")

            if (
                get_origin(param.annotation) is UnionType
                or get_origin(param.annotation) is Union
            ) and type(None) in get_args(param.annotation):
                non_none_types = [
                    arg for arg in get_args(param.annotation) if arg is not type(None)
                ]
                if len(non_none_types) != 1:
                    raise ValueError(
                        f"Argument {name} of `run` is a non-optional union. Only one non-None type is allowed."
                    )
                param_type = non_none_types[0]
            else:
                param_type = param.annotation

            if param_type not in allowed_types_set:
                raise ValueError(
                    f"Argument {name} of `run` should be one of: {ALLOWED_TYPES}, got: {param_type} in tool {self.name}"
                )

            self.inputs[name] = Arg(
                param_type,
                self.arg_doc[name],
                is_required=param.default == inspect.Parameter.empty,
                default_value=param.default
                if param.default != inspect.Parameter.empty
                else None,
            )

        self.output_type = inspect.signature(self.run).return_annotation
        if self.output_type not in allowed_types_set and self.output_type is not None:
            raise ValueError(
                f"Return type of `run` should be one of: {ALLOWED_TYPES}, got {self.output_type} for tool {self.name}"
            )

    async def run(self, *args, **kwargs) -> Any: ...

    async def __call__(self, *args, **kwargs) -> Any:
        func_params = {
            name: param
            for name, param in inspect.signature(self.run).parameters.items()
        }

        try:
            bound_args = inspect.signature(self.run).bind_partial(*args, **kwargs)
            bound_args.apply_defaults()
        except Exception as e:
            raise ToolInvocationException(str(e)) from e

        errors: list[str] = []
        for name, value in bound_args.arguments.items():
            if name in func_params:
                param = func_params[name]
                annot = param.annotation
                origin = get_origin(annot)

                # 1. Build a safe tuple of base types to check against
                if origin is UnionType or origin is Union:
                    # Strip generics from inside the Union (e.g., list[int] | None -> (list, type(None)))
                    check_types = tuple(
                        get_origin(arg) or arg for arg in get_args(annot)
                    )
                else:
                    # Strip generic from single type (e.g., list[int] -> list)
                    check_types = origin or annot

                # 2. Safely perform the isinstance check
                if not isinstance(value, check_types):
                    # 3. Safely format the error string (avoiding __name__ on generics)
                    type_str = getattr(annot, "__name__", str(annot))
                    errors.append(
                        f"Argument {name} should be of type {type_str}. Got {value.__class__.__name__} instead."
                    )
            else:
                errors.append(f"Unexpected argument {name}")

        for name, param in func_params.items():
            if param.default is param.empty and name not in bound_args.arguments:
                errors.append(f"Missing required argument {name}")

        if errors:
            raise ToolInvocationException(",".join(errors))

        try:
            return await self.run(**bound_args.arguments)
        except Exception as e:
            raise ToolRuntimeException(str(e)) from e

    def full_python_doc(self) -> str:
        arg_str = ", ".join(
            f"{name}: {arg.python_annotation()}" for name, arg in self.inputs.items()
        )
        output = self.output_type.__name__ if self.output_type is not None else "None"
        signature = f"def {self.name}({arg_str}) -> {output}"

        docs: list[str] = ['"""', self.doc, "", "Args:"]
        for name, arg in self.inputs.items():
            docs.append(f"  {name}: {arg.doc}")
        docs.append('"""')

        doc = textwrap.indent("\n".join(docs), "  ")

        return signature + "\n" + doc
