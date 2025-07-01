#!/usr/bin/env python3
"""
function_schema_builder.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
A lightweight utility that introspects ordinary Python functions and emits an
OpenAI-function-calling-compatible JSON schema (RFC 8259 compliant).

The module focuses on *clarity over completeness*—covering the most common
Python type annotations you'll run into when designing tool functions for
Chat Completions. It's a single-file dependency (apart from `docstring-parser`)
so you can vendor-drop it into almost any project.

Quick example
-------------
>>> from function_schema_builder import function_to_json_schema
>>> def greet(name: str, excited: bool = False):
...     "Say hello.
...     Args:
...         name: The person's name.
...         excited: End with an exclamation mark.
...     "
...     return f"Hello, {name}{'!' if excited else '.'}"
>>> import json, pprint
>>> pprint.pprint(function_to_json_schema(greet))
{'name': 'greet',
 'description': 'Say hello.',
 'parameters': {'type': 'object',
                'properties': {'name': {'type': 'string',
                                         'description': "The person's name."},
                               'excited': {'type': 'boolean',
                                           'description': 'End with an exclamation mark.',
                                           'default': False}},
                'required': ['name']}}

Install
-------
    pip install docstring-parser

Supported mappings
------------------
- ``str``   → ``{"type": "string"}``
- ``int``   → ``{"type": "integer"}``
- ``float`` → ``{"type": "number"}``
- ``bool``  → ``{"type": "boolean"}``
- ``list[T]`` / ``List[T]`` → ``{"type": "array", "items": schema(T)}``
- ``dict`` / ``Dict``      → ``{"type": "object"}``
- ``typing.Literal``       → ``{"type": "string", "enum": [...]}``
- ``enum.Enum`` subclasses → ``{"type": "string", "enum": [...]}``
If a parameter lacks an annotation we assume ``string``.

"""
from __future__ import annotations

import inspect
import textwrap
import typing as _t
from enum import Enum
from pathlib import Path

try:
    import docstring_parser as _docparse  # type: ignore
except ImportError as exc:
    raise SystemExit("docstring-parser is required. pip install docstring-parser") from exc

JSONSchema = dict[str, _t.Any]

# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _python_type_to_schema(py_type: _t.Any) -> JSONSchema:
    """Convert a Python/typing annotation to a draft-07 JSON Schema snippet."""
    origin = _t.get_origin(py_type)
    args = _t.get_args(py_type)

    # Optional[T] == Union[T, None]
    if origin is _t.Union and type(None) in args:
        non_none = [a for a in args if a is not type(None)][0]
        return _python_type_to_schema(non_none)

    # Literal
    if origin is _t.Literal:  # type: ignore[attr-defined]
        return {"type": "string", "enum": list(args)}

    # Containers
    if origin in {list, _t.List}:  # type: ignore[attr-defined]
        item_type = args[0] if args else str
        return {"type": "array", "items": _python_type_to_schema(item_type)}
    if origin in {dict, _t.Dict}:  # type: ignore[attr-defined]
        return {"type": "object"}

    # Built-ins
    if py_type is str:
        return {"type": "string"}
    if py_type is int:
        return {"type": "integer"}
    if py_type is float:
        return {"type": "number"}
    if py_type is bool:
        return {"type": "boolean"}

    # Enums
    if inspect.isclass(py_type) and issubclass(py_type, Enum):
        return {"type": "string", "enum": [e.value for e in py_type]}

    # Fallback
    return {"type": "string"}


def _extract_param_descriptions(docstring: str | None) -> dict[str, str]:
    """Parse *Google/NumPy/ReST* style docstring and return {param: description}."""
    if not docstring:
        return {}
    try:
        parsed = _docparse.parse(docstring)
    except Exception:
        return {}
    return {p.arg_name: p.description or "" for p in parsed.params}


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def function_to_json_schema(func: _t.Callable[..., _t.Any]) -> JSONSchema:
    """Return an OpenAI-compatible function schema for *func* using its signature."""

    sig = inspect.signature(func)
    type_hints = _t.get_type_hints(func)
    param_docs = _extract_param_descriptions(inspect.getdoc(func))

    properties: dict[str, JSONSchema] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name == "self":
            continue  # skip methods
        annotation = type_hints.get(name, str)
        schema_piece = _python_type_to_schema(annotation)
        # description from docstring if present
        if desc := param_docs.get(name):
            schema_piece["description"] = " ".join(desc.split())  # single-line
        # default handling
        if param.default is not inspect.Parameter.empty:
            schema_piece["default"] = param.default
        else:
            required.append(name)
        properties[name] = schema_piece

    func_doc = inspect.getdoc(func) or ""
    func_first_line = textwrap.dedent(func_doc).splitlines()[0] if func_doc else ""

    return {
        "name": func.__name__,
        "description": func_first_line,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required or None,  # omit if empty for cleaner JSON
        },
    }


# -----------------------------------------------------------------------------
# CLI helper (python -m function_schema_builder path/to/module.py Object.func)
# -----------------------------------------------------------------------------

def _load_callable_from_string(qualname: str):
    """Import *qualname* (``some.module:func`` or ``some.module:Class.method``)."""
    if ":" in qualname:
        mod_path, obj_path = qualname.split(":", 1)
    else:
        raise SystemExit("Please use the format module.py:func")
    # Allow relative file paths like ./foo.py
    mod_path = str(Path(mod_path).with_suffix("").resolve())
    # Turn /path/to/pkg/module into pkg.module for import
    if mod_path.endswith("/__init__"):
        mod_import = Path(mod_path).parent.name
    else:
        mod_import = ".".join(Path(mod_path).parts[-Path(mod_path).parts[::-1].index("__pycache__")-1:]) if "__pycache__" in mod_path else mod_path.replace("/", ".")
    module = __import__(mod_import, fromlist=["*"])
    obj = module
    for attr in obj_path.split('.'):
        obj = getattr(obj, attr)
    if not callable(obj):
        raise SystemExit(f"{qualname} is not callable")
    return obj


def _cli() -> None:
    import argparse, json

    parser = argparse.ArgumentParser(description="Generate JSON schema for a Python callable.")
    parser.add_argument("callable", help="Target callable in the form module.py:func or module.py:Class.method")
    parser.add_argument("--outfile", help="Save schema JSON to file instead of stdout")
    args = parser.parse_args()

    func = _load_callable_from_string(args.callable)
    schema = function_to_json_schema(func)
    if args.outfile:
        Path(args.outfile).write_text(json.dumps(schema, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(schema, indent=2, ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover
    _cli()
