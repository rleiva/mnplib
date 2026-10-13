"""Version-independent semantic node encoding with explicit type validation."""

from .nodes import (
    Constant, Label, Feature, Negate, Binary, And, Conditional,
    Vector, Element, Index, Sum, Comprehension, Exp,
)
from .normalize import normalize

_NAMES = {"+": "add", "-": "subtract", "*": "multiply", "/": "divide",
          "**": "power", "<": "less", "<=": "less_equal", ">": "greater",
          ">=": "greater_equal", "==": "equal"}


def node_to_dict(node):
    """Export only JSON values and stable semantic names."""
    return _node_to_dict(normalize(node))


def _node_to_dict(node):
    kind = type(node)
    if kind is Constant:
        return {"type": "constant", "value": node.value}
    if kind is Label:
        return {"type": "label", "value": node.value}
    if kind is Feature:
        return {"type": "feature", "index": node.index, "name": node.name}
    if kind is Index:
        return {"type": "index", "name": node.name}
    if kind in (Sum, Comprehension):
        return {"type": "sum" if kind is Sum else "comprehension",
                "index": node.index, "size": node.size,
                "expression": _node_to_dict(node.expression)}
    if kind is Exp:
        return {"type": "exp", "operand": _node_to_dict(node.operand)}
    if kind is Negate:
        return {"type": "negate", "operand": _node_to_dict(node.operand)}
    if kind is Binary:
        return {"type": _NAMES[node.operator], "left": _node_to_dict(node.left),
                "right": _node_to_dict(node.right)}
    if kind is And:
        return {"type": "and", "items": [_node_to_dict(x) for x in node.items]}
    if kind is Conditional:
        return {"type": "conditional", "condition": _node_to_dict(node.condition),
                "if_true": _node_to_dict(node.if_true), "if_false": _node_to_dict(node.if_false)}
    if kind is Vector:
        return {"type": "vector", "items": [_node_to_dict(x) for x in node.items]}
    if kind is Element:
        index = _node_to_dict(node.index) if type(node.index) is Index else node.index
        return {"type": "element", "vector": _node_to_dict(node.vector), "index": index}
    raise TypeError(f"Unsupported model node: {kind.__name__}")


def node_from_dict(data):
    """Decode the closed semantic vocabulary without resolving Python objects."""
    return normalize(_node_from_dict(data))


def _node_from_dict(data):
    if not isinstance(data, dict):
        raise ValueError("A semantic node must be a dictionary.")
    kind = data.get("type")
    try:
        if kind in ("constant", "label"):
            node = (Constant if kind == "constant" else Label)(data["value"])
        elif kind == "feature":
            node = Feature(data["index"], data.get("name"))
        elif kind == "index":
            node = Index(data["name"])
        elif kind in ("sum", "comprehension"):
            node = (Sum if kind == "sum" else Comprehension)(
                data["index"], data["size"], _node_from_dict(data["expression"]))
        elif kind == "exp":
            node = Exp(_node_from_dict(data["operand"]))
        elif kind == "negate":
            node = Negate(_node_from_dict(data["operand"]))
        elif kind in _NAMES.values():
            op = next(op for op, name in _NAMES.items() if name == kind)
            node = Binary(op, _node_from_dict(data["left"]), _node_from_dict(data["right"]))
        elif kind == "and":
            node = And(tuple(_node_from_dict(x) for x in data["items"]))
        elif kind == "conditional":
            node = Conditional(*(_node_from_dict(data[key]) for key in
                                 ("condition", "if_true", "if_false")))
        elif kind == "vector":
            node = Vector(tuple(_node_from_dict(x) for x in data["items"]))
        elif kind == "element":
            index = data["index"]
            if isinstance(index, dict):
                index = _node_from_dict(index)
            node = Element(_node_from_dict(data["vector"]), index)
        else:
            raise ValueError(f"Unsupported semantic node: {kind}")
        return node
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Invalid {kind!r} semantic node.") from exc
