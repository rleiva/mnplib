"""Version-independent semantic node encoding with explicit type validation."""

from .nodes import (
    Constant, Label, Feature, Negate, Binary, Conditional,
    Vector, Element, Classify, Dense,
)
from .normalize import normalize

_NAMES = {"+": "add", "-": "subtract", "*": "multiply", "/": "divide",
          "**": "power", "<": "less", "<=": "less_equal", ">": "greater",
          ">=": "greater_equal", "==": "equal"}


def node_to_dict(node):
    """Export only JSON values and stable semantic names."""
    node = normalize(node)
    kind = type(node)
    if kind is Constant:
        return {"type": "constant", "value": node.value}
    if kind is Label:
        return {"type": "label", "value": node.value}
    if kind is Feature:
        return {"type": "feature", "index": node.index, "name": node.name}
    if kind is Negate:
        return {"type": "negate", "operand": node_to_dict(node.operand)}
    if kind is Binary:
        return {"type": _NAMES[node.operator], "left": node_to_dict(node.left),
                "right": node_to_dict(node.right)}
    if kind is Conditional:
        return {"type": "conditional", "condition": node_to_dict(node.condition),
                "if_true": node_to_dict(node.if_true), "if_false": node_to_dict(node.if_false)}
    if kind is Vector:
        return {"type": "vector", "items": [node_to_dict(x) for x in node.items]}
    if kind is Element:
        return {"type": "element", "vector": node_to_dict(node.vector), "index": node.index}
    if kind is Classify:
        return {"type": "classify", "scores": node_to_dict(node.scores),
                "labels": [node_to_dict(x) for x in node.labels]}
    if kind is Dense:
        return {"type": "dense", "inputs": node_to_dict(node.inputs),
                "weights": [[x.value for x in row] for row in node.weights],
                "bias": [x.value for x in node.bias], "activation": node.activation}
    raise TypeError(f"Unsupported model node: {kind.__name__}")


def node_from_dict(data):
    """Decode the closed semantic vocabulary without resolving Python objects."""
    if not isinstance(data, dict):
        raise ValueError("A semantic node must be a dictionary.")
    kind = data.get("type")
    try:
        if kind in ("constant", "label"):
            node = (Constant if kind == "constant" else Label)(data["value"])
        elif kind == "feature":
            node = Feature(data["index"], data.get("name"))
        elif kind == "negate":
            node = Negate(node_from_dict(data["operand"]))
        elif kind in _NAMES.values():
            op = next(op for op, name in _NAMES.items() if name == kind)
            node = Binary(op, node_from_dict(data["left"]), node_from_dict(data["right"]))
        elif kind == "conditional":
            node = Conditional(*(node_from_dict(data[key]) for key in
                                 ("condition", "if_true", "if_false")))
        elif kind == "vector":
            node = Vector(tuple(node_from_dict(x) for x in data["items"]))
        elif kind == "element":
            node = Element(node_from_dict(data["vector"]), data["index"])
        elif kind == "classify":
            node = Classify(node_from_dict(data["scores"]),
                            tuple(node_from_dict(x) for x in data["labels"]))
        elif kind == "dense":
            node = Dense(node_from_dict(data["inputs"]),
                         tuple(tuple(Constant(x) for x in row) for row in data["weights"]),
                         tuple(Constant(x) for x in data["bias"]), data["activation"])
        else:
            raise ValueError(f"Unsupported semantic node: {kind}")
        return normalize(node)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Invalid {kind!r} semantic node.") from exc
