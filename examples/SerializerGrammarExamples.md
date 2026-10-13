# Canonical Model Descriptions

## Direct Use

```python
import json
import numpy as np
from sklearn.linear_model import LinearRegression
from mnplib import describe_model, Surfeit
from mnplib.models.language import execute, parse

X = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
y = 1 + 2 * X[:, 0] - 3 * X[:, 1]
model = LinearRegression().fit(X, y)
description = describe_model(model, feature_names=["length", "width"])

print(description.canonical)
# 1.00e+00+2.00e+00*x0-3.00e+00*x1

predictions = execute(description.ast, X)
np.testing.assert_allclose(predictions, model.predict(X))
assert parse(description.canonical) == description.ast
payload = json.dumps(description.to_dict(), allow_nan=False)

metric = Surfeit().fit(X, y)
assert metric.surfeit_model(model) == metric.surfeit_string(description.canonical)
```

`describe_model` is available from `mnplib` and `mnplib.models`.
It validates fitted estimators but does not compute predictions. AutoML
estimators expose the selected description through the same function.
Evaluation artifacts pair a `description` with `subset` and `predictions`;
their `model_string` property is exactly `description.canonical`.

## Language

The authoritative Lark grammar is
`mnplib/models/language/model.lark`. A description is one expression.
It supports arithmetic, unary negation, exponentials, comparisons, Boolean
conjunction, conditional expressions, vectors, and bounded iteration.
Canonical feature references are `x0`,
`x1`, and so on. Arithmetic binds more tightly than comparisons, followed by
`and`, then conditional expressions. Conjunction evaluates to a Boolean vector.

The immutable semantic nodes are Constant, Label, Feature, Negate, Exp, Binary,
And, Conditional, Vector, Element, Index, Sum, and Comprehension. Binary operations export
individual semantic names such as `add` and `less_equal`.

Vector access and exponentials have fixed meanings:

- `at(vector,index)`: select one element using a non-negative integer or a bound
  iteration index. Nested access selects entries in explicit parameter matrices.
- `exp(expression)`: the natural exponential, applied elementwise.

`label(encoded_json_string)` represents a categorical scalar exactly.
For example, `label("123456")` preserves an integer category, while
`label("\\"species A\\"")` preserves a string category. Labels are identities,
not numerical model parameters, and are not rounded.

No arbitrary functions, attributes, imports, assignments, mutation, loop
statements, arbitrary iterables, or Python evaluation are accepted. The interpreter dispatches
only declared node types. Feature and vector indices are bounds checked.
Conditional branches evaluate only on the observations taking that branch.
External strings always pass through Lark before execution.

### Bounded Iteration

`Sum(index, size, expression)` performs repeated accumulation. Its canonical
form is `sum(expression for i in range(N))`. `Comprehension` performs repeated
vector construction and renders as `[expression for j in range(N)]`.
`N` must be a literal non-negative integer; dynamic bounds, multiple range
arguments, filters, and arbitrary collections are not accepted.

Each iteration introduces exactly one integer index in its expression.
Names match `[a-z][a-z0-9_]*`, excluding Python keywords, language function names,
and feature tokens such as `x0`. Indices must be bound by an enclosing iteration.
Nested reuse of an index name is rejected; sibling expressions may reuse names.
No other variable binding is available. Bounds and index values are exact
integers, unaffected by parameter quantization.

For example, this self-contained expression multiplies two explicit weights
by two input features:

```python
from mnplib.models.language import execute, parse, render

text = "sum(at([2.00e+00,3.00e+00],i)*at([x0,x1],i) for i in range(2))"
expression = parse(text)
assert render(expression) == text
assert execute(expression, [[1., 2.]]).tolist() == [8.0]
```

Nested `Sum` and `Comprehension` express whole layers. With `w`, `b`, and `x`
standing for explicit vectors, the structure is:

```text
[at(b,j)+sum(at(at(w,j),i)*at(x,i) for i in range(3)) for j in range(2)]
```

Here `w` is output-by-input. Canonical text contains the actual parameter
vectors and feature expressions in place of these explanatory names.
Execution returns one result per observation; each comprehension adds a vector
axis after the observation axis. `at` selects along that axis and evaluates
only the requested element of a vector or comprehension. A sum accumulates in
increasing index order. A zero-size sum returns zero; a zero-size comprehension
returns an empty vector. Bounds are finite but are not a runtime resource quota.

The Theory of Nescience treats descriptions as executable procedures. Bounded
iteration expresses genuine algorithmic regularity without repeating every
operation in the text. The fitted parameters remain explicit; only repeated
computational structure is compact. Surfeit measures this canonical program,
without metadata or application formatting.

### Neural Networks

Every network layer is a `Comprehension`, and each neuron accumulates its
weighted inputs with `Sum`. Layer-specific names `i0`, `j0`, `i1`, `j1`, and so
on keep nested scopes distinct. This representation is deterministic and does
not depend on which alternative string happens to be shorter.

Hidden identity outputs use the score itself. ReLU uses `z if z>0 else 0`.
Sigmoid uses arithmetic equivalent to `1/(1+exp(-z))`, and tanh uses arithmetic
equivalent to `(exp(2*z)-1)/(exp(2*z)+1)`. The serialized expressions branch on
the sign of `z` to keep exponential arguments non-positive. `z` is an
explanatory placeholder for the full neuron expression, not a local variable.
Regression returns the numeric output. Binary classification thresholds the
final score at zero; multiclass classification compares the final scores
directly without computing probabilities.

### Multiclass Decisions

Class prediction uses explicit comparisons and nested conditional expressions.
For three class scores `z0`, `z1`, `z2` and corresponding labels `c0`, `c1`, `c2`,
the decision rule is:

```text
c0 if z0>=z1 and z0>=z2 else c1 if z1>=z2 else c2
```

These names are explanatory placeholders: canonical text substitutes each full
score expression and encodes each categorical value with `label(...)`.
Earlier classes use `>=` against every later score, so ties select the first
class. The description repeats the score expressions explicitly; it contains
no assignments or hidden class-selection operation.

Logistic regression uses linear class scores directly: the largest softmax
probability has the same index as the largest score
(`argmax softmax(z) == argmax z`). Computing exponentials or probabilities is
therefore unnecessary for class prediction. Binary logistic regression uses
`c1 if z>0 else c0`.

## Canonicalization

Numerical model parameters are rounded to **three significant digits** before
execution and exported in scientific notation: `1.00e+00`, `-4.97e+01`,
`0.00e+00`. Negative zero becomes zero. Non-finite parameters are rejected.
Exponents have at least two digits; `1.00e-100` retains the full exponent.
Feature and vector indices are exact structural integers.

The AST, text, parsed AST, interpreter, and JSON all use the same rounded
parameters. There is no hidden higher-precision prediction path in the language.

Normalization is a single bottom-up pass. It removes multiplication by one or
zero, addition of zero, subtraction of zero, and double negation. Addition of a
negative constant becomes subtraction. Conjunctions with one item reduce to
that item; nested conjunctions flatten in their original order. Zero-size sums
and comprehensions reduce to zero and an empty vector after validating their
bodies. Normalization
does not reorder arithmetic terms or comparisons, expand expressions, optimize
conditions, unroll iteration, or perform symbolic algebra.
Parentheses preserve evaluation order and are omitted when precedence permits.

### Numerical Accuracy

Serialization describes a quantized model, not a bit-identical copy of the
estimator. Execution from its AST and from its canonical text must agree.
Tests compare representative regression models with `rtol=0.02, atol=0.03`;
these are test tolerances, **not a universal error bound**. Cancellation, large
inputs, and deep networks can amplify rounding. Classification tests require
exact labels on the tested inputs, but rounding can change decisions near a
tree threshold or class-score boundary. For threshold-equality semantics,
the left tree branch uses `<=`.

Use the fitted estimator for full-precision production predictions.
AutoML's inaccuracy continues to use estimator predictions; surfeit uses the
canonical description. Changes to canonical text therefore change description
length and potentially surfeit and model ranking.

## Supported Families

- LinearRegression: affine arithmetic for a one-dimensional target.
- LogisticRegression and LinearSVC: binary score threshold or explicit nested
  class-score comparisons.
- LinearSVR: affine arithmetic.
- DecisionTreeClassifier and DecisionTreeRegressor: nested conditionals,
  including scalar and multi-output leaves and actual class labels.
- GaussianNB: arithmetic log-class scores with precomputed logarithmic constants.
- MLPClassifier and MLPRegressor: bounded neuron sums and layer comprehensions, including
  multi-output regression. Multilabel classification is unsupported.
- Finite-window autoregression, moving averages, and exponential smoothing:
  affine arithmetic over lag columns.

AutoML StandardScaler preprocessing is represented by feature expressions
`(xj-mean)/scale`, not by a text prefix. Original column indices remain
available after subset selection.

ARIMA and structural state-space model serialization are **not supported by
schema version 3**. These require stateful filter/forecast semantics, including
initialization and observation updates. Their searchers produce an
`unsupported_model_description` diagnostic and do not create scored candidates.
A search containing only those families raises a clear error. A mixed search
continues with supported families. Tree ensembles and nonlinear SVMs are also
outside the supported serializer registry.

## Semantic JSON

`description.to_dict()` returns JSON-compatible data:

```json
{
  "schema": "mnplib-model",
  "schema_version": 3,
  "model_type": "LinearRegression",
  "ast": {
    "type": "multiply",
    "left": {"type": "constant", "value": 2.0},
    "right": {"type": "feature", "index": 0, "name": "length"}
  },
  "canonical": "2.00e+00*x0",
  "feature_indices": [0],
  "feature_names": ["length"],
  "metadata": {"features_used": [0], "n_terms": 1}
}
```

Node type names are semantic, not Python class or module names.
Conjunction exports as `{"type": "and", "items": [...]}`, preserving condition
order. Multiclass score-selection expressions contain `conditional`,
`greater_equal`, `and`, and `label` nodes surrounding their score expressions.
Iteration nodes use `{"type": "sum", "index": "i", "size": 3, "expression": ...}`
and the corresponding `comprehension` type. Bound references use
`{"type": "index", "name": "i"}`. An `element` index is either an integer or
an `index` node. Exponentials use `{"type": "exp", "operand": ...}`.
`ModelDescription.from_dict(payload)` validates the schema, nodes, and
agreement with any supplied canonical text. An incompatible schema change
requires a schema-version increment.

Human-readable names are metadata and never alter canonical text. Tree
metadata includes depth, node count, and leaf count; linear metadata includes
term count. Network parameters appear as ordinary constants and vectors in
their executable expressions.
Prediction vectors are not part of model JSON.

mnplib owns semantic ASTs and computational metadata. Cajal and other
applications consume this JSON and own all presentation decisions; they need
neither sklearn objects nor a canonical-text parser.
