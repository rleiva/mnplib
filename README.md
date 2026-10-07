## Machine Learning with the Minimum Nescience Principle

`mnplib` is an open-source Python library for machine learning based on the **Minimum Nescience Principle**. It is built on top of [scikit-learn](https://scikit-learn.org/stable/) and [statsmdodels](https://www.statsmodels.org/) and provides tools for evaluating datasets, models, and automated machine-learning candidates through the lens of nescience.

The library is based on the [*Minimum Nescience Principle*](https://www.amazon.com/dp/B0GZH1FZ9J), a mathematical framework for measuring how well a problem is understood given a **representation** and a **description**. In machine learning, the representation is usually a dataset, while the description is a fitted model.

`mnplib` is currently focused on three main goals:

* measuring the quality of data representations;
* measuring the quality and complexity of fitted models;
* exploring nescience-guided alternatives to conventional AutoML.

> **Development status**
>
> `mnplib 2` is an alpha release. The API, documentation, examples, and automated searchers are still under active development. The package is suitable for experimentation and testing, but it should not yet be considered production-ready.

### Installation

Install the development API from this checkout:

```bash
pip install -e .
```

The library returns numerical metrics and factual diagnostics. Applications own interpretation thresholds, qualitative labels, recommendations, and explanatory wording. Reliability decisions and machine-readable failure codes remain part of the numerical API.

Use `Nescience.analysis(subset=..., predictions=..., model_string=...)` for explicit artifacts, `model_analysis(model)` for fitted estimators, and `analysis()` on `NescienceClassifier`, `NescienceRegressor`, `TimeSeries`, or `ResidualAnalysis`. These reports do not assign qualitative profiles. Canonical model descriptions remain in mnplib because their compressed lengths determine surfeit.

`Mismodel` evaluates predictions and a model description through inaccuracy and surfeit. `from mnplib.mismodel import mismodel` provides the scalar `mismodel(inaccuracy=..., surfeit=...)` helper when these components are already available. Both interfaces use their equal-weight root mean square. Mismodel is also included in nescience analysis reports and AutoML result dataframes.

### Residual and Correction Analysis

`ResidualAnalysis` analyzes explicit predictions without training or selecting a
model. It combines full-population correction information with model-relative
anomaly detection, optional attribute explanations, and predicted-state
compressibility.

```python
from mnplib import ResidualAnalysis, format_analysis

diagnostics = ResidualAnalysis(task="regression").fit(
    X, y, predictions=fitted_model.predict(X),
)
print(format_analysis(diagnostics.analysis()))
samples = diagnostics.results_dataframe()  # all evaluated observations
patterns = diagnostics.patterns_dataframe(only_anomalies=True)
attributes = diagnostics.feature_analysis(kind="under_predicted")  # lazy, cached
distribution = diagnostics.attribute_distribution(X.columns[0])
compression = diagnostics.compressibility()
```

Use `fit_y(y, predictions=...)` when no explanatory attributes are available.
Optional `sample_ids` must be unique and aligned with y. Predictions must be
finite, non-missing, one-dimensional and aligned; no rows are silently dropped.
Returned reports are independent copies. Refitting invalidates cached details.

Classification anomalies are label mismatches. Regression anomalies are
mismatches in common uniform bins learned from the observed target (one bin for
a constant target), with separate out-of-range prediction states. This is not
an information threshold. Local correction information is
`-log2 p(observed_state | predicted_state)`; negative local explanatory gain is
`max(0, -log2[p(observed_state | predicted_state) / p(observed_state)])`.
Probabilities always use the full fitted population, including when reporting
only anomalies.

`analysis()` reports empirical code lengths K(target), K(predictions), K(joint),
K(target | predictions) and K(predictions | target), without codebook costs.
These are empirical Shannon descriptions, not exact Kolmogorov complexity or
generalization estimates. Joint occupancy and singleton-sample fraction expose
sparsity; deterministic wrong mappings may have zero conditional information.
The independently computed `inaccuracy` retains Inaccuracy's encoding, which
fits numeric edges independently. Common-state correction lengths are therefore
**not a decomposition of that score**. Model selection and metric formulas are
unchanged.

`feature_analysis()` uses all supplied attributes to describe variation between
correction patterns within a requested anomaly group. It does not explain
anomaly membership or establish causality. Empty/insufficient anomaly groups,
single patterns, and missing attributes have machine-readable statuses.
Reliability and NaN behavior are delegated to Miscoding. The optional
`max_features` bounds the greedy selection; returned DataFrames preserve names.

`compressibility()` concerns predicted states among the requested anomalies,
relative to a uniform code over the complete target alphabet. It does not
measure numerical-residual compression or surfeit. Empty groups return
`status="no_anomalies"`; their numeric zero convention is not evidence of
compression. `attribute_distribution()` returns all-population and anomaly
counts using shared bins. Qualitative interpretations belong to applications.

### Core Concepts

The library combines four primitive components through three standard scalar metrics:

```text
miscoding = sqrt((deficiency**2 + surplus**2) / 2)
mismodel  = sqrt((inaccuracy**2 + surfeit**2) / 2)
nescience = sqrt(weight * miscoding**2 + (1 - weight) * mismodel**2)
```

The first two metrics always use equal-weight root mean square (RMS). Only the
final combination accepts `weight` in [0, 1] for miscoding; mismodel receives
`1 - weight`. The default is `weight=0.5`. For the same primitive values, default nescience
equals `sqrt((deficiency**2 + surplus**2 + inaccuracy**2 + surfeit**2) / 4)`.
These scalar computations are practical estimates, not exact computations of
the theory's non-computable quantities.

#### Miscoding

`Miscoding` measures how well a dataset represents the target variable of a learning problem. It has two complementary components: `deficiency` and `surplus`. `Deficiency` measures the amount of target-relevant information that is missing from the available features, while `surplus` measures the amount of information contained in the features that is not useful to describe the target. A good representation should therefore have low deficiency and low surplus: it should contain the information needed to explain the target, but avoid unnecessary or irrelevant information.

```python
import pandas as pd
from sklearn.datasets import load_iris
from mnplib.miscoding import Miscoding

iris = load_iris()
X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
y = iris.target

miscoding = Miscoding()
miscoding.fit(X, y)

miscoding.feature_analysis()
```

```text
	feature_index	feature_name	is_numeric	code_length_bits	deficiency	surplus	miscoding
0	3	petal width (cm)	True	429.162687	0.109141	0.506488	0.366362
1	2	petal length (cm)	True	426.897505	0.147596	0.525286	0.385817
2	0	sepal length (cm)	True	466.167349	0.542982	0.766922	0.664454
3	1	sepal width (cm)	True	426.921835	0.725199	0.846968	0.788438
```

For interactive applications, fit once and request only the needed diagnostics:

```python
# Subset metrics and reliability.
overview = miscoding.subset_analysis([0, 1, 2])
ordering = miscoding.rank_features(
    criterion="miscoding", return_details=True, include_pairwise_miscoding=False,
)
selection = miscoding.select_features(return_details=True, include_pairwise_miscoding=False)

# Request the full pairwise matrix only when needed; values are cached.
pairwise_miscoding = miscoding.pairwise_miscoding_matrix()
```

The pairwise miscoding matrix is symmetric, with values between zero and one and a zero diagonal. Low values indicate that the two features describe each other well; high values indicate little shared information. It compares features without using the target.

Every feature, subset, fitted-model, and pairwise miscoding score is the RMS of
its own deficiency and surplus. Pairwise comparisons treat one feature as the
representation and the other as the target; exchanging them swaps the two
components without changing the RMS. A constant versus a nonconstant feature
has components 0 and 1 and therefore miscoding `1 / sqrt(2)`.
The empty subset also has this miscoding when the target has positive code
length; all three values are zero for a constant target. The scalar
`mnplib.miscoding.miscoding(deficiency=..., surplus=...)` helper combines
already-computed components without fitting.

Subset analysis reports metrics, reliability, and selected-feature metadata without computing pairwise miscoding. For detailed selection and ranking reports, `include_pairwise_miscoding=False` omits the `pairwise_miscoding` field without changing metric values, reliability decisions, selection, or ranking paths. The functional `rank_features` and `select_features` helpers accept the same option. Fitted data and variable types are snapshots; refitting resets caches. Serialize concurrent access to a retained estimator, as its lazy caches are mutable.

#### Discretization

Metrics and model searches resolve numeric discretization internally. `Miscoding` uses `max(2, floor(2 * n_samples**(1/3) / log2(d + 1)))` bins for a subset of `d` features, excluding the target. It applies that count consistently to joint and marginal distributions. Feature diagnostics use `d=1`; pairwise miscoding uses `d=2`.

`Inaccuracy` and `Surfeit` use the target-vector rule, `max(2, floor(2 * n_samples**(1/3)))`. Inaccuracy shares the count across target, prediction, and joint distributions. Regression anomaly detection uses the same vector rule with common target-domain bin edges. Nescience, AutoML, and time series delegate to these component policies.

Reports expose resolved bin counts where relevant. Sparse subset distributions remain subject to reliability checks. Explicit bin configuration is available in the low-level `mnplib.utils` functions: `discretize_vector`, `empirical_distribution_vector`, and `empirical_distribution_array`.

#### Inaccuracy

`Inaccuracy` measures how well a model predicts the target variable. It compares the observed target values with the values predicted by the model, estimating how much information is lost or distorted by the prediction process. A model with low inaccuracy produces predictions that preserve most of the relevant information in the target, while a model with high inaccuracy leaves many errors to be corrected. In `mnplib`, inaccuracy captures the predictive adequacy of a model.

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from mnplib.inaccuracy import Inaccuracy
from mnplib.reporting import format_analysis

X, y = load_iris(return_X_y=True)

model = DecisionTreeClassifier(min_samples_leaf=5)
model.fit(X, y)

inaccuracy = Inaccuracy()
inaccuracy.fit(X, y)

report = inaccuracy.model_analysis(model)
print(format_analysis(report))
```

Inaccuracy reports contain the information-based score, empirical code lengths
in bits, and joint-distribution diagnostics for both numeric and categorical
targets.

#### Surfeit

`Surfeit` measures the unnecessary complexity contained in a model description. A model may predict the target accurately but still include redundant structure, accidental details, or overly complex rules that are not needed to describe the underlying regularities of the data. A model with low surfeit provides a compact and economical description, while a model with high surfeit may be memorizing details that do not improve understanding. In `mnplib`, surfeit captures the descriptive economy of a model.

The library applies a fixed compression policy to canonical model descriptions,
shared by standalone metrics, AutoML, and time-series searches. Analysis reports
include raw, compressed, and reference code lengths in bits.

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from mnplib.surfeit import Surfeit
from mnplib.reporting import format_analysis

X, y = load_iris(return_X_y=True)

model = DecisionTreeClassifier(min_samples_leaf=5)
model.fit(X, y)

surfeit = Surfeit()
surfeit.fit(X, y)

report = surfeit.model_analysis(model)
print(format_analysis(report))
```

#### Mismodel

`Mismodel` coordinates `Inaccuracy` and `Surfeit`: the former measures mismatch
between targets and predictions, and the latter measures redundancy in the
model description. Its practical scalar estimate is
`sqrt((inaccuracy**2 + surfeit**2) / 2)`, not an exact computation of the theory's
non-computable quantities. There are no internal weights or alternative
aggregation rules.

Use `fit_y(y)` when predictions and a model-description string are available:

```python
from mnplib import Mismodel
from mnplib.reporting import format_analysis

y = [0, 0, 1, 1, 0, 1]
predictions = [0, 1, 1, 1, 0, 1]
model_string = "def predict(x):\n    return x[0]\n"

metric = Mismodel(y_type="categorical").fit_y(y)
components = metric.components(predictions=predictions, model_string=model_string)
value = metric.mismodel(predictions=predictions, model_string=model_string)
report = metric.analysis(predictions=predictions, model_string=model_string)
print(format_analysis(report))
```

Use `fit(X, y)` to retain evaluation inputs for an already-fitted supported
estimator. Mismodel does not train the supplied model or compute feature
relevance. Its model methods obtain predictions and the description from the
same canonical artifact layer used by Nescience and AutoML:

```python
from sklearn.linear_model import LinearRegression
from mnplib import Mismodel

X = [[0], [1], [2], [3], [4], [5]]
y = [1, 3, 5, 7, 9, 11]
model = LinearRegression().fit(X, y)

metric = Mismodel(y_type="numeric").fit(X, y)
value = metric.mismodel_model(model)
report = metric.model_analysis(model)
assert value == report["mismodel"]
print(report)
```

All evaluation methods require successful fitting. After `fit_y(y)`, supply
`X` explicitly to model methods. Its sample count and row order must match the
fitted target; row order cannot be verified automatically. `feature_names` and
`feature_indices` follow canonical artifact conventions: explicit X contains
estimator input columns, whereas indices select columns from stored X.
DataFrame labels and mixed feature types are preserved when retaining inputs.
Refitting replaces evaluation state; a failed fit leaves the metric unfitted.

`components()` returns exactly `inaccuracy` and `surfeit`; `analysis()` adds
their `mismodel`. `model_analysis()` also supplies `model_type` and `model_string`.
Normalized finite components yield a mismodel in [0, 1]; exact zero components
remain zero. A NaN or infinite component yields NaN mismodel. Malformed
predictions and invalid descriptions still raise validation errors. The scalar
`mismodel()` helper and `Mismodel.aggregate_components()` work without fitting;
finite negative component values are rejected.

Nescience owns a fitted `mismodel_` coordinator. The top-level weight does not change
the reported inaccuracy, surfeit, or equal-weight mismodel. Unreliable miscoding
subsets do not prevent independent mismodel diagnostics.

#### Nescience

`Nescience` measures how well a dataset, a target variable, and a model together describe a learning problem. It combines representation quality (`miscoding`) with predictive and descriptive quality (`mismodel`) using weighted RMS. Lower nescience indicates a better balance between these two dimensions.

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from mnplib.nescience import Nescience
from mnplib.reporting import format_analysis

X, y = load_iris(return_X_y=True)

model = DecisionTreeClassifier(min_samples_leaf=5)
model.fit(X, y)

nescience = Nescience()
nescience.fit(X, y)

report = nescience.model_analysis(model)
print(format_analysis(report))
```

For an advanced comparison, give the model dimension three times the weight of
the representation dimension:

```python
weighted = Nescience(weight=0.25).fit(X, y)
report = weighted.model_analysis(model)
print(format_analysis(report))
```

`weight` must be a finite real number between 0 and 1, inclusive. It controls
miscoding's contribution; mismodel's coefficient is always `1 - weight`.
The default `0.5` balances both dimensions. Set `weight=0` to score only mismodel
or `weight=1` to score only miscoding. Reports expose the single `weight` value.
The same weight convention
applies to `NescienceClassifier`, `NescienceRegressor`, and `TimeSeries`.

A zero weight removes that dimension's numerical contribution. A zero weighted
score therefore need not mean that every primitive is zero. Nonfinite estimates
remain visible: either nonfinite primitive makes its derived metric NaN, and
either nonfinite derived metric makes nescience NaN, even with zero weight.
An unavailable metric does not erase the other metric's diagnostics.

`components(...)` returns exactly deficiency, surplus, inaccuracy, and surfeit.
`analysis(...)` and `model_analysis(...)` additionally return miscoding,
mismodel, nescience, weight, and reliability diagnostics. When the two
derived scores are already available, use
`Nescience(weight=...).aggregate_components(miscoding=..., mismodel=...)`.

#### Text Reports

Analysis methods return dictionaries or DataFrames. Use `format_analysis()` for
an aligned text summary in a notebook, terminal, or log:

```python
from mnplib.reporting import format_analysis

selected_features = miscoding.select_features()
report = miscoding.subset_analysis(selected_features)
print(format_analysis(report))
print(format_analysis(miscoding.feature_analysis()))
```

| Class | Supported report methods |
| --- | --- |
| `Miscoding` | `subset_analysis()`, `model_analysis()`, `feature_analysis()` |
| `Inaccuracy` | `prediction_analysis()`, `model_analysis()` |
| `Surfeit` | `description_analysis()`, `model_analysis()` |
| `Mismodel` | `analysis()`, `model_analysis()` |
| `Nescience` | `analysis()`, `model_analysis()` |
| `NescienceClassifier`, `NescienceRegressor` | `analysis()` |
| `TimeSeries` | `analysis()`, `lag_analysis()` |
| `ResidualAnalysis` | `analysis()` |

Functional analysis helpers return the same supported structures. Formatting
does not fit or score a model, modify a report, or print automatically. Summaries
include metric values, relevant context, and available reliability diagnostics.
Code lengths are labeled in bits. AutoML estimator scores are identified as
training-data scores, not held-out results; classification scores use percentages.
Model-description strings and prediction arrays are omitted, long feature lists
are abbreviated, and tables show at most 20 rows. The structured report remains
available for programmatic use.

See [Reporting.ipynb](examples/Reporting.ipynb) for examples covering every class.

### Automated Machine Learning

`mnplib` includes AutoML tools for classification and regression.

Unlike conventional AutoML systems, the goal is not to perform a large hyperparameter grid search, or to run time consuming cross-validations. Instead, the current design explores candidate models generated by nescience-guided principles, such as:

* feature selection based on miscoding-guided ranking;
* robust accuracy computation;
* model complexity measured by serialization;
* optimal pruning paths, bounded local searches, ...

AutoML uses feature ordering and model complexity analysis because the final objective is total nescience, not inaccuracy alone. A feature that increases inaccuracy may still be useful if it allows a model to reduce miscoding or surfeit enough to reduce total nescience.

The library intentionally avoids time consuming AutoML mechanisms in its default methodology, such as:

* broad hyperparameter grid search;
* cross-validation-based model selection;
* ensemble search as a default strategy.

This is a design choice: the purpose of `mnplib` is to explore machine learning through nescience-guided representations, descriptions, and model comparison.

#### Auto Classification

The auto-classification class in `mnplib` automatically searches for a good classification model using the Minimum Nescience Principle. It analyzes the available features, builds candidate classifiers, evaluates their predictions, and compares their nescience components to select the model that best balances relevant data, predictive quality, and model simplicity. Instead of optimizing accuracy alone, auto-classification looks for a classifier that explains the target with low miscoding, low inaccuracy, and low surfeit.

```python
from sklearn.datasets import load_breast_cancer
from mnplib.classifier import NescienceClassifier
from mnplib.reporting import format_analysis

X, y = load_breast_cancer(return_X_y=True)

model = NescienceClassifier()
model.fit(X, y)

predictions = model.predict(X)

print(format_analysis(model.analysis()))
```

#### Auto Regression

The auto-regressor class in `mnplib` automatically searches for a good regression model using the Minimum Nescience Principle. It analyzes the available features, builds candidate regressors, evaluates their predictions, and compares their nescience components to select the model that best balances relevant data, predictive quality, and model simplicity. Instead of minimizing prediction error alone, auto-regression looks for a regressor that explains the target with low miscoding, low inaccuracy, and low surfeit.

```python
from sklearn.datasets import load_diabetes
from mnplib.regressor import NescienceRegressor
from mnplib.reporting import format_analysis

X, y = load_diabetes(return_X_y=True)

model = NescienceRegressor()
model.fit(X, y)

predictions = model.predict(X)

print(format_analysis(model.analysis()))
```

### User Guide

The user guide contains the following sections:

* [Feature Selection](https://github.com/rleiva/fastautoml/wiki/Feature-Selection)
* [Model Inaccuacy](https://github.com/rleiva/fastautoml/wiki/Model-Inaccuracy)
* [Model Complexity](https://github.com/rleiva/fastautoml/wiki/Model-Complexity)
* [Hyperparameters Selection](https://github.com/rleiva/fastautoml/wiki/Hyperparameters-Selection)
* [Auto Classification](https://github.com/rleiva/fastautoml/wiki/Auto-Classification)
* [Auto Regression](https://github.com/rleiva/fastautoml/wiki/Auto-Regression)
* [Time Series](https://github.com/rleiva/fastautoml/wiki/Time-Series-Analysis)
* [Anomalies Detection](https://github.com/rleiva/nescience/wiki/Anomalies-Detection)

### Testing

To run the test suite from the repository root:

```bash
python -m pip install -e .
python -m pytest tests
```

Using `python -m pytest` is recommended because it ensures that tests run with the same Python interpreter in which `mnplib` is installed.

### Contributing

This is an alpha release, and feedback is welcome.

Useful contributions include:

* testing the package on real datasets;
* reporting bugs;
* improving documentation;
* adding examples;
* reviewing the mathematical consistency of the metrics;
* improving or extending model serializers;
* proposing better nescience-guided search strategies.

Please open issues or pull requests through the GitHub repository:

https://github.com/rleiva/mnplib

#### Reporting Issues

When reporting a bug, please include:

* the installed `mnplib` version;
* the Python version;
* the operating system;
* a minimal code example;
* the full error traceback;
* the dataset shape and target type, when relevant.

You can check the installed version with:

```python
import mnplib

print(mnplib.__version__)
```

### Project Status

`mnplib 2` is in alpha release, and is being redesigned. The package is being actively revised, and several parts of the API may still change.

The current priority is to stabilize:

* empirical code-length utilities;
* miscoding, inaccuracy, surfeit, and nescience metrics;
* explicit model serialization;
* automated classification and regression workflows;
* examples and documentation.

### Candidate-specific time-series forecasts

```python
from mnplib.timeseries import TimeSeries

capabilities = TimeSeries.family_capabilities()
search = TimeSeries(models=list(capabilities)).fit(y)
candidate = search.results_dataframe().iloc[-1]["candidate"]
future = search.forecast(steps=12, candidate=candidate)
fitted = search.fitted_values(candidate=candidate)
```

Omitting `candidate` uses the minimum-nescience model. Named-candidate calls do
not change `best_result_`, `model_`, or the selected metrics. `fitted_values()`
returns an independent array aligned with the original series, with NaN in the
initial lag-window positions. These values are in-sample predictions, not
held-out forecast performance.

`family_capabilities()` returns fresh records keyed by supported family ID,
including external-input support, use of future external inputs, forecast
strategy, and subset semantics. For autoregressive forecasting, omitted future
external values retain the existing last-observation behavior. ARIMA and
state-space subsets are `diagnostic_proxy` representations for metric evaluation,
not literal lag inputs; other families report `lag_inputs`.

Candidate-report `metadata` contains `subset_semantics`. Statsmodels-backed
candidates additionally expose `converged` and `optimizer_iterations` (null if
unavailable). `diagnostics_` includes `not_converged` records. Nonconvergence does
not change candidate ranking; applications can surface it separately from
representation reliability. Qualitative interpretations belong to the caller.

### License

See the repository license file for licensing details.

### Citation

If you use `mnplib` in academic or research work, please cite the associated theoretical work on the [*Minimum Nescience Principle*](https://www.amazon.com/dp/B0GZH1FZ9J).

```tex
@book{garcialeiva2026unknown,
  author    = {García Leiva, Rafael Ángel},
  title     = {A Mathematical Theory of the Unknown: Journey Beyond the Frontiers of Human Understanding},
  year      = {2026},
  isbn      = {9798258956484},
  note      = {ASIN: B0GZH1FZ9J},
  url       = {https://www.amazon.com/dp/B0GZH1FZ9J},
  urldate   = {2026-09-14}
}
```
