# Machine Learning with the Minimum Nescience Principle

`mnplib` is an open-source Python library for machine learning based on the **Minimum Nescience Principle**. It is built on top of [scikit-learn](https://scikit-learn.org/stable/) and provides tools for evaluating datasets, models, and automated machine-learning candidates through the lens of nescience.

The library is based on the [*Minimum Nescience Principle*](https://www.amazon.com/dp/B0GZH1FZ9J), a mathematical framework for measuring how well a problem is understood given a **representation** and a **description**. In machine learning, the representation is usually a dataset, while the description is a fitted model.

`mnplib` is currently focused on three main goals:

* measuring the quality of data representations;
* measuring the quality and complexity of fitted models;
* exploring nescience-guided alternatives to conventional AutoML.

> **Development status**
>
> `mnplib 2` is an alpha release. The API, documentation, examples, and automated searchers are still under active development. The package is suitable for experimentation and testing, but it should not yet be considered production-ready.

## Installation

Install the current alpha release from PyPI:

```bash
pip install mnplib==2.0.0a1
```

Because this is a pre-release, installing the explicit version is recommended.

## Core Concepts

### Evaluate a Trained Model

All four metrics evaluate the same data through a consistent fitted-model API:

```python
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from mnplib import Miscoding, Inaccuracy, Surfeit, Nescience

X, y = make_regression(n_samples=500, n_features=2, noise=1, random_state=42)
model = LinearRegression().fit(X, y)

miscoding = Miscoding(n_bins=3).fit(X, y).miscoding_model(model)
inaccuracy = Inaccuracy(n_bins=3).fit(X, y).inaccuracy_model(model)
surfeit = Surfeit(n_bins=3).fit(X, y).surfeit_model(model)
metric = Nescience(n_bins=3).fit(X, y)
nescience = metric.nescience_model(model)
report = metric.model_analysis(model)
```

Lower metric values are better. Metric fitting establishes the evaluation data;
it does not fit the supplied model. Supported models use canonical serializers.
The explicit `inaccuracy_predictions()`, `surfeit_string()`, and
`Nescience.nescience(subset=..., predictions=..., model_string=...)` APIs provide
direct access to the numerical primitives.

All four classes provide `model_analysis(model)` for a diagnostic dictionary:

```python
reports = {
    cls.__name__: cls(n_bins=3).fit(X, y).model_analysis(model)
    for cls in (Miscoding, Inaccuracy, Surfeit, Nescience)
}
```

For a compact statistical summary, use the shared text formatter:

```python
from mnplib.reporting import format_analysis

print(format_analysis(reports["Inaccuracy"]))
```

`format_analysis(report)` returns a plain string without modifying the report or
printing automatically. It works with all four metrics' `model_analysis()`
reports, plus `subset_analysis()`, `prediction_analysis()`, and
`description_analysis()`. Scores use four decimal places, accuracy is displayed
as a percentage, and code lengths are labeled in bits. The underlying dictionary
retains full precision and all diagnostics. Unreliable subsets always show their
failure reason; model strings and large arrays are omitted from the summary.
No plotting or notebook dependencies are required by the formatter.

Targets and predictions must be one-dimensional. `n_bins` accepts integers
greater than or equal to two, `"auto"`, or `"adaptive"`. DataFrame column labels
are preserved in fitted feature metadata and analysis reports.

Model methods accept keyword-only `X`, `feature_names`, and `feature_indices`.
For a model trained on a subset, `feature_indices` maps its input columns into
the metric's original feature space. Omitting `X` selects those columns from
the metric's fitted data. Explicit `X` must contain the estimator input columns
in order and correspond to the fitted target rows.

The library is built around the notion of **nescience**, which combines several complementary quantities.

### Miscoding

`Miscoding` measures how well a dataset represents the target variable. It can be used to analyze feature relevance, feature redundancy, and the representational quality of the available data.

`select_features()` performs strict miscoding-based subset selection. It keeps a feature only when adding it reduces subset miscoding by the configured improvement threshold.

`rank_features()` produces a miscoding-guided empirical feature order for model construction. It keeps ordering features even when subset miscoding stops improving.

```python
metric = Miscoding(n_bins=3).fit(X, y)
one_feature = metric.miscoding_feature(0)
all_features = metric.miscoding_feature()
subset = metric.subset_analysis([0, 1])
mask = metric.select_features(min_improvement=0.01)
order = metric.rank_features(max_features=2)
```

Integer sequences specify feature indices; Boolean arrays specify masks.
Feature methods also accept DataFrame column names. `deficiency_feature()` and
`surplus_feature()` follow the same convention. `deficiency_subset()` and
`surplus_subset()` return the corresponding subset quantities.
`feature_analysis()` reports each feature's code length as `code_length_bits`.
`model_analysis(model)` reports subset diagnostics for the features effectively
used by the canonical serialized model.

Subset diagnostics include `is_reliable`, `failure_reason`, `resolved_n_bins`,
joint-state counts, and occupancy diagnostics. `resolved_n_bins` is the numeric
bin count for the evaluated subset, or `None` for an empty subset or a purely
categorical joint distribution. Categorical variables retain their categories.
Unreliable subsets return NaN metrics. Feature
ranking and selection stop when no reliable extension exists. Reports use
`selected_features` for effective feature indices, `selected_feature_names` for
labels, and `n_selected_features` for their count. Selection details expose a
Boolean `mask` separately.

Subset-based metrics and AutoML default to `n_bins="adaptive"`, including
classification, regression, time series, and the AutoML used by AnomalyDetector.
An explicit integer or `n_bins="auto"` selects a fixed-resolution policy.
`"auto"` uses `b = max(2, floor(2 * n^(1/3)))`. `"adaptive"` uses
`b(S) = max(2, floor(2 * n^(1/3) / log2(|S| + 1)))` for subset-level empirical
quantities; for `|S| = 1` it equals `"auto"`, and for larger subsets it coarsens
discretization to reduce empirical joint sparsity.

Adaptive discretization does not guarantee reliability. Direct
`miscoding_model()` and `nescience_model()` calls emit a `RuntimeWarning` when returning NaN;
`model_analysis()` provides diagnostics without this warning, and AutoML quietly
excludes unreliable candidates from selection. Use the same discretization
policy when comparing models; changing it changes the evaluated representation.

### Inaccuracy

`Inaccuracy` compares true target values with model predictions using empirical
code lengths. Its analysis methods put this information-based measure alongside
conventional prediction errors:

```python
metric = Inaccuracy().fit(X, y)
report = metric.model_analysis(model)
prediction_report = metric.prediction_analysis(model.predict(X))
assert report == prediction_report
```

Both return a flat dictionary containing `inaccuracy`, `n_samples`, resolved
`y_type`, `resolved_n_bins`, and `target_code_length_bits`,
`prediction_code_length_bits`, and `joint_code_length_bits`. The report also
includes observed joint-state counts, mean occupancy, and singleton fraction.
These sparsity statistics are descriptive; they do not impose a rejection rule
on Inaccuracy. Categorical targets include `accuracy` and have no numeric bin
count. Numeric targets include `mae` and `rmse`, measured on the original values
in target units. Information-based inaccuracy is not classification error or
residual magnitude.

`model_analysis()` uses the same optional `X`, `feature_names`, and
`feature_indices` arguments as `inaccuracy_model()` and requires only a fitted
predictor, not a serializer. `fit_y(y)` supports prediction-only analysis.

### Surfeit

`Surfeit` measures the unnecessary complexity of a model description. A model may be accurate but unnecessarily complex; surfeit is intended to capture this excess descriptive cost.

```python
from sklearn.linear_model import LinearRegression
from mnplib.surfeit import Surfeit

model = LinearRegression()
model.fit(X, y)

surfeit = Surfeit()
surfeit.fit(X, y)

value = surfeit.surfeit_model(model)
```

When a canonical model description is already available, the string API remains available:

```python
value = surfeit.surfeit_string(model_string)
```

Analyze a fitted model or its explicit canonical description:

```python
report = surfeit.model_analysis(model)
description_report = surfeit.description_analysis(report["model_string"])
assert report["surfeit"] == description_report["surfeit"]
```

Analysis reports express all code lengths in **bits**, with the fields
`model_code_length_bits`, `compressed_code_length_bits`,
`effective_compressed_code_length_bits`, `target_code_length_bits`, and
`reference_code_length_bits`. The effective compressed length subtracts the
configured zlib overhead and is clipped between zero and the raw model length.
Surfeit is `1 - reference_code_length_bits / model_code_length_bits`, where the
reference is the smaller of the target and effective compressed code lengths.
`reference_source` identifies the limiting quantity as `"target"`,
`"compression"`, or `"both"` for a tie.

`compression_ratio` is raw compressed size divided by raw model size, before
overhead correction or clipping; it may exceed one. `model_analysis()` also
returns the canonical model string, model type, and effective feature indices.
These diagnostics explain the computation and do not establish overfitting.
The target-independent `description_lengths(model_string)` utility reports
UTF-8 and compressed **byte** counts without requiring fitting.

Functional analysis helpers use the same names in their respective modules:

```python
from mnplib.inaccuracy import prediction_analysis
from mnplib.surfeit import description_analysis

prediction_report = prediction_analysis(model.predict(X), y=y)
description_report = description_analysis(report["model_string"], y=y)
```

Each module also exposes `model_analysis(model, X=X, y=y, ...)`.
Functional metric helpers take evaluation data as keyword-only `X` and `y`
arguments and expose their supported configuration parameters explicitly:

```python
from mnplib.miscoding import feature_analysis, select_features

features = feature_analysis(X=X, y=y, n_bins=3)
mask = select_features(X=X, y=y, n_bins=3)
```

### Nescience

`Nescience` combines representation quality, model error, and model complexity into a single quantity. It can be used to compare models from different model families, provided that each model can be converted into an explicit representation containing:

* the selected feature subset;
* the prediction vector;
* the serialized model description.

The default adaptive policy supports direct evaluation of a fitted Iris tree:

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from mnplib import Nescience

X_iris, y_iris = load_iris(return_X_y=True)
tree = DecisionTreeClassifier(min_samples_leaf=5, random_state=42).fit(X_iris, y_iris)
metric = Nescience().fit(X_iris, y_iris)
value = metric.nescience_model(tree)
diagnostics = metric.model_analysis(tree)
print(value, diagnostics["is_reliable"], diagnostics["resolved_n_bins"])
```

`model_analysis()` returns `nescience`, `deficiency`, `surplus`, `inaccuracy`,
and `surfeit` as top-level fields alongside subset reliability diagnostics,
aggregation settings, and the canonical model description. The explicit-artifact
`explain()` method uses the same flat metric fields. `components()` returns only
the four component values when supplied with explicit model artifacts.

## Automated Machine Learning

`mnplib` includes experimental AutoML tools for classification and regression.

Unlike conventional AutoML systems, the goal is not to perform a large hyperparameter grid search, or to run time consuming cross-validations. Instead, the current design explores candidate models generated by nescience-guided principles, such as:

* feature prefixes based on miscoding-guided ranking;
* robust accuracy computation;
* model complexity measured by serialization;
* optimal pruning paths, bounded local searches, ...

AutoML uses feature ordering because the final objective is total nescience, not miscoding alone. A feature that increases miscoding may still be useful if it allows a model to reduce inaccuracy or surfeit enough to reduce total nescience.

The automated searchers are still experimental and may change substantially before a stable release.

The classification and regression interfaces share `nescience()`, `components()`,
`explain()`, `results_dataframe()`, and `model_description(candidate=None)`.
`nescience()` returns the selected candidate's search metric; `score(X, y)`
returns predictive accuracy or R-squared. The selected fitted model is `model_`.
Result tables place reliable candidates first and sort by increasing nescience.
No reliable candidate produces a clear fitting error.

```python
from mnplib import NescienceClassifier, NescienceRegressor

classifier = NescienceClassifier(models=["decision_tree"], n_bins=3).fit(X, y > 0)
regressor = NescienceRegressor(
    models=["linear_regression", "decision_tree"],
    n_bins=3,
    weights={"inaccuracy": 2.0},
    search_options={"linear_regression": {"patience": 2}},
).fit(X, y)

for estimator in (classifier, regressor):
    predictions = estimator.predict(X)
    print(estimator.nescience(), estimator.components())
    print(estimator.results_dataframe())
    print(estimator.model_description()["model_string"])
```

The same text formatter summarizes the selected AutoML candidate:

```python
from mnplib.reporting import format_analysis

print(format_analysis(classifier.explain()))
print(format_analysis(regressor.explain()))
```

AutoML summaries show the selected candidate, model family, effective features,
nescience components, reliability, aggregation, and reported hyperparameters.
`explain()` includes `task`, `native_estimator_score`, and `evaluation_context`.
For classification and regression, the recorded score is training-data accuracy
or R-squared, respectively (`evaluation_context="training"`), not held-out
performance. Formatting does not fit or score the model again. Use
`results_dataframe()` for the full candidate comparison.

`search_options` is keyed by model family and validates option names. For example,
logistic regression accepts `max_iter`, decision trees accept `n_jobs`, and MLP
search accepts `max_candidates` and `max_iter`. Component `weights` and
`aggregation` have the same meaning across Nescience and the search classes.

## Forecasting

```python
import numpy as np
from mnplib import TimeSeries
from mnplib.reporting import format_analysis

series = np.sin(np.arange(240) / 12)
forecaster = TimeSeries(
    window_size=4, models=["arima", "state_space"], n_bins=3,
    search_options={"arima": {"orders": [(1, 0, 0)], "max_iter": 50}},
).fit(series[:-12])
forecast = forecaster.forecast(12)
forecast_r2 = forecaster.score(series[-12:])
print(forecaster.results_dataframe())
print(format_analysis(forecaster.explain()))
```

`fit(series, X=...)` accepts optional exogenous data. `forecast(steps,
X_future=...)` accepts future exogenous inputs for models that use them.
`score(y_future, X_future=...)` evaluates observations immediately following
training. `fitted_values_` aligns training predictions with the series, using
NaN for initial positions without a complete lag window. `lag_analysis()`
combines target and exogenous lag diagnostics into one DataFrame.

The time-series text summary also shows the lag window and selected lag features.
The shared `hyperparameters` field contains the selected averaging window,
smoothing alpha, ARIMA order and trend, or state-space specification when applicable.
Its evaluated sample count is the number of lagged training rows, not the length
of the original series. The candidate-evaluation R-squared uses that same lagged
representation (`evaluation_context="lagged_training"`). It does not measure
future forecasting performance; `score(y_future, X_future=...)` evaluates that
separately.

## Model-Relative Anomalies

```python
from mnplib import AnomalyDetector

detector = AnomalyDetector(task="regression", n_bins=3).fit(X, y, model=regressor)
indices = detector.anomalies()
print(detector.results_dataframe())
explanation = detector.explain()
```

Supply a fitted model or `predictions=...`; omitting both runs the appropriate
AutoML search. Classification anomalies are misclassified observations;
regression anomalies are mismatches under a common target discretization.
`explain()` includes summary, correction-pattern feature analysis, and
compressibility diagnostics.

Functional helpers live in the corresponding metric modules and use the same
operation names, for example `nescience_model(model, X=X, y=y, n_bins=3)` and
`surfeit_string(model_string, y=y)`.

## User Guide

The user guide contains the following sections:

* [Feature Selection](https://github.com/rleiva/fastautoml/wiki/Feature-Selection)
* [Model Inaccuacy](https://github.com/rleiva/fastautoml/wiki/Model-Inaccuracy)
* [Model Complexity](https://github.com/rleiva/fastautoml/wiki/Model-Complexity)
* [Hyperparameters Selection](https://github.com/rleiva/fastautoml/wiki/Hyperparameters-Selection)
* [Auto Classification](https://github.com/rleiva/fastautoml/wiki/Auto-Classification)
* [Auto Regression](https://github.com/rleiva/fastautoml/wiki/Auto-Regression)
* [Time Series](https://github.com/rleiva/fastautoml/wiki/Time-Series-Analysis)
* [Anomalies Detection](https://github.com/rleiva/nescience/wiki/Anomalies-Detection)

## Quick Start: Classification

```python
from sklearn.datasets import load_breast_cancer

from mnplib.classifier import NescienceClassifier

X, y = load_breast_cancer(return_X_y=True)

model = NescienceClassifier()
model.fit(X, y)

print(model.predict(X))
print(model.score(X, y))
print(model.components())
```

## Quick Start: Regression

```python
from sklearn.datasets import make_regression

from mnplib.regressor import NescienceRegressor

X, y = make_regression(
    n_samples=100,
    n_features=5,
    n_informative=3,
    noise=0.1,
    random_state=42,
)

model = NescienceRegressor()

model.fit(X, y)

print(model.predict(X))
print(model.score(X, y))
print(model.components())
```

## Inspecting AutoML Results

After fitting an automated model, you can inspect the evaluated candidates:

```python
results = model.results_dataframe()
print(results.head())
```

The results table contains information such as:

* candidate name;
* model family;
* nescience score;
* miscoding components;
* inaccuracy;
* surfeit;
* selected features;
* description length;
* native estimator score.

You can also retrieve the best fitted estimator:

```python
best_model = model.model_
```

and obtain a structured explanation:

```python
explanation = model.explain()
print(explanation)
```

## Main Modules

The main public modules are:

* `mnplib.miscoding`
* `mnplib.inaccuracy`
* `mnplib.surfeit`
* `mnplib.nescience`
* `mnplib.classifier`
* `mnplib.regressor`
* `mnplib.timeseries`
* `mnplib.utils`

Some modules are still experimental and may change before the stable `2.0.0` release.

## Current Scope

The current alpha version includes support for several model families, including:

* decision trees;
* linear regression;
* logistic regression;
* Gaussian Naive Bayes;
* linear support-vector models;
* multilayer perceptrons.

Support varies by task and by model family. Some serializers and searchers are experimental.

The library intentionally avoids some common AutoML mechanisms in its default methodology, such as:

* broad hyperparameter grid search;
* cross-validation-based model selection;
* ensemble search as a default strategy.

This is a design choice: the purpose of `mnplib` is to explore machine learning through nescience-guided representations, descriptions, and model comparison.

## Testing

To run the test suite from the repository root:

```bash
python -m pip install -e .
python -m pytest mnplib/tests
```

Using `python -m pytest` is recommended because it ensures that tests run with the same Python interpreter in which `mnplib` is installed.

## Contributing

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

## Reporting Issues

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

## Project Status

`mnplib 2` is in alpha release, and is being redesigned. The package is being actively revised, and several parts of the API may still change.

The current priority is to stabilize:

* empirical code-length utilities;
* miscoding, inaccuracy, surfeit, and nescience metrics;
* explicit model serialization;
* automated classification and regression workflows;
* examples and documentation.

## License

See the repository license file for licensing details.

## Citation

If you use `mnplib` in academic or research work, please cite the associated theoretical work on the [*Minimum Nescience Principle*](https://www.amazon.com/dp/B0GZH1FZ9J). A formal citation entry will be added in a future release.
