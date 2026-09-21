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

Use `Nescience.analysis(subset=..., predictions=..., model_string=...)` for explicit artifacts, `model_analysis(model)` for fitted estimators, and `analysis()` on `NescienceClassifier`, `NescienceRegressor`, `TimeSeries`, or `AnomalyDetector`. These reports do not assign qualitative profiles. Canonical model descriptions remain in mnplib because their compressed lengths determine surfeit.

`from mnplib.mismodel import mismodel` provides `mismodel(inaccuracy=..., surfeit=...)`, the root-mean-square of the two components. It is also included in nescience analysis reports and AutoML result dataframes; it does not change minimum-nescience model selection.

### Core Concepts

The `mnplib` library is built around the notion of **nescience**, which combines several complementary quantities: `miscoding`, `inaccuracy`, and `surfeit`.

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
0	3	petal width (cm)	True	429.162687	0.109141	0.506488	0.506488
1	2	petal length (cm)	True	426.897505	0.147596	0.525286	0.525286
2	0	sepal length (cm)	True	466.167349	0.542982	0.766922	0.766922
3	1	sepal width (cm)	True	426.921835	0.725199	0.846968	0.846968
```

#### Inaccuracy

`Inaccuracy` measures how well a model predicts the target variable. It compares the observed target values with the values predicted by the model, estimating how much information is lost or distorted by the prediction process. A model with low inaccuracy produces predictions that preserve most of the relevant information in the target, while a model with high inaccuracy leaves many errors to be corrected. In `mnplib`, inaccuracy captures the predictive adequacy of a model.

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from mnplib.inaccuracy import Inaccuracy
from pprint import pprint

X, y = load_iris(return_X_y=True)

model = DecisionTreeClassifier(min_samples_leaf=5)
model.fit(X, y)

inaccuracy = Inaccuracy()
inaccuracy.fit(X, y)

report = inaccuracy.model_analysis(model)
pprint(report)
```

#### Surfeit

`Surfeit` measures the unnecessary complexity contained in a model description. A model may predict the target accurately but still include redundant structure, accidental details, or overly complex rules that are not needed to describe the underlying regularities of the data. A model with low surfeit provides a compact and economical description, while a model with high surfeit may be memorizing details that do not improve understanding. In `mnplib`, surfeit captures the descriptive economy of a model.

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from mnplib.surfeit import Surfeit
from pprint import pprint

X, y = load_iris(return_X_y=True)

model = DecisionTreeClassifier(min_samples_leaf=5)
model.fit(X, y)

surfeit = Surfeit()
surfeit.fit(X, y)

report = surfeit.model_analysis(model)
pprint(report)
```

#### Nescience

`Nescience` measures how well a dataset, a target variable, and a model together describe a learning problem. It combines the quality of the data representation, the accuracy of the predictions, and the economy of the model description. In `mnplib`, nescience is built from `miscoding`, `inaccuracy`, and `surfeit`: a good model should use relevant data, make informative predictions, and avoid unnecessary complexity. Lower nescience indicates a better balance between data, prediction, and model simplicity.

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from mnplib.nescience import Nescience
from pprint import pprint

X, y = load_iris(return_X_y=True)

model = DecisionTreeClassifier(min_samples_leaf=5)
model.fit(X, y)

nescience = Nescience()
nescience.fit(X, y)

report = nescience.model_analysis(model)
pprint(report)
```

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
from pprint import pprint

X, y = load_breast_cancer(return_X_y=True)

model = NescienceClassifier()
model.fit(X, y)

predictions = model.predict(X)

pprint(model.analysis())
```

#### Auto Regression

The auto-regressor class in `mnplib` automatically searches for a good regression model using the Minimum Nescience Principle. It analyzes the available features, builds candidate regressors, evaluates their predictions, and compares their nescience components to select the model that best balances relevant data, predictive quality, and model simplicity. Instead of minimizing prediction error alone, auto-regression looks for a regressor that explains the target with low miscoding, low inaccuracy, and low surfeit.

```python
from sklearn.datasets import load_diabetes
from mnplib.regressor import NescienceRegressor
from pprint import pprint

X, y = load_diabetes(return_X_y=True)

model = NescienceRegressor()
model.fit(X, y)

predictions = model.predict(X)

pprint(model.analysis())
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
search = TimeSeries(models=list(capabilities), n_bins="adaptive").fit(y)
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
