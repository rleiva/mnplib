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

Install the current alpha release from PyPI:

```bash
pip install mnplib==2.0.0a6
```

Because this is a pre-release, installing the explicit version is recommended.

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
from mnplib.reporting import format_analysis

X, y = load_iris(return_X_y=True)

model = DecisionTreeClassifier(min_samples_leaf=5)
model.fit(X, y)

inaccuracy = Inaccuracy()
inaccuracy.fit(X, y)

report = inaccuracy.model_analysis(model)
print(format_analysis(report))
```

```text
Inaccuracy Analysis
========================================
Samples                              150
Target type                  categorical

Inaccuracy                        0.0991
Accuracy                          97.33%

Code lengths (bits)
----------------------------------------
Target                           237.744
Predictions                      237.629
Joint                            261.189
```

#### Surfeit

`Surfeit` measures the unnecessary complexity contained in a model description. A model may predict the target accurately but still include redundant structure, accidental details, or overly complex rules that are not needed to describe the underlying regularities of the data. A model with low surfeit provides a compact and economical description, while a model with high surfeit may be memorizing details that do not improve understanding. In `mnplib`, surfeit captures the descriptive economy of a model.

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

```text
Surfeit Analysis
============================================
Model type            DecisionTreeClassifier
Selected count                             3
Selected features                    0, 2, 3

Surfeit                               0.8578
Compression ratio                     0.4498
Reference source                      target

Code lengths (bits)
--------------------------------------------
Model                               1672.000
Compressed                           752.000
Effective compressed                 704.000
Target                               237.744
Reference                            237.744
============================================
```

#### Nescience

`Nescience` measures how well a dataset, a target variable, and a model together describe a learning problem. It combines the quality of the data representation, the accuracy of the predictions, and the economy of the model description. In `mnplib`, nescience is built from `miscoding`, `inaccuracy`, and `surfeit`: a good model should use relevant data, make informative predictions, and avoid unnecessary complexity. Lower nescience indicates a better balance between data, prediction, and model simplicity.

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

```text
Nescience Analysis
====================================================================
Samples                                                          150
Model type                                    DecisionTreeClassifier
Numeric bins                                                       5
Selected count                                                     3
Selected features                                         x0, x2, x3
Subset reliability                                          Reliable

Nescience                                                     0.5375
Deficiency                                                    0.0868
Surplus                                                       0.6345
Inaccuracy                                                    0.0991
Surfeit                                                       0.8578

Aggregation
--------------------------------------------------------------------
Method                                                     euclidean
Weights             deficiency=1, surplus=1, inaccuracy=1, surfeit=1
====================================================================
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
from mnplib.reporting import format_analysis

X, y = load_breast_cancer(return_X_y=True)

model = NescienceClassifier()
model.fit(X, y)

predictions = model.predict(X)

print(format_analysis(model.explain()))
```

```text
Auto-Classification Analysis
====================================================================
Selected candidate                      logistic_regression_prefix_2
Model family                                     logistic_regression
Model type                                        LogisticRegression
Evaluated samples                                                569
Numeric bins                                                      10
Selected count                                                     2
Selected features                                           x22, x27
Subset reliability                                          Reliable

Nescience                                                     0.4744
Deficiency                                                    0.1843
Surplus                                                       0.8402
Inaccuracy                                                    0.3398
Surfeit                                                       0.2121

Candidate evaluation
--------------------------------------------------------------------
Data                                                   Training data
Accuracy                                                      94.02%

Hyperparameters
--------------------------------------------------------------------
max_iter                                                        1000
penalty                                                         None
solver                                                         lbfgs

Aggregation
--------------------------------------------------------------------
Method                                                     euclidean
Weights             deficiency=1, surplus=1, inaccuracy=1, surfeit=1
====================================================================
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

print(format_analysis(model.explain()))
```

```text
Auto-Regression Analysis
====================================================================
Selected candidate                               linear_svr_prefix_3
Model family                                              linear_svr
Model type                                                 LinearSVR
Evaluated samples                                                442
Numeric bins                                                       7
Selected count                                                     3
Selected features                                         x2, x8, x1
Subset reliability                                          Reliable

Nescience                                                     0.6687
Deficiency                                                    0.6510
Surplus                                                       0.8301
Inaccuracy                                                    0.8175
Surfeit                                                       0.0875

Candidate evaluation
--------------------------------------------------------------------
Data                                                   Training data
R-squared                                                    -0.2927

Hyperparameters
--------------------------------------------------------------------
C                                                                1.0
epsilon                                                          0.0
max_iter                                                        5000
tol                                                           0.0001

Aggregation
--------------------------------------------------------------------
Method                                                     euclidean
Weights             deficiency=1, surplus=1, inaccuracy=1, surfeit=1
====================================================================
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
