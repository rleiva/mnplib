# Machine Learning
## with the Minimum Nescience Principle

`mnplib` is a highly efficient open source library for machine learning based on Python and built on top of [scikit-learn](https://scikit-learn.org/stable/). The library is based on the [_minimum nescience principle_](https://www.amazon.com/dp/B0GZH1FZ9J), a novel mathematical theory that measures how well we understand a problem given a representation and a description. In case of machine learning, representations are based on datasets, and descriptions are based on mathematical models.

The minimum nescience principle allow us to automate the common tasks performed by data scientists, from feature selection, model selection, or hyperparameters optimization.

`mnplib` can dramatically increase the productivity of the data scientist, reducing the time to analyze and model a dataset. With `mnplib` we can have results in very short time, without decreasing the accuracy (in fact, we usually have a better accuracy). `mnplib` is fast because:

* It does not requires cross-validation
* It use a greedy search for hyperparameters
* It is not based on ensembles of models

Warning: This is an alpha release of mnplib 2.0. The API and documentation are still under active revision.

## The Library

The `mnplib` library is composed of the following classes:

* `Miscoding` measures the quality of the dataset we are using to represent our problem.
* `Inaccuracy` measures the error made by the model we have trained.
* `Surfeit` measures how (unnecessarily) complex is the model we have identified.

All these metrics are combined into a single quantity, called `Nescience`, as a measure of how well we understand our problem given a dataset and a model. `Nescience` allow us to evaluate and compare models from different model families.

The `mnplib` library also contains the following utilities:

* `Anomalies` for the identification and classification of anomalies.
* `Causal` for root-cause analysis.

Besides to these classes, the `mnplib` library provide the following automated machine-learning tools:

* `Regression` for automated regression problems.
* `Classification` for automated classification problems.
* `TimeSeries` for time series based analysis and forecasting.

## Installation

The `mnplib` library can be installed using the standard `pip` installer:

```python
pip install mnplib
```

The following code constains a simple example of how to use the auto-classifer module of the library:

```python
from sklearn.datasets import load_breast_cancer
from mnplib.classifier import Classifier

X, y = load_breast_cancer(return_X_y=True)

model = Classifier()
model.fit(X, y)
model.score(X, y)
```

## User Guide

This user guide contains the following sections:

* [Feature Selection](https://github.com/rleiva/fastautoml/wiki/Feature-Selection)
* [Model Inaccuacy](https://github.com/rleiva/fastautoml/wiki/Model-Inaccuracy)
* [Model Complexity](https://github.com/rleiva/fastautoml/wiki/Model-Complexity)
* [Hyperparameters Selection](https://github.com/rleiva/fastautoml/wiki/Hyperparameters-Selection)
* [Auto Classification](https://github.com/rleiva/fastautoml/wiki/Auto-Classification)
* [Auto Regression](https://github.com/rleiva/fastautoml/wiki/Auto-Regression)
* [Time Series](https://github.com/rleiva/fastautoml/wiki/Time-Series-Analysis)
* [Anomalies Detection](https://github.com/rleiva/nescience/wiki/Anomalies-Detection)
