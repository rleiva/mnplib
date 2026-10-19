"""Linear and polynomial SVM rules expressed through generic arithmetic."""

import numpy as np
from sklearn.svm import LinearSVC, LinearSVR, SVC, SVR
from sklearn.utils.validation import check_is_fitted
from ..._types import ResolvedTask
from ..language import Constant, Label, Binary, Conditional, Vector, Element, Index, Sum
from ..language.normalize import quantize
from .base import SklearnSerializer
from .linear import LinearModelSerializer, classification_rule


class LinearSVMSerializer(LinearModelSerializer):
    name = "linear_svm"
    supported_types = (LinearSVC, LinearSVR)

    def task(self, model) -> ResolvedTask:
        return "classification" if isinstance(model, LinearSVC) else "regression"

    def metadata(self, model):
        metadata = {"kernel": "linear", **super().metadata(model)}
        if isinstance(model, LinearSVC):
            metadata["n_classes"] = int(len(model.classes_))
        return metadata

    def serialize(self, model, *, features):
        if isinstance(model, LinearSVC):
            return classification_rule(model, features)
        return super().serialize(model, features=features)


def _dense(values):
    return np.asarray(values.toarray() if hasattr(values, "toarray") else values)


def _polynomial_kernel(features, support_vectors, support_index, *, gamma, coef0, degree):
    """Construct a polynomial kernel with a bounded input-coordinate sum."""
    j = Index("j")
    dot = Sum("j", len(features), Binary(
        "*", Element(Vector(tuple(features)), j),
        Element(Element(support_vectors, support_index), j),
    ))
    return Binary("**", Binary("+", Binary("*", Constant(float(gamma)), dot),
                               Constant(float(coef0))), Constant(int(degree)))


class PolynomialSVMSerializer(SklearnSerializer):
    """Describe polynomial SVR and binary SVC using explicit support vectors."""

    name = "polynomial_svm"
    supported_types = (SVC, SVR)

    def _validate(self, model):
        check_is_fitted(model)
        if callable(model.kernel):
            raise ValueError("Callable SVM kernels cannot be serialized canonically.")
        if model.kernel == "rbf":
            raise ValueError("RBF SVM kernels require exp, which is not part of the canonical language.")
        if model.kernel == "sigmoid":
            raise ValueError("Sigmoid SVM kernels require tanh, which is not part of the canonical language.")
        if model.kernel != "poly":
            raise ValueError(
                f"Unsupported SVM kernel {model.kernel!r}: SVC and SVR require kernel='poly'. "
                "Use LinearSVC or LinearSVR for linear models."
            )
        if isinstance(model, SVC) and len(model.classes_) != 2:
            raise ValueError("Polynomial SVC serialization supports binary classification only.")
        if quantize(model.degree) != model.degree:
            raise ValueError("Polynomial degree must be exactly representable at canonical precision.")

    def task(self, model) -> ResolvedTask:
        return "classification" if isinstance(model, SVC) else "regression"

    def description(self, model, *, feature_names=None, feature_indices=None):
        self._validate(model)
        return super().description(model, feature_names=feature_names, feature_indices=feature_indices)

    def subset(self, model):
        """Keep coordinates in nonzero dual terms; constant kernels use no inputs.

        This conservative structural rule does not eliminate cancellations
        between support-vector terms. The kernel offset adds no input dependence.
        """
        self._validate(model)
        if model.degree == 0 or model._gamma == 0:
            return []
        active = _dense(model.dual_coef_)[0] != 0
        return np.flatnonzero(np.any(_dense(model.support_vectors_)[active] != 0, axis=0)).tolist()

    def metadata(self, model):
        metadata = {"kernel": "poly", "degree": int(model.degree),
                    "n_support_vectors": int(model.support_vectors_.shape[0])}
        if isinstance(model, SVC):
            metadata["n_classes"] = int(len(model.classes_))
        return metadata

    def serialize(self, model, *, features):
        self._validate(model)
        support = Vector(tuple(Vector(tuple(Constant(float(value)) for value in row))
                               for row in _dense(model.support_vectors_)))
        dual = Vector(tuple(Constant(float(value)) for value in _dense(model.dual_coef_)[0]))
        i = Index("i")
        kernel = _polynomial_kernel(features, support, i, gamma=model._gamma,
                                    coef0=model.coef0, degree=model.degree)
        score = Binary("+", Constant(float(model.intercept_[0])), Sum(
            "i", model.support_vectors_.shape[0], Binary("*", Element(dual, i), kernel),
        ))
        if isinstance(model, SVC):
            # Public binary coefficients orient nonnegative scores to classes_[1].
            return Conditional(Binary(">=", score, Constant(0)),
                               Label(model.classes_[1]), Label(model.classes_[0]))
        return score
