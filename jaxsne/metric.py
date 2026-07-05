"""Metrics for jaxsne.

A metric function is a distance-like measure between points in a space. SNE only
needs a non-negative, symmetric distance that is zero for identical points; it
does not rely on the triangle inequality, so metrics need not be proper
[metrics](https://en.wikipedia.org/wiki/Metric_space). The shipped `cosine`, for
example, is an angular dissimilarity that violates the triangle inequality. To
work with this library a metric must satisfy:

1. It must be non-negative and symmetric, and zero for identical points (after
   any projection it applies, see rule 4).
2. It must treat the last dimension as the dimension of the points.
3. It must be [jax jit-able](https://docs.jax.dev/en/latest/jit-compilation.html).
4. It must apply to points in R^d. If it actually applies to some subset of
   points, it should first project into a space of the same dimension. Such a
   projection may be undefined at singular points (e.g. `cosine` at the origin,
   where the direction is undefined).
"""

from ._metric import Metric, cosine, euclidean, poincare

__all__ = ("cosine", "euclidean", "poincare", "Metric")
