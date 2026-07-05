"""Tests for initialization utilities."""

from jax import numpy as jnp
from jax import random

from jaxsne import _utils as utils


def test_pca_selects_top_components() -> None:
    """Test that pca keeps the leading principal direction.

    Standardization whitens per-column variance, so the leading component is
    driven by correlation structure: two strongly correlated columns dominate a
    third independent one.
    """
    key = random.key(0)
    base_key, noise_key = random.split(key)
    base = random.normal(base_key, (400,))
    noise = random.normal(noise_key, (400, 2))
    embedded = jnp.stack([base, base + 0.05 * noise[:, 0], noise[:, 1]], 1)
    res = utils.pca(embedded, 1)
    assert res.shape == (400, 1)
    # the leading component lives in the correlated (base) plane
    min_correlation = 0.99
    corr = jnp.corrcoef(res[:, 0], base)[0, 1]
    assert jnp.abs(corr) > min_correlation


def test_pca_full_rank() -> None:
    """Test that pca handles n_components == dim without crashing."""
    key = random.key(0)
    data = random.normal(key, (50, 2))
    res = utils.pca(data, 2)
    assert res.shape == (50, 2)
    assert jnp.all(jnp.isfinite(res))
