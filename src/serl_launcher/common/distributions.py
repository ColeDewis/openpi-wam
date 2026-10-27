"""Minimal, JAX-only stand-in for the two ``distrax`` distributions hil-serl uses.

hil-serl depends on ``distrax`` for ``MultivariateNormalDiag`` and its
tanh-squashed variant. Adding distrax to this repo would also drag in a newer JAX
(and TFP) than the lockfile pins, so the two distributions are implemented here
directly. The public surface is the same subset hil-serl calls:

    Distribution
    MultivariateNormalDiag(loc, scale_diag)
    TanhMultivariateNormalDiag(loc, scale_diag, low=None, high=None)

Each supports ``sample(seed)``, ``sample_and_log_prob(seed)``, ``log_prob(x)``,
``mode()`` and ``stddev()``. The classes are ``flax.struct`` pytrees so a policy
can return one across a ``jax.jit`` boundary, exactly as with distrax.
"""

from __future__ import annotations

import flax.struct
import jax
import jax.numpy as jnp

_EPS = 1e-6

# Type alias used in the agent signatures (`distrax.Distribution` upstream).
Distribution = flax.struct.PyTreeNode


@flax.struct.dataclass
class MultivariateNormalDiag(flax.struct.PyTreeNode):
    """Diagonal-covariance multivariate normal."""

    loc: jax.Array
    scale_diag: jax.Array

    def log_prob(self, x: jax.Array) -> jax.Array:
        z = (x - self.loc) / self.scale_diag
        return jnp.sum(-0.5 * (jnp.square(z) + jnp.log(2.0 * jnp.pi)) - jnp.log(self.scale_diag), axis=-1)

    def sample(self, seed) -> jax.Array:
        return self.loc + self.scale_diag * jax.random.normal(seed, self.loc.shape)

    def sample_and_log_prob(self, seed):
        x = self.sample(seed)
        return x, self.log_prob(x)

    def mode(self) -> jax.Array:
        return self.loc

    def stddev(self) -> jax.Array:
        return self.scale_diag


@flax.struct.dataclass
class TanhMultivariateNormalDiag(flax.struct.PyTreeNode):
    """``y = tanh(z)`` where ``z ~ N(loc, scale_diag**2)``.

    Optionally rescales the tanh output to ``[low, high]`` (the upstream bijector
    chain is ``rescale(tanh(x))``), which is unused by hil-serl's default policy
    but kept for parity.
    """

    loc: jax.Array
    scale_diag: jax.Array
    low: jax.Array | None = None
    high: jax.Array | None = None

    def _rescale(self, y: jax.Array) -> jax.Array:
        if self.low is None or self.high is None:
            return y
        return (y + 1.0) / 2.0 * (self.high - self.low) + self.low

    def _unrescale(self, y: jax.Array) -> jax.Array:
        if self.low is None or self.high is None:
            return y
        return (y - self.low) / (self.high - self.low) * 2.0 - 1.0

    def _forward(self, z: jax.Array) -> jax.Array:
        return self._rescale(jnp.tanh(z))

    def _inverse(self, y: jax.Array) -> jax.Array:
        return jnp.arctanh(jnp.clip(self._unrescale(y), -1.0 + _EPS, 1.0 - _EPS))

    def _forward_log_det_jacobian(self, z: jax.Array) -> jax.Array:
        # d tanh(z)/dz = 1 - tanh(z)**2; the optional rescale adds a constant.
        logdet = jnp.sum(jnp.log(1.0 - jnp.square(jnp.tanh(z)) + _EPS), axis=-1)
        if self.low is not None and self.high is not None:
            logdet = logdet + jnp.sum(jnp.log(0.5 * (self.high - self.low)), axis=-1)
        return logdet

    def log_prob(self, y: jax.Array) -> jax.Array:
        z = self._inverse(y)
        base = MultivariateNormalDiag(loc=self.loc, scale_diag=self.scale_diag)
        return base.log_prob(z) - self._forward_log_det_jacobian(z)

    def sample(self, seed) -> jax.Array:
        z = self.loc + self.scale_diag * jax.random.normal(seed, self.loc.shape)
        return self._forward(z)

    def sample_and_log_prob(self, seed):
        z = self.loc + self.scale_diag * jax.random.normal(seed, self.loc.shape)
        base = MultivariateNormalDiag(loc=self.loc, scale_diag=self.scale_diag)
        y = self._forward(z)
        return y, base.log_prob(z) - self._forward_log_det_jacobian(z)

    def mode(self) -> jax.Array:
        return self._forward(self.loc)

    def stddev(self) -> jax.Array:
        return self._forward(self.scale_diag)
