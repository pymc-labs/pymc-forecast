"""Optional JAX-native mean-field ADVI backend.

PyMC exposes a JAX-compiled log-density, but compiling the complete ``pm.fit``
update graph with PyTensor's JAX linker is fragile for models whose free
variables have heterogeneous shapes.  This module keeps PyMC's model,
transforms, initial point, and ``MeanField`` approximation, while running the
reparameterized ELBO and Adam updates as one JAX ``lax.scan`` on the selected
accelerator.
"""

import numpy as np
import pymc as pm

from pymc_forecast.exceptions import OptionalDependencyError


def fit_advi_jax(model, *, num_steps: int, learning_rate: float, random_seed=None):
    """Fit PyMC mean-field ADVI with a JAX-native optimization loop.

    Returns the ordinary PyMC ``MeanField`` approximation, so posterior draws
    and every downstream Forecaster operation remain backend-independent.

    .. note::
       This sets JAX's process-wide ``jax_enable_x64`` flag to match
       PyTensor's ``floatX`` (enabling it for ``float64``, disabling it
       otherwise) so JAX does not silently truncate PyMC's initial point.
       The flag is global, so it also affects any other JAX code in the
       same process.
    """
    try:
        import jax
        import jax.numpy as jnp
        import pytensor
        from pymc.sampling.jax import get_jaxified_logp
    except ImportError as err:  # pragma: no cover - exercised without the extra
        raise OptionalDependencyError("jax", "jax", "JAX ADVI") from err

    if num_steps <= 0:
        msg = f"num_steps must be positive, got {num_steps}"
        raise ValueError(msg)

    # Match PyTensor's configured floating-point precision. JAX otherwise
    # silently truncates PyMC's float64 initial point to float32.
    jax.config.update("jax_enable_x64", pytensor.config.floatX == "float64")

    inference = pm.ADVI(model=model, random_seed=random_seed)
    approx = inference.approx
    group = approx.groups[0]
    ordering = {name: (slice_, shape) for name, slice_, shape, _dtype in group.ordering.values()}
    missing = [value.name for value in model.value_vars if value.name not in ordering]
    if missing:  # pragma: no cover - defensive against a PyMC ordering change
        msg = f"JAX ADVI could not map model value variables: {missing}"
        raise ValueError(msg)

    logp_fn = get_jaxified_logp(model)

    def unpack(flat):
        return [
            flat[ordering[value.name][0]].reshape(ordering[value.name][1])
            for value in model.value_vars
        ]

    def loss_fn(mu, rho, key):
        epsilon = jax.random.normal(key, mu.shape, dtype=mu.dtype)
        std = jax.nn.softplus(rho)
        point = mu + std * epsilon
        logq = (-0.5 * epsilon**2 - jnp.log(std) - 0.5 * jnp.log(2.0 * jnp.pi)).sum()
        return logq - logp_fn(unpack(point))

    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    @jax.jit
    def optimize(mu, rho, key):
        def step(carry, index):
            mu, rho, mu_m, rho_m, mu_v, rho_v, last_loss, key = carry
            key, subkey = jax.random.split(key)
            loss, (mu_grad, rho_grad) = jax.value_and_grad(loss_fn, argnums=(0, 1))(mu, rho, subkey)

            # Neutralize non-finite steps instead of poisoning the Adam state:
            # one pathological ELBO sample (an overflow at an extreme draw)
            # would otherwise turn every later iterate into NaN. On such a step
            # the gradient is zeroed, so the parameters only drift by the
            # decaying Adam momentum rather than taking a real update.
            # pm.fit's PyTensor path survives the same event, so match it.
            finite = jnp.isfinite(loss) & jnp.isfinite(mu_grad).all() & jnp.isfinite(rho_grad).all()
            mu_grad = jnp.where(finite, mu_grad, 0.0)
            rho_grad = jnp.where(finite, rho_grad, 0.0)

            mu_m = beta1 * mu_m + (1.0 - beta1) * mu_grad
            rho_m = beta1 * rho_m + (1.0 - beta1) * rho_grad
            mu_v = beta2 * mu_v + (1.0 - beta2) * mu_grad**2
            rho_v = beta2 * rho_v + (1.0 - beta2) * rho_grad**2
            step_number = index + 1
            step_size = (
                learning_rate * jnp.sqrt(1.0 - beta2**step_number) / (1.0 - beta1**step_number)
            )
            mu = mu - step_size * mu_m / (jnp.sqrt(mu_v) + epsilon)
            rho = rho - step_size * rho_m / (jnp.sqrt(rho_v) + epsilon)
            # Carry the last finite loss into the history so a skipped step
            # does not leave a NaN in ``approx.hist`` for the convergence check.
            recorded = jnp.where(finite, loss, last_loss)
            return (mu, rho, mu_m, rho_m, mu_v, rho_v, recorded, key), recorded

        zeros = jnp.zeros_like(mu)
        initial = (mu, rho, zeros, zeros, zeros, zeros, jnp.asarray(jnp.inf, mu.dtype), key)
        final, losses = jax.lax.scan(step, initial, jnp.arange(num_steps))
        return final[0], final[1], losses

    mu_shared, rho_shared = group.params_dict["mu"], group.params_dict["rho"]
    mu = jnp.asarray(mu_shared.get_value())
    rho = jnp.asarray(rho_shared.get_value())
    if isinstance(random_seed, np.random.RandomState):
        seed = int(random_seed.randint(np.iinfo(np.uint32).max))
    else:
        seed = int(np.random.default_rng(random_seed).integers(np.iinfo(np.uint32).max))
    mu, rho, losses = optimize(mu, rho, jax.random.key(seed))

    mu_shared.set_value(np.asarray(mu))
    rho_shared.set_value(np.asarray(rho))
    approx.hist = np.asarray(losses)
    return approx
