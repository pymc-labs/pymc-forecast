"""User-injectable prior configuration for forecasting model objects."""

from collections.abc import Mapping

__all__ = ["PriorConfig"]


class PriorConfig:
    """Mixin for model objects with named, overridable prior specifications.

    Subclasses declare :attr:`default_priors`; callers override any entry with
    the ``priors=`` constructor argument. A specification is normally a
    :class:`pymc_extras.prior.Prior`, but any object exposing
    ``create_variable(name)`` or a callable accepting ``name`` is supported.
    This protocol keeps pymc-extras optional until a user chooses its Prior API.

    Parameters
    ----------
    priors
        Named prior specifications that override or extend
        :attr:`default_priors` for this model instance.
    """

    default_priors: Mapping[str, object] = {}
    """Class-level defaults copied into each model instance."""

    def __init__(self, *, priors: Mapping[str, object] | None = None) -> None:
        self.prior_config = dict(self.default_priors)
        self.prior_config.update(priors or {})

    def create_prior(self, name: str):
        """Create the configured variable named ``name`` in a PyMC context."""
        try:
            prior = self.prior_config[name]
        except KeyError as err:
            available = sorted(self.prior_config)
            msg = f"no prior configured for {name!r}; available priors: {available}"
            raise KeyError(msg) from err

        create_variable = getattr(prior, "create_variable", None)
        if create_variable is not None:
            return create_variable(name)
        if callable(prior):
            return prior(name)
        msg = (
            f"prior {name!r} must expose create_variable(name) or be callable; "
            f"got {type(prior).__name__}"
        )
        raise TypeError(msg)
