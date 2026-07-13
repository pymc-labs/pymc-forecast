# Examples

The notebooks are complete, dims-first forecasting workflows and are executed
with reduced inference settings in CI.

| Example | What it demonstrates |
| --- | --- |
| [BART weekly ridership](forecasting_univariate.ipynb) | Local trend, Fourier seasonality, Student-T observations, and rolling backtests |
| [Hierarchical BART panel](forecasting_hierarchical.ipynb) | Batch `series` dims and partially pooled station priors |
| [Victoria electricity](victoria_electricity.ipynb) | Known future temperature covariates, daily/weekly Fourier terms, and heavy-tailed noise |
| [Exponential smoothing](exponential_smoothing_state_space.ipynb) | A custom innovations state-space model built with PyTensor scans |
| [Scan vs statespace](scan_vs_statespace_local_level.ipynb) | Equivalent latent-scan and marginalized pymc-extras state-space models |

Normal notebook settings are intended for useful posterior inference. The CI
runner caps draws and optimization steps only in an in-memory copy, leaving the
published notebooks unchanged.
