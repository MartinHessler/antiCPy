"early_warnings" subpackage
---------------------------


This is the `antiCPy.early_warnings` subpackage. Up to now it contains a `LangevinEstimation` class that provides a few possibilities to parameterize the drift and diffusion function in terms of polynomials. The
parameters of these functions can be estimated via Markov Chain Monte Carlo (MCMC) sampling and the drift slope can be calculated in a rolling window approach in order to detect changes in the resilience and noise level of a system.


.. automodule:: antiCPy.early_warnings.langevin_estimation
	:members: