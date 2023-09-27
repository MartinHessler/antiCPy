.. antiCPy documentation master file, created by
   sphinx-quickstart on Mon Dec 13 10:57:34 2021.


Welcome to `antiCPy's` documentation!
===================================

The package abbreviation **antiCPy** stands for ''**anti**\ cipate **C**\ ritical **P**\ oints (and if you like **C**\
hange **P**\ oints) with **Py**\ thon''. The vision of the `antiCPy` package is designing a package collection of state-of-the-art
early warning measures, leading indicators and time series analysis tools that focus on system stability and
resilience in general as well as algorithms that might be helpful to estimate time horizons of future transitions or resilience changes.
It provides an easy applicable and efficient toolbox

#. to estimate the drift slope :math:`\hat{\zeta}` of a polynomial Langevin equation as an early warning signal via Markov Chain Monte Carlo
   (MCMC) sampling or maximum posterior (MAP) estimation,
#. to estimate a non-Markovian two-time scale polynomial system via MCMC or MAP with the option of a priori activated time scale separation,
#. to estimate the dominant eigenvalue by empiric dynamic modelling approaches like delay embedding and shadow manifolds combined with
   iterated map's linear stability formalism,
#. extrapolate an early warning signal trend to find the probable transition horizon based on the current data information.

Computationally expensive algorithms are implemented both, serially and strongly parallelized to minimize computation times. In case of
the change point trend extrapolation it involves furthermore algorithms that allow for computing of complicated fits with high numbers
of change points without memory errors.
The package aims to provide easily applicable methods and guarantee high flexibility and  access to the derived interim results
for research purposes.

You can find the `package on github <https://github.com/MartinHessler/antiCPy>`_.

Citing `antiCPy`
===============
If you use antiCPy's `drift_slope` measure, please cite

Martin Heßler et al. Bayesian on-line anticipation of critical transitions. New J. Phys. (2022). https://doi.org/10.1088/1367-2630/ac46d4.

If you use antiCPy's `dominant_eigenvalue` instead, please cite

Martin Heßler et al. Anticipation of Oligocene's climate heartbeat by simplified eigenvalue estimation.
arXiv (2023). https://doi.org/10.48550/arXiv.2309.14179

Install
=======

The package can be installed via ::

   pip install antiCPy


.. toctree::
   :maxdepth: 4
   :caption: Contents:
   
   early_warnings/early_warnings
   trend_extrapolation/trend_extrapolation   



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
