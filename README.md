[![DOI](https://zenodo.org/badge/440811484.svg)](https://zenodo.org/badge/latestdoi/440811484) [![Documentation Status](https://readthedocs.org/projects/anticpy/badge/?version=latest)](https://anticpy.readthedocs.io/en/latest/?badge=latest) [![PyPI version](https://badge.fury.io/py/antiCPy.svg)](https://badge.fury.io/py/antiCPy) [![GitHub release](https://img.shields.io/github/release/MartinHessler/antiCPy?color)](https://github.com/MartinHessler/antiCPy) [![license](https://img.shields.io/github/license/MartinHessler/antiCPy?color=blue)](https://img.shields.io/github/MartinHessler/antiCPy/blob/main/LICENSE)

antiCPy
=======

The package abbreviation **antiCPy** stands for ''**anti**cipate **C**ritical **P**oints (and if you like **C**hange **P**oints) 
with **Py**thon''. The vision of the **antiCPy** package is designing a package collection of state-of-the-art
early warning measures, leading indicators and time series analysis tools that focus on system stability and
resilience in general as well as algorithms that might be helpful to estimate time horizons of future transitions or resilience changes.
It provides an easy applicable and efficient toolbox

1. to estimate the drift slope <img src="https://render.githubusercontent.com/render/math?math=\hat{\zeta}"> of a polynomial Langevin equation as an early warning signal via Markov Chain Monte Carlo
   (MCMC) sampling or maximum posterior (MAP) estimation,
2. to estimate a non-Markovian two-time scale polynomial system via MCMC or MAP with the option of a priori activated time scale separation,
3. to estimate the dominant eigenvalue by empiric dynamic modelling approaches like delay embedding and shadow manifolds combined with
   iterated map's linear stability formalism,
4. extrapolate an early warning signal trend to find the probable transition horizon based on the current data information.

Computationally expensive algorithms are implemented both, serially and strongly parallelized to minimize computation times. In case of
the change point trend extrapolation it involves furthermore algorithms that allow for computing of complicated fits with high numbers
of change points without memory errors.
The package aims to provide easily applicable methods and guarantee high flexibility and access to the derived interim results
for research purposes.

Citing antiCPy
==============

If you use **antiCPy's** `drift_slope` measure, please cite

Martin Heßler et al. Bayesian on-line anticipation of critical transitions. New J. Phys. (2022). https://doi.org/10.1088/1367-2630/ac46d4.

If you use **antiCPy's** `dominant_eigenvalue` instead, please cite

Martin Heßler et al. Anticipation of Oligocene's climate heartbeat by simplified eigenvalue estimation.
arXiv (2023). https://doi.org/10.48550/arXiv.2309.14179

Documentation
=============

You can find the [documentation on read the docs](https://anticpy.readthedocs.io/en/latest/).

Install
=======

The package can be installed via

```
pip install antiCPy
```

Related publications
====================
Up to now the package is accompanied by
- the publication [Efficient Multi-Change Point Analysis to Decode Economic Crisis Information from the S&P500 Mean Market Correlation](https://www.mdpi.com/1099-4300/25/9/1265),
- the publication [Memory Effects, Multiple Time Scales and Local Stability in Langevin Models of the S&P500 Market Correlation](https://www.mdpi.com/1099-4300/25/9/1257),
- the publication [Identifying dominant industrial sectors in market states of the S&P 500 financial data](https://iopscience.iop.org/article/10.1088/1742-5468/accce0),
- the publication [Quantifying resilience and the risk of regime shifts under strong correlated noise](https://academic.oup.com/pnasnexus/article/2/2/pgac296/6960580),
- the publication [Bayesian on-line anticipation of critical transitions](https://iopscience.iop.org/article/10.1088/1367-2630/ac46d4),

- the preprint [Anticipation of Oligocene's climate heartbeat by simplified eigenvalue estimation](https://arxiv.org/abs/2309.14179),
- the preprint [Estimating tipping risk factors in complex systems with application to power outage data](https://arxiv.org/abs/2212.06780).