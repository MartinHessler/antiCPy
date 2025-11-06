[![DOI](https://zenodo.org/badge/440811484.svg)](https://zenodo.org/badge/latestdoi/440811484) [![Documentation Status](https://readthedocs.org/projects/anticpy/badge/?version=latest)](https://anticpy.readthedocs.io/en/latest/?badge=latest) [![PyPI version](https://badge.fury.io/py/antiCPy.svg)](https://badge.fury.io/py/antiCPy) [![GitHub release](https://img.shields.io/github/release/MartinHessler/antiCPy?color)](https://github.com/MartinHessler/antiCPy) [![license](https://img.shields.io/github/license/MartinHessler/antiCPy?color=blue)](https://img.shields.io/github/MartinHessler/antiCPy/blob/main/LICENSE) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/MartinHessler/antiCPy/HEAD?filepath=examples/Quickstart_Resilience_and_Change_Point_Analysis_with_antiCPy.ipynb)

antiCPy
=======

The package abbreviation **antiCPy** stands for ''**anti**cipate **C**ritical **P**oints (and if you like **C**hange **P**oints) 
with **Py**thon''. The vision of the **antiCPy** package is designing a package collection of state-of-the-art
early warning measures, leading indicators and time series analysis tools that focus on system stability and
resilience in general as well as algorithms that might be helpful to estimate time horizons of future transitions or resilience changes.
It provides an easy applicable and efficient toolbox

1. to estimate the drift slope $\hat{\zeta}$ of a polynomial Langevin equation as an early warning signal via Markov Chain Monte Carlo
   (MCMC) sampling or maximum posterior (MAP) estimation,
2. to estimate a non-Markovian two-time scale polynomial system via MCMC or MAP with the option of a priori activated time scale separation,
3. to estimate the dominant eigenvalue by empiric dynamic modelling approaches like delay embedding and shadow manifolds combined with
   iterated map's linear stability formalism,
4. to extrapolate an early warning signal trend to find the probable transition horizon based on the current data information.

Computationally expensive algorithms are implemented both, serially and strongly parallelized to minimize computation times. In case of
the change point trend extrapolation it involves furthermore algorithms that allow for computing of complicated fits with high numbers
of change points without memory errors.
The package aims to provide easily applicable methods and guarantee high flexibility and access to the derived interim results
for research purposes.

![An illustration of the drift slope procedure.](https://github.com/MartinHessler/antiCPy/blob/main/images/compound_BLE_illustration.jpg?raw=true)

Quickstart
==========
Launch the interactive tutorial in your browser using Binder:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/MartinHessler/antiCPy/HEAD?filepath=examples/Quickstart_Resilience_and_Change_Point_Analysis_with_antiCPy.ipynb)


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
Up to now the package is accompanied by:
- the publication [Quantifying local stability and noise levels from time series in the US Western Interconnection blackout on 10th August 1996](https://www.nature.com/articles/s41467-025-60877-0). In: Nature Communications 16, 6246 (2025). DOI: 10.1038/s41467-025-60877-0.
- the publication [Efficient Multi-Change Point Analysis to Decode Economic Crisis Information from the S&P500 Mean Market Correlation](https://www.mdpi.com/1099-4300/25/9/1265). In:
Entropy 25.9, 1265 (Aug. 2023). DOI: 10.3390/e25091265.
- the publication [Memory Effects, Multiple Time Scales and Local Stability in Langevin Models of the S&P500 Market Correlation](https://www.mdpi.com/1099-4300/25/9/1257). In: Entropy
25.9, 1257 (Aug. 2023). DOI: 10.3390/e25091257.
- the publication [Identifying dominant industrial sectors in market states of the S&P 500 financial data](https://iopscience.iop.org/article/10.1088/1742-5468/accce0). In: Journal of Statistical Mechanics: Theory
and Experiment 2023.4, 043402 (Apr. 2023). DOI: 10.1088/1742-5468/accce0.
- the publication [Quantifying resilience and the risk of regime shifts under strong correlated noise](https://academic.oup.com/pnasnexus/article/2/2/pgac296/6960580). In: PNAS Nexus 2.2, pgac296 (Dec. 2022). ISSN: 2752-6542. DOI:
10.1093/pnasnexus/pgac296.
- the publication [Bayesian on-line anticipation of critical transitions](https://iopscience.iop.org/article/10.1088/1367-2630/ac46d4). In:
New Journal of Physics 24, 063021 (Dec. 2021). DOI: 10.1088/1367-2630/ac46d4.

- the preprint [Do Inner Greenland's Melt Rate Dynamics Approach Coastal Ones?](https://arxiv.org/abs/2411.07248). Preprint at arXiv (Oct. 2024). arXiv:2411.07248.
- the preprint [Anticipation of Oligocene's climate heartbeat by simplified eigenvalue estimation](https://arxiv.org/abs/2309.14179). Preprint at arXiv (Sept. 2023). arXiv:2309.14179.
- the preprint [Quantifying Tipping Risks in Power Grids and beyond](https://arxiv.org/abs/2212.06780). Preprint at arXiv (Dec. 2023). arXiv:2212.06780.


Further publications
====================
Further research contributions indirectly related to **antiCPy**:
- the publication [Anticipating the occurrence and type of critical transitions](https://www.science.org/doi/10.1126/sciadv.abq4558). In:
Science Advances 9.1, eabq4558 (Jan. 2023). doi: 10.1126/sciadv.abq4558.