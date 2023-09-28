"early_warnings" package
------------------------


This is the `antiCPy.early_warnings` package. It contains two resilience or early warning metrics:

#. the `drift_slope` with basically three fit objects:

    #. the `LangevinEstimation` class that provides possibilities to parameterize the drift and diffusion function in terms of polynomials. The parameters of these functions can be estimated via Markov Chain Monte Carlo (MCMC) sampling or maximum posterior (MAP) estimation and the drift slope can be calculated in a rolling window approach in order to detect changes in the resilience and noise level of a system.
    #. the `BinningLangevinEstimation` class that is designed to perform a bin-wise MAP LangevinEstimation inspired by [Kleinhans2012]_ . In general, the MAP approach is of course much faster than the MCMC approach. In cases of high amount of data per window or a single estimation over a huge dataset the binning approach is computationally more efficient.
    #. the `NonMarkovEstimation` class that introduces a second unobserved Ornstein-Uhlenbeck (OU) process that couples with a constant coefficient into the observed dynamics. The short-term propagator is adjusted following [Willers2021]_ . Furthermore, a time scale separation of adjustable degree can be assumed a priori.
    .. important::
        A fourth `RocketFastResilienceEstimation` is introduced. As superclass it works as a kind of wrapper of the former.
        Basically, it adds the ``fast_resilience_scan(...)`` and ``fast_MAP_resilience_scan(___)`` functionalities to the
        former Fit classes. They involve a strong parallelization of the fitting procedures on multiple rolling windows at
        the `same` time. This improves the computational costs significantly and is the most up-to-date procedure.

#. the `dominant_eigenvalue` which provides modules to

    #. optimize the method parameters in `param_opt`,
    #. perform the dominant eigenvalue estimation in `analysis`,
    #. visualize the parameter optimization and the results in `graphics`,
    #. get started with the package by a `tutorial`.

.. hint::
    The `dominant_eigenvalue` package is procedural up to now. Therefore, its modules contain
    a collection of functions. In consequence, the import has to be done via

    .. code-block::

        # correct import
        import antiCPy.early_warnings.dominant_eigenvalue as dev
        ...
        dev.analysis.AR_EV_calc(...)
        ...
        # not working import
        from antiCPy.early_warnings.dominant_eigenvalue import *

    In case you want to have easy access to a subset of functionalities of `antiCPy` including the `dominant_eigenvalue`
    package you can use the following work arounds:

    .. code-block::

        import antiCPy.early_warnings as ews
        ...
        ews.AR_EV_calc(...)
        ...
        ews.LangevinEstimation(...)
        ...
        # or
        import antiCPy as apy
        ...
                ...
        apy.AR_EV_calc(...)
        ...
        apy.LangevinEstimation(...)
        ...
        apy.CPSegmentFit(...)
        ...



The original implementation [Grziwotz2023]_ uses sequential maps (S-maps) to estimate the Jacobian's coefficients involved in
the procedure. We replaced the S-map approach by a simple autoregression scheme of order :math:`p` (AR(:math:`p`)).

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   drift_slope/drift_slope
   dominant_eigenvalue/dominant_eigenvalue_docu

Bibliography
^^^^^^^^^^^^
.. [Kleinhans2012] David Kleinhans. ”Estimation of drift and diffusion functions from time series data: A maximum likelihood framework”.
        In: Phys. Rev. E 85(2), pp. 026705-026715 (Feb 2012). American Physical Society, 10.1103/PhysRevE.85.026705,
        https://link.aps.org/doi/10.1103/PhysRevE.85.026705.

.. [Willers2021] Clemens Willers and Oliver Kamps. ”Non-parametric estimation of a Langevin model driven by correlated noise”.
        In: Eur. Phys. J. B 94, 149 (2021). https://doi.org/10.1140/epjb/s10051-021-00149-0.

.. [Grziwotz2023] Florian Grziwotz et al. ”Anticipating the occurrence and type of critical transitions”. In: Sci. Adv. 9, eabq4558(2023).
        DOI:10.1126/sciadv.abq4558