"trend_extrapolation" package
--------------------------------

This is the `antiCPy.trend_extrapolation` package. It contains

#. the class `CPSegmentFit` (basic serial implementation),
#. the class `BatchedCPSegmentFit` (strongly parallelized version).

`CPSegmentFit` incorporates all attributes needed to implement the Bayesian non-parametric linear segment fit
which takes into account possible change points (CPs). The basic procedure is described in [vdL14]_ [K14]_
and the nomenclature is chosen congruent to that. Each of the calculation steps is realized
by a class method of `CP_segment_fit`. You can follow the instructions of the cited
papers to interpret the coding. For example, the segment fit can be applied to drift slope
estimate :math:`\hat{\zeta}(t) \equiv y(x)` time series computed with the `antiCPy.early_warnings` module.

The simple serial implementation `CPSegmentFit` can be rather time consuming. A first improvement is to used its
multiprocessing option which computes each CP configuration in parallel with a predefined number of workers.
Additionally, large amounts of CP configurations will without a doubt result in memory errors.
The `BatchedCPSegmentFit` class solves these issues by parallel computation of batches of CP configurations while
each worker only constructs a suitable subset of configurations. This leads to a major computation time improvement and
avoids memory issues for a complicated CP segment fit with an arbitrary number of CPs.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

    .. automodule:: antiCPy.trend_extrapolation.batched_cp_segment_fit
        :members:
        :show-inheritance:

    .. automodule:: antiCPy.trend_extrapolation.cp_segment_fit
        :members:

    batched_configs_helper/batched_configs_helper


Bibliography
^^^^^^^^^^^^


.. [vdL14]	Linden, W., Dose, V., & Toussaint, U. (2014). Bayesian Probability Theory: 
			Applications in the Physical Sciences. Cambridge: Cambridge University Press. 
			doi:10.1017/CBO9781139565608

.. [K14]	A. Klöckner, F. van der Linden, and D. Zimmer, in Proceedings of the 10th International 
			Modelica Conference, March 10-12, 2014, Lund, Sweden (Linköping University Electronic Press, 2014)
