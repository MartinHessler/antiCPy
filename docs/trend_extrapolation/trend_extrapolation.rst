"trend_extrapolation" subpackage
--------------------------------

This is the `antiCPy.trend_extrapolation` subpackage. It contains the class `CPSegmentFit`
which incorporates all attributes needed to implement the Bayesian non-parametric fit 
which takes into account possible change points. The basic procedure is described in [vdL14]_ [K14]_
and the nomenclature is chosen congruent to that. Each of the calculation steps is realized
by a class function of CP_segment_fit. You can follow the instructions of the cited
papers to interpret the coding. For example, the segment fit can be applied to drift slope
estimate :math:`\hat{\zeta}(t) \equiv y(x)` time series computed with the `antiCPy.early_warnings` module.

.. automodule:: antiCPy.trend_extrapolation.cp_segment_fit
	:members:


Bibliography
^^^^^^^^^^^^


.. [vdL14]	Linden, W., Dose, V., & Toussaint, U. (2014). Bayesian Probability Theory: 
			Applications in the Physical Sciences. Cambridge: Cambridge University Press. 
			doi:10.1017/CBO9781139565608

.. [K14]	A. Klöckner, F. van der Linden, and D. Zimmer, in Proceedings of the 10th International 
			Modelica Conference, March 10-12, 2014, Lund, Sweden (Linköping University Electronic Press, 2014)
