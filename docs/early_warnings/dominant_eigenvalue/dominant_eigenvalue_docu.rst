
.. _The dominant_eigenvalue subpackage documentation:

The "dominant_eigenvalue" package
---------------------------------

The package is designed by procedural programming and contains various functions to
optimize the method's parameters, perform the dominant eigenvalue analysis and visualize
the optimization and the analysis results. The package provides useful tools to estimate the eigenvalues
and absolute dominant eigenvalues of a system that is described by a single time series. The package
contains three modules with the following functions:

	#.  dominant_eigenvalue.param_opt

			#. embedding_attractor_reconstruction(data, E, index_shift)
			#. false_NN(data, time, index_shift, start_order = 1, end_order = 15, NN_threshold = 30)
			#. various_R_threshold_fnn(data, index_shift= 1, start_threshold = 15, end_threshold = 50, start_order = 1, end_order = 15, save = False, save_name = 'fnn_R_threshold_series_default00.npy')
			#. avg_distance_from_diagonal(data, E, start_lag = 1, end_lag = 10, image = False)

	#.	dominant_eigenvalue.analysis

			#. interaction_coeff_calc(data, order)
			#. jacobian(AR_params)
			#. max_eigenvalue_calc(matrix)
			#. AR_EV_calc(gendata, rolling_window_size, order)
			#. detrend_fct(...)

	#.	dominant_eigenvalue.graphics

			#. abs_max_eigval_plot(...)
			#. prep_plot_imaginary_plane(eigvals)
			#. max_eigval_gauss_plot(...)
			#. plot_fnn(...)
			#. plot_avg_DD(...)

The first `dominant_eigenvalue.param_opt` module contains basic tools to optimize the parameters for the
eigenvalue estimation. The eigenvalue estimation tools can be found in the `dominant_eigenvalue.analysis` module.
The third `dominant_eigenvalue.graphics` module provides some functions to plot the results of the analysis.

.. automodule:: antiCPy.early_warnings.dominant_eigenvalue.__init__
.. highlight:: python

.. toctree:: 
   :maxdepth: 1

   param_opt_API
   analysis_API
   graphics_API
   dominant_eigenvalue_maths
   dominant_eigenvalue_tutorial


