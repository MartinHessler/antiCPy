# -*- coding: utf-8 -*-

"""
The ``eigval_tsa.analysis`` module contains the necessary tools for an eigenvalue estimation of a system that is described by a single time series. The optimal parameters for the analysis can be calculated using the module ``eigval_tsa.param_opt``. 
The ``eigval_tsa.analysis`` module contains the functions

	#. ``interaction_coeff_calc(data, order)``
	#. ``jacobian(AR_params)``
	#. ``max_eigenvalue_calc(matrix)``
	#. ``AR_EV_calc(...)``
	#. ``detrend_fct(...)``

These functions are explained in the following.
"""


import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from scipy.ndimage.filters import gaussian_filter


def interaction_coeff_calc(data, order, cov_type):
	"""
	:param data: A one dimensional numpy array containing the times series interval for an autoregressive analysis.
	:param order: The optimal embedding dimension. This is equivalent to the order of the autoregressive model that should be fitted to the data.
	:type data: One dimensional numpy array of floats.
	:type order: Integer with :math:`order \in \mathbb{N} \setminus \{ 0, 1\}`.
	:return: The function first returns the fitted parameters of the autoregression model that is applied to the data. 
		The first entry is the intercept of the model, the following entry the coefficient of the one time lagged contribution, 
		the third entry is of the two time lagged contribution and so on. The function second returns the fit object of the autoregressive model created with ``statsmodels.tsa.ar_model.AR``.
	:rtype: The first return is a one dimensional array of length ``order + 1``. The second return is an fit object of ``statsmodels.tsa.ar_model.AR``.
	"""

	model=AutoReg(data, lags=order) # create AR model of the data
	model_fit=model.fit(cov_type = cov_type) # train the AR model over the data window
	return model_fit.params, model_fit # return the fitted parameters

def jacobian(AR_params):
	"""
	:param AR_params: The parameters that represent the first row of the specially designed Jacobian. 
		The name suggests the AR coefficients of a given time series as they are given by the function ``eigval_tsa.analysis.interaction_coeff_calc`` with the 
		intercept of the AR model as the first entry, the coefficient of the one time lagged contribution as the second entry and so on.
	:type AR_params: One dimensional array of floats.
	:return: The result is a two dimensional numpy array with a Jacobian of the specific design.
	:rtype: A two dimensional numpy array.
	"""

	### Create the Jacobian of the desired form for EV calculations ###
	jacobian_size=AR_params.size
	J=np.zeros((jacobian_size-1,jacobian_size-1))
	J[0,:]=AR_params[1:] # First row filled with the fitted AR coefficients
	J=J+np.diag(np.ones(jacobian_size-2),k=-1) # First subdiagonal with value of unity
	return J

def max_eigenvalue_calc(matrix):
	"""
	:param matrix: A two dimensional square matrix.
	:type matrix: Two dimensional square numpy array.
	:return: The result contains two objects. The first one is a single float containing the absolute maximum of the eigenvalues. The second return object contains 
		a one dimensional array with all eigenvalues of the given matrix. The array is not ordered.
	:rtype: The first object is a float. The second object is a one dimensional array.
	"""

	eigenvalues=np.linalg.eig(matrix)[0] # calculate the eigenvalues of a given matrix
	max_abs_eigval=np.max(np.abs(eigenvalues)) # search for the absolute maximum of these eigenvalues
	return max_abs_eigval, eigenvalues

def AR_EV_calc(gendata, rolling_window_size, order, show_steps = False, detrend = False, detrend_mode = 'linear', gauss_filter_mode = 'reflect', gauss_filter_sigma = 6, gauss_filter_order = 0, gauss_filter_cval = 0.0, gauss_filter_truncate = 4.0, cov_type = 'HC1'):
	"""
	:param gendata: Defines the time series or data interval of which the eigenvalues will be estimated.
	:param rolling_window_size: Defines the number of time series or data elements of gendata that will be used for one eigenvalue estimation in time.
	:param order: Defines the order of the autoregression scheme that is used for the eigenvalue estimation. The optimal order can be estimated with the function ``eigval_tsa.param_opt.false_NN``.
	:param show_steps: If ``True``, the window number of the current calculation is printed. Default is ``False``.
	:param detrend: If ``True``, the data windows are detrended according to the specified detrending parameters. Default is `` False``.
	:param detrend_mode: Each window is linearly detrended for `` detrend = 'linear'`` and with a Gaussian kernel filter via ``detrend = 'gauss_kernel'``. The linear detrending is the default option.
	:param gauss_filter_mode: According to the ``scipy.ndimage.filters.gaussian_filter`` option.
	:param gauss_filter_sigma: According to the ``scipy.ndimage.filters.gaussian_filter`` option.
	:param gauss_filter_order: According to the ``scipy.ndimage.filters.gaussian_filter`` option.
	:param gauss_filter_cval: According to the ``scipy.ndimage.filters.gaussian_filter`` option.
	:param gauss_filter_truncate: According to the ``scipy.ndimage.filters.gaussian_filter`` option.
	:type gendata: One dimensional numpy array.
	:type rolling_window_size: integer
	:type order: integer
	:type show_steps: Boolean.
	:type detrend: Boolean.
	:type detrend_mode: String.
	:type gauss_filter_mode: String.
	:type gauss_filter_sigma: Float.
	:type gauss_filter_order: Integer.
	:type gauss_filter_cval: Float.
	:type gauss_filter_truncate: Float.
	:return: The functions returns two objects. The first object is a one dimensional numpy array that contains the absolute maximum value of the eigenvalues that are calculated for windows
		of the chosen rolling_window_size. The windows are shifted by one in each iteration. The array should have the length ``gendata.size-rolling_window_size+1``. The second object is 
		a two dimensional numpy array with all eigenvalues of each window. All eigenvalues of a given time series window are represented by a row of this matrix.
	:rtype: The first object that is returned is a one dimensional numpy array of the length ``order + 1``. The second object is a two dimensional numpy array of the dimensions ``gendata.size-rolling_window_size+1`` :math:``\times`` ``order``.
	"""

	### define the rolling windows ###
	rolling_window=np.zeros(rolling_window_size)
	### allocate space for the absolute DEV and the EVs ###
	max_eigvals=np.zeros(int(gendata.size-rolling_window_size+1)) # absolute DEVs
	eigenvalues=np.zeros((int(gendata.size-rolling_window_size+1), order), dtype=complex) # EVs

	### iteration over the rolling windows ###
	for i in range(int(gendata.size-rolling_window_size+1)):
	### show window steps ###
		if show_steps:
			print('Window number: ' + str(i))
		### definition of the rolling windows in each step ###
		rolling_window=np.roll(gendata, shift=-i)[0:rolling_window_size]
		### Compute detrending if specified ###
		if detrend:
			rolling_window = detrend_fct(rolling_window, detrend_mode, gauss_filter_mode, gauss_filter_sigma, gauss_filter_order, gauss_filter_cval, gauss_filter_truncate)
	### calculation of the absolute DEV and EV in each step ###
		max_eigvals[i], eigenvalues[i,:]=max_eigenvalue_calc(jacobian(interaction_coeff_calc(rolling_window, order, cov_type)[0]))
	return max_eigvals, eigenvalues

def detrend_fct(gendata, detrend_mode = 'linear', gauss_filter_mode = 'reflect', gauss_filter_sigma = 6, gauss_filter_order = 0, gauss_filter_cval = 0.0, gauss_filter_truncate = 4.0):
	"""
	:param gendata: Time series that is to be detrended.
	:param detrend_mode: Each window is linearly detrended for ``detrend = 'linear'`` and with a Gaussian kernel filter via ``detrend = 'gauss_kernel'``. The linear detrending is the default option.
	:param gauss_filter_mode: According to the ``scipy.ndimage.filters.gaussian_filter`` option.
	:param gauss_filter_sigma: According to the ``scipy.ndimage.filters.gaussian_filter`` option.
	:param gauss_filter_order: According to the ``scipy.ndimage.filters.gaussian_filter`` option.
	:param gauss_filter_cval: According to the ``scipy.ndimage.filters.gaussian_filter`` option.
	:param gauss_filter_truncate: According to the ``scipy.ndimage.filters.gaussian_filter`` option.
	:type gendata: One dimensional numpy array.
	:type rolling_window_size: integer
	:type order: integer
	:type show_steps: Boolean.
	:type detrend: Boolean.
	:type detrend_mode: String.
	:type gauss_filter_mode: String.
	:type gauss_filter_sigma: Float.
	:type gauss_filter_order: Integer.
	:type gauss_filter_cval: Float.
	:type gauss_filter_truncate: Float.
	:return: With ``detrend_mode = 'linear'`` the linear detrended data is returned in form of a one dimensional numpy array. With ``detrend_mode = 'gaussian_kernel'`` the modeled slow trend of the Gaussian filter is subtracted from the data. The detrended version is returned in form of a one dimensional numpy array as first arguement. In a second arguement the estimated slow trend is returned in form of a one dimensional numpy array.
	:rtype: All possible return objects are one dimensional numpy arrays.
	"""
	if detrend_mode == 'linear':
		degree = 1
		dummy_fit_time = np.arange(gendata.size)
		popt = np.polyfit(dummy_fit_time, gendata, deg = degree)
		return gendata - ( popt[0] * dummy_fit_time + popt[1] ) 
	if detrend_mode == 'gauss_kernel':
		slow_trend = gaussian_filter(gendata, gauss_filter_sigma, order = gauss_filter_order, output=None, mode= gauss_filter_mode, cval= gauss_filter_cval, truncate= gauss_filter_truncate)
		detrended_data = gendata - slow_trend
		return detrended_data# , slow_trend