import numpy as np

from antiCPy.early_warnings.drift_slope.langevin_estimation import LangevinEstimation

class BinningLangevinEstimation(LangevinEstimation):
	"""
	Child class of the ``LangevinEstimation`` class. Inherits all features and methods from the
	parent class. The class is used to apply a computation shortcut of the implemented resilience
	screening methods by a data binning approach (cf. [Kleinhans2012]_ ).
	Needed attributes and (overloaded) methods are provided.

	.. hint::
		The fitting procedure is faster than the MAP estimation if the windows contain a certain amount of data.

	:param _bin_centers: Attribute that contains the bin centers of the time series.
	:type _bin_centers: One-dimensional numpy array of float
	:param _bin_num: Number of chosen bins for the time series binning procedure.
	:type _bin_num: int
	:param _bin_array: Attribute that contains the limits of each bin.
	:type _bin_array: One-dimensional numpy array of float
	:param _bin_labels: Attribute that contains the bin labels of each data point. Entries reach from
					one to ``_bin_num``.

	:type _bin_labels: One-dimensional numpy array of int
	:param _number_bin_members: Attribute that contains the number of data points of a time series that
					belong to each bin.

	:type _number_bin_members: One-dimensional numpy array of int
	:param _bin_mean_increment: Attribute that contains the mean increments of each bin.
	:type _bin_mean_increment: One-dimensional numpy array of float
	:param _bin_mean_increment_squared: Attribute that contains the squared mean increments of each bin.
	:type _bin_mean_increment_squared: One-dimensional numpy array of float
	:param _bin_MAP: Attribute that contains the maximum a posteriori probability of each bin.
	:type _bin_MAP: One-dimensional numpy array of float
	"""

	def __init__(self, data, time, bin_num, drift_model = '3rd order polynomial',
				diffusion_model = 'constant',
				prior_type = 'Non-informative linear and Gaussian Prior',
				prior_range = None, scales = np.array([4,8])):

		super().__init__(data, time, drift_model,
				diffusion_model, prior_type, prior_range, scales)

		self.antiCPyObject = 'BinningLangevinEstimation'
		self._bin_num = bin_num
		self._bin_array = None
		self._bin_labels = None
		self._number_bin_members = None
		self._bin_centers = None
		self._bin_increment_mean = None
		self._bin_increment_mean_squared = None
		self._bin_MAP = None

	@property
	def bin_centers(self):
		"""
		Contains the center value of the binned data.
		"""
		return self._bin_centers

	@property
	def bin_increment_mean(self):
		"""
		Contains the mean increments per bin.
		"""
		return self._bin_increment_mean

	@property
	def bin_increment_mean_squared(self):
		"""
		Contains the squared mean increment of each bin.
		"""
		return self._bin_increment_mean_squared


	def D1(self):
		"""
		Overloaded drift function in the case of the binning data approach.
		"""
		return self.calc_D1(self.theta, self._bin_centers, self.drift_model)

	def D2(self):
		"""
		Overloaded diffusion function in the case of the binning data approach.
		"""
		return self.calc_D2(self.theta, self._bin_centers, self.diffusion_model)

	@staticmethod
	def calc_bin_neg_log_likelihood(pars, bin_incr_mean, bin_incr_mean_squared, num_bin_members, dt):
		"""
		Calculates the negative logarithmic likelihood function of a bin in the binning data approach.
		"""
		#Log-likelihood-function for bin-wise estimation based on statistical pars
		var = pars[1]**2 * dt
		ML_theta = num_bin_members / 2. * (np.log(2 * np.pi * var)
			+ (bin_incr_mean_squared - 2. * pars [0] * dt * bin_incr_mean
			+ (pars[0])**2 * dt**2) / var)
		return ML_theta

	def bin_neg_log_likelihood(self):
		"""
		Calculates the negative logarithmic likelihood function of a bin in the binning data approach.
		The method calls the static helper function ``calc_bin_neg_log_likelihood(...)``.
		"""
		return self.calc_bin_neg_log_likelihood(np.array([self.D1(), self.D2()]),
											self._bin_increment_mean,
											self._bin_increment_mean_squared,
											self._number_bin_members, self.dt)

	def log_posterior(self,theta):
		"""
		Calculates the logarithmic posterior for a given tuple of parameters :math:`\\underline{\\theta}` .
		"""
		self.theta = theta
		self._bin_MAP = np.zeros(self._bin_num)
		self._bin_MAP = - self.bin_neg_log_likelihood() + __class__.log_prior(self.theta,self.drift_model,self.diffusion_model,self.prior_type,self.prior_range,self.scales)
		return np.sum(self._bin_MAP)


	def neg_log_posterior(self, theta):
		"""
		Calculates the negative logarithmic posterior for a given tuple of parameters :math:`\\underline{\\theta}` .
		"""
		return (-1) * self.log_posterior(theta)


	def _prepare_data(self, data_window = None, bin_num = None, printbool = False):
		"""
		Overloaded function that prepares the data in form of a binning procedure of the time series
		in case of a ``BinningLangevinEstimation`` object. Calls the static method ``_fast_prepare_data(...)``.

		:param data_window: The data on which the binning should be performed.
		:type data_window: One-dimensional numpy array of float.
		:param bin_num: Number of bins in the binning procedure.
		:type bin_num: int
		:param printbool: Decides whether a detailed output of the procedure is printed.
		:type printbool: Boolean
		"""
		if data_window != None:
			self.data_window = data_window
		if bin_num != None:
			self._bin_num = bin_num
		(self._bin_array, self._bin_centers, self._bin_labels, self._number_bin_members,
		self._bin_increment_mean, self._bin_increment_mean_squared) = self._fast_prepare_data(self.data_window, self._bin_num, printbool)


	@staticmethod
	def _fast_prepare_data(data_window, bin_num, printbool):
		"""
		Static helper method.
		"""
		if printbool:
			print('________________')
			print('Perform binning!')
			print('________________')
		bin_array = np.linspace(np.min(data_window), np.max(data_window), bin_num + 1)
		bin_centers = bin_array[:-1] + (bin_array[1] - bin_array[0]) / 2.
		bin_labels = np.digitize(data_window, bin_array, right = False)
		if printbool:
			print('bin_array: ', bin_array)
			print('bin_centers: ', bin_centers)
			print('bin_labels: ', bin_labels)
		if printbool:
			print('shape of bin_labels: ', bin_labels.shape)
			print('________________')
			print('Done!')
			print('________________')
			print('_________________________________________')
			print('Initialize and compute statistics arrays!')
			print('_________________________________________')
		num_bin_members = np.zeros(bin_num)
		bin_incr_mean = np.zeros(bin_num)
		bin_incr_mean_squared = np.zeros(bin_num)
		for j in range(1,bin_num + 1):
			if printbool:
				print('bin_label: ' + str(j))
			bin_indizes = np.where(bin_labels == j)[0]
			if printbool:
				print('bin_indizes: ', bin_indizes)
			num_bin_members[j - 1] = bin_indizes.size
			zero_members = False
			if num_bin_members[j - 1] == 0:
				num_bin_members[j - 1] = 1
				zero_members = True
			if printbool:
				print('bin_indizes size: ', bin_indizes.size)
			print(data_window.size - 1)
			if np.any(bin_indizes == data_window.size - 1):
				bin_indizes = np.delete(bin_indizes, bin_indizes == data_window.size -1)
			incr = data_window[bin_indizes+1] - data_window[bin_indizes]
			if printbool:
				print('Increments:', incr)
			bin_incr_mean[j - 1] = np.sum(incr) / num_bin_members[j - 1]
			if printbool:
				print('mean increments: ', bin_incr_mean)
			bin_incr_mean_squared[j - 1] = np.sum(incr**2) / num_bin_members[j - 1]
			if printbool:
				print('mean squared increments: ', bin_incr_mean_squared)
				print('_____________________________')
				print('Done!')
				print('_____________________________')
			if zero_members:
				num_bin_members[j - 1] = 0
		return bin_array, bin_centers, bin_labels, num_bin_members, bin_incr_mean, bin_incr_mean_squared