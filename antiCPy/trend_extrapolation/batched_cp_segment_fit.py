import numpy as np
import multiprocessing as mp

from antiCPy.trend_extrapolation.cp_segment_fit import CPSegmentFit


class BatchedCPSegmentFit(CPSegmentFit):
    """
	The ``BatchedCPSegmentFit`` is a child class of ``CPSegmentFit``. It can be used to calculate
	the change point configuations with the corresponding segment fit in a batch-wise manner to avoid
	memory errors in the case of high amount of data and change point configurations.
	"""

    def __init__(self, x_data, y_data, number_expected_changepoints, num_MC_cp_samples, batchsize=5000,
                 predict_up_to=None, z_array_size=100, print_batch_info=True):

        super().__init__(x_data, y_data, number_expected_changepoints, num_MC_cp_samples, predict_up_to, z_array_size)

        self.batched_D_factor = None
        self.batched_DELTA_D2_factor = None
        self.batchsize = batchsize
        adapt_batch_flag = True
        while self.n_MC_samples % self.batchsize != 0:
            if print_batch_info and adapt_batch_flag:
                print('Adapt batchsize!')
                adapt_batch_flag = False
            self.batchsize -= 1
        self.total_batches = int(self.n_MC_samples / self.batchsize)
        if print_batch_info:
            print('Final batch size: ' + str(self.batchsize))
            print('Total batches: ' + str(self.total_batches))

    @staticmethod
    def init_batch_execute(marginal_log_likelihood_connector, batched_D_factor_memory_connector,
                           batched_DELTA_D2_factor_connector):
        global shared_memory_dict
        shared_memory_dict = {}
        shared_memory_dict['marginal_LLH_connector'] = marginal_log_likelihood_connector
        shared_memory_dict['batched_D_factor_connector'] = batched_D_factor_memory_connector
        shared_memory_dict['batched_DELTA_D2_connector'] = batched_DELTA_D2_factor_connector

    @staticmethod
    def execute_batch(batch_num, print_batch_info, prepare_fit, total_batches, x, d, n_cp, batchsize,
                      prediction_horizon, MC_cp_configurations, z_array, n_MC_samples):
        if print_batch_info:
            print('Batch: ' + str(batch_num + 1) + '/' + str(total_batches))
        one_batch_helper = CPSegmentFit(x, d, n_cp, batchsize, prediction_horizon,
                                        z_array.size)
        one_batch_helper.MC_cp_configurations = np.roll(MC_cp_configurations, shift=- batch_num * batchsize,
                                                        axis=0)[:batchsize]
        one_batch_helper.initialize_A_matrices()
        try:
            one_batch_helper.Q_matrix_and_inverse_Q()
        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                one_batch_helper.MC_cp_configurations = np.roll(MC_cp_configurations,
                                                                shift=- batch_num * batchsize, axis=0)[:batchsize]
                one_batch_helper.initialize_A_matrices()
        one_batch_helper.calculate_f0()
        one_batch_helper.calculate_residue()
        one_batch_helper.calculate_marginal_likelihood()
        if prepare_fit:
            for j in range(z_array.size):
                one_batch_helper.initialize_prediction_factors(z_array[j])
                batched_D_factor_helper = np.frombuffer(shared_memory_dict['batched_D_factor_connector']).reshape(
                    (z_array.size, n_MC_samples))
                batched_DELTA_D2_factor_helper = np.frombuffer(
                    shared_memory_dict['batched_DELTA_D2_connector']).reshape((z_array.size, n_MC_samples))

                batched_D_factor_helper[j,
                batch_num * batchsize: (batch_num + 1) * batchsize] = one_batch_helper.D_factor
                batched_DELTA_D2_factor_helper[j,
                batch_num * batchsize: (batch_num + 1) * batchsize] = one_batch_helper.DELTA_D2_factor
        marginal_log_likelihood_helper = np.frombuffer(shared_memory_dict['marginal_LLH_connector'])
        marginal_log_likelihood_helper[
        batch_num * batchsize:(batch_num + 1) * batchsize] = one_batch_helper.marginal_log_likelihood

    def cp_scan(self, print_sum_control=False, integration_method='Riemann sum', print_batch_info=True,
                config_output=False, prepare_fit=False, multiprocessing=True, num_processes='half',
                print_CPU_count=False):
        self.initialize_MC_cp_configurations(print_sum_control=print_sum_control, config_output=config_output)
        self.marginal_log_likelihood = mp.RawArray('d', self.n_MC_samples)
        storage_configs = np.copy(self.MC_cp_configurations)
        if multiprocessing:
            if print_CPU_count:
                print('CPU count: ', mp.cpu_count())
            if isinstance(num_processes, str):
                if num_processes == 'all':
                    processes = mp.cpu_count()
                elif num_processes == 'half':
                    processes = int(mp.cpu_count() / 2.)
                else:
                    print('ERROR: String num_processes unknown.')
            elif isinstance(num_processes, int):
                processes = num_processes
            else:
                print('ERROR: Type of num_processes must be int or str.')
            if print_CPU_count:
                print('CPUs used for subprocesses: ' + str(processes))
            with mp.Pool(processes=processes, initializer=self.init_batch_execute, initargs=(
                    self.marginal_log_likelihood, self.batched_D_factor, self.batched_DELTA_D2_factor)) as pool:
                print('Start parallel processing!')
                pool.starmap_async(self.execute_batch, [(batch_num, print_batch_info, prepare_fit, self.total_batches,
                                                         self.x, self.d, self.n_cp, self.batchsize,
                                                         self.prediction_horizon, self.MC_cp_configurations,
                                                         self.z_array, self.n_MC_samples) for batch_num in
                                                        range(self.total_batches)])
                pool.close()
                pool.join()
                print('Parallel processing finished!')
        else:
            self.init_batch_execute(self.marginal_log_likelihood, self.batched_D_factor, self.batched_DELTA_D2_factor)
            for batch_num in range(self.total_batches):
                self.execute_batch(batch_num, print_batch_info, prepare_fit, self.total_batches, self.x, self.d,
                                   self.n_cp, self.batchsize, self.prediction_horizon, self.MC_cp_configurations,
                                   self.z_array, self.n_MC_samples)

        self.calculate_marginal_cp_pdf(integration_method=integration_method)
        self.calculate_prob_cp(integration_method=integration_method)
        self.MC_cp_configurations = np.copy(storage_configs)

    def fit(self, sigma_multiples=3, print_progress=True, integration_method='Riemann sum', config_output=False,
            print_sum_control=True, multiprocessing=True, num_processes='half', print_CPU_count=False):
        '''
		Computes the segmental linear fit of the time series data with integrated change point assumptions
		over the ``z_array`` which contains ``z_array_size`` equidistant data points in the range from the
		first entry of ``x`` up to the ``prediction_horizon``. The fit results and corresponding variances
		are saved in the attributes ``D_array`` and ``DELTA_D2_array``, respectively.

		:param sigma_multiples: Specifies which multiple of standard deviations is chosen to determine the
		    ``upper_uncertainty_bound`` and the ``lower_uncertainty_bound``. Default is 3.
		:type sigma_multiples: float

		:param integration_method: Determines the integration method to compute the change point probability.
			Default is ``'Riemann sum'`` for numerical integration with rectangles. Alternatively, the
			``'Simpson rule'`` can be chosen under the assumption of one change point.
			Sometimes the Simpson rule tends to be unstable. The method should be the same as the
			integration method used in ``calculate_marginal_cp_pdf(...)``.

		:type integration_method: str
		'''
        self.batched_D_factor = mp.RawArray('d', self.z_array.size * self.n_MC_samples)
        self.batched_DELTA_D2_factor = mp.RawArray('d', self.z_array.size * self.n_MC_samples)
        self.cp_scan(print_sum_control=print_sum_control, integration_method=integration_method,
                     config_output=config_output, prepare_fit=True, multiprocessing=multiprocessing,
                     num_processes=num_processes, print_CPU_count=print_CPU_count)
        prediction_flag = False
        upper_flag = False
        lower_flag = False
        self.D_array = np.zeros(self.z_array.size)
        self.DELTA_D2_array = np.zeros(self.z_array.size)
        self.batched_D_factor = np.array(self.batched_D_factor).reshape((self.z_array.size, self.n_MC_samples))
        self.batched_DELTA_D2_factor = np.array(self.batched_DELTA_D2_factor).reshape(
            (self.z_array.size, self.n_MC_samples))
        for i in range(self.z_array.size):
            for m in range(self.n_MC_samples):
                self.D_array[i] += self.prob_cp[m] * self.batched_D_factor[i, m]
                self.DELTA_D2_array[i] += self.prob_cp[m] * self.batched_DELTA_D2_factor[i, m]
            if self.D_array[i] >= 0 and prediction_flag == False:
                self.transition_time = self.z_array[i]
                prediction_flag = True
            if self.D_array[i] + sigma_multiples * np.sqrt(self.DELTA_D2_array[i]) >= 0 and upper_flag == False:
                self.upper_uncertainty_bound = self.z_array[i]
                upper_flag = True
            if self.D_array[i] - sigma_multiples * np.sqrt(self.DELTA_D2_array[i]) >= 0 and lower_flag == False:
                self.lower_uncertainty_bound = self.z_array[i]
                lower_flag = True
