import multiprocessing as mp

from antiCPy.trend_extrapolation.cp_segment_fit import CPSegmentFit


from antiCPy.trend_extrapolation.batched_configs_helper.create_configs_helper import *

class BatchedCPSegmentFit(CPSegmentFit):
    """
    The ``BatchedCPSegmentFit`` is a child class of ``CPSegmentFit``. It can be used to calculate
    the change point configurations with the corresponding segment fit in a strongly parallelized batch-wise manner to avoid
    memory errors and speed up computation times significantly in the case of high amount of data and
    change point configurations.

    .. important::

        In any case make sure that you use

        .. code-block::

            import multiprocessing
            ...
            if __name__ == '__main__':
                multiprocessing.set_start_method('spawn').

        Windows should use the method ``'spawn'`` by default. But in general it depends on your system, so it might be better
        to set the option always before using a ``BatchedSegmentFit`` object. If you use a Linux distribution the method to create
        new workers is usually `fork`. This will copy some features of the main process. Amongst others the needed ``lock`` to
        avoid race conditions, might be copied and the new process will freeze. After longer runs this leads to all processes
        getting frozen and killed after some time. You end up with incomplete tasks, but without error message.

	"""

    def __init__(self, x_data, y_data, number_expected_changepoints, num_MC_cp_samples, batchsize=5000,
                 predict_up_to=None, z_array_size=100, print_batch_info=True, efficient_memory_management=False):

        super().__init__(x_data, y_data, number_expected_changepoints, num_MC_cp_samples, predict_up_to, z_array_size)

        self.efficient_memory_management = efficient_memory_management
        if efficient_memory_management:
            self.prob_cp = mp.RawArray('d', num_MC_cp_samples)
            self.cp_prior_pdf = 1. / (self.n_MC_samples)
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
    def init_batch_execute(memory_connectorI, memory_connectorII, memory_connectorIII,
                           memory_connectorIV,memory_management, multiprocessing):
        """
        Internal static method to initialize the workers of a multiprocessing pool.
        """
        global shared_memory_dict
        shared_memory_dict = {}
        shared_memory_dict['memory_management'] = memory_management
        shared_memory_dict['multiprocessing'] = multiprocessing
        if memory_management:
            shared_memory_dict['prob_cp_connector'] = memory_connectorI
            shared_memory_dict['z_prediction_summands_connector'] = memory_connectorII
            shared_memory_dict['normalizing_Z_factor_connector'] = memory_connectorIII
            shared_memory_dict['completion_control_connector'] = memory_connectorIV
        elif not memory_management:
            shared_memory_dict['marginal_LLH_connector'] = memory_connectorI
            shared_memory_dict['batched_D_factor_connector'] = memory_connectorII
            shared_memory_dict['batched_DELTA_D2_connector'] = memory_connectorIII
            shared_memory_dict['completion_control_connector'] = memory_connectorIV

    @staticmethod
    def execute_batch(batch_num, print_batch_info, exact_sum_control, config_output, prepare_fit, total_batches, x, d, n_cp, batchsize,
                      prediction_horizon, z_array, MC_cp_configurations, n_MC_samples, lock, first_round, second_round):
        """
        Working order for the subprocesses. Creates a ``CPSegmentFit'' object of the batch and calculates the corresponding CP pdfs.
        """
        if print_batch_info:
            print('Batch: ' + str(batch_num + 1) + '/' + str(total_batches))
        one_batch_helper = CPSegmentFit(x, d, n_cp, batchsize, prediction_horizon,
                                        z_array.size)
        if shared_memory_dict['memory_management']:
            configs = batched_configs(batch_num, batchsize, x, prediction_horizon, n_cp,
                                      exact_sum_control=exact_sum_control, config_output=config_output)
            one_batch_helper.MC_cp_configurations = configs
        elif not shared_memory_dict['memory_management']:
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
        one_batch_helper.cp_prior_pdf = np.ones(batchsize) / (total_batches * batchsize)
        one_batch_helper.calculate_marginal_likelihood()

        if not shared_memory_dict['memory_management']:
            completion_control = shared_memory_dict['completion_control_connector']
            if shared_memory_dict['multiprocessing']:
                with lock:
                    completion_control.value += 1
            else:
                completion_control.value += 1
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
        elif shared_memory_dict['memory_management']:
            if prepare_fit:
                if first_round:
                    if print_batch_info:
                        print('First round with Batch: ' + str(batch_num + 1) + '/' + str(total_batches))
                    normalization = shared_memory_dict['normalizing_Z_factor_connector']
                    completion_control = shared_memory_dict['completion_control_connector']
                    one_batch_helper.calculate_marginal_cp_pdf()
                    if shared_memory_dict['multiprocessing']:
                        with lock:
                            normalization.value += one_batch_helper.normalizing_Z_factor
                            completion_control.value += 1
                    else:
                        normalization.value += one_batch_helper.normalizing_Z_factor
                        completion_control.value += 1
                elif second_round:
                    if print_batch_info:
                        print('Second round with Batch: ' + str(batch_num + 1) + '/' + str(total_batches))
                    normalization = shared_memory_dict['normalizing_Z_factor_connector']
                    completion_control = shared_memory_dict['completion_control_connector']
                    z_prediction_summands = np.frombuffer(
                        shared_memory_dict['z_prediction_summands_connector'].get_obj()).reshape((z_array.size, 2))
                    if shared_memory_dict['multiprocessing']:
                        marginal_cp_log_pdf = one_batch_helper.marginal_log_likelihood[:] + np.log(
                            one_batch_helper.cp_prior_pdf[:])
                        one_batch_helper.marginal_cp_pdf = 1. / normalization.value * np.exp(marginal_cp_log_pdf)
                        one_batch_helper.calculate_prob_cp()
                        with lock:
                            completion_control.value += 1
                            for j in range(z_array.size):
                                one_batch_helper.initialize_prediction_factors(z_array[j])
                                z_prediction_summands[j, 0] += np.sum(one_batch_helper.prob_cp * one_batch_helper.D_factor)
                                z_prediction_summands[j, 1] += np.sum(
                                    one_batch_helper.prob_cp * one_batch_helper.DELTA_D2_factor)
                        prob_cp_helper = np.frombuffer(shared_memory_dict['prob_cp_connector'])
                        prob_cp_helper[
                        batch_num * batchsize:(batch_num + 1) * batchsize] = one_batch_helper.prob_cp
                    else:
                        completion_control.value += 1
                        marginal_cp_log_pdf = one_batch_helper.marginal_log_likelihood[:] + np.log(
                            one_batch_helper.cp_prior_pdf[:])
                        one_batch_helper.marginal_cp_pdf = 1. / normalization.value * np.exp(marginal_cp_log_pdf)
                        one_batch_helper.calculate_prob_cp()
                        for j in range(z_array.size):
                            one_batch_helper.initialize_prediction_factors(z_array[j])
                            z_prediction_summands[j, 0] += np.sum(one_batch_helper.prob_cp * one_batch_helper.D_factor)
                            z_prediction_summands[j, 1] += np.sum(
                                one_batch_helper.prob_cp * one_batch_helper.DELTA_D2_factor)
                        prob_cp_helper = np.frombuffer(shared_memory_dict['prob_cp_connector'])
                        prob_cp_helper[
                        batch_num * batchsize:(batch_num + 1) * batchsize] = one_batch_helper.prob_cp

            else:
                normalization = shared_memory_dict['normalizing_Z_factor_connector']
                completion_control = shared_memory_dict['completion_control_connector']
                one_batch_helper.calculate_marginal_cp_pdf()
                if shared_memory_dict['multiprocessing']:
                    with lock:
                        normalization.value += one_batch_helper.normalizing_Z_factor
                        completion_control.value += 1
                else:
                    normalization.value += one_batch_helper.normalizing_Z_factor
                    completion_control.value += 1
                prob_cp_helper = np.frombuffer(shared_memory_dict['prob_cp_connector'])
                prob_cp_helper[
                batch_num * batchsize:(batch_num + 1) * batchsize] = one_batch_helper.marginal_log_likelihood



    def cp_scan(self, print_sum_control=False, integration_method='Riemann sum', print_batch_info=True,
                config_output=False, prepare_fit=False, multiprocessing=True, num_processes='half',
                print_CPU_count=False):
        """
        Adapted method from ``CPSegmentFit.cp_scan(...)'' method for strong parallelization and batch structure.
        """
        self.completion_control = mp.Value('i', 0)
        if not self.efficient_memory_management:
            self.initialize_MC_cp_configurations(print_sum_control=print_sum_control, config_output=config_output)
            self.marginal_log_likelihood = mp.RawArray('d', self.n_MC_samples)
            storage_configs = np.copy(self.MC_cp_configurations)
            mp_initargs = (self.marginal_log_likelihood, self.batched_D_factor, self.batched_DELTA_D2_factor, self.completion_control,
                            self.efficient_memory_management, multiprocessing)
            rounds = 1
        elif self.efficient_memory_management:
            self.z_prediction_summands = mp.Array('d', self.z_array_size * 2)  # , lock = True)
            self.normalizing_Z_factor = mp.Value('d', 0.0)
            mp_initargs = (self.prob_cp, self.z_prediction_summands, self.normalizing_Z_factor, self.completion_control,
                           self.efficient_memory_management, multiprocessing)
            if prepare_fit:
                rounds = 2
            else:
                rounds = 1
        first_round = True
        second_round = False
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
            chunksize, extra = divmod(self.total_batches, processes * 4)
            if extra:
                chunksize += 1
            for i in range(rounds):
                with mp.Manager() as manager:
                    lock = manager.Lock()
                    with mp.Pool(processes=processes, initializer=self.init_batch_execute, initargs= mp_initargs) as pool:
                        print('Start parallel processing!')
                        pool.starmap_async(self.execute_batch, [(batch_num, print_batch_info, self.exact_sum_control,config_output, prepare_fit,
                                                                 self.total_batches,
                                                                 self.x, self.d, self.n_cp, self.batchsize,
                                                                 self.prediction_horizon, self.z_array, self.MC_cp_configurations,
                                                                 self.n_MC_samples, lock, first_round, second_round) for batch_num in
                                                                range(self.total_batches)], error_callback = custom_error_callback, chunksize = chunksize)
                        pool.close()
                        pool.join()
                        print('Parallel processing finished!')
                first_round = False
                second_round = True
                print(str(self.completion_control.value) + ' tasks of ' + str(
                    self.total_batches) + ' are executed in round ' + str(i + 1) + ' of ' + str(
                    rounds) + ' rounds.')
                if prepare_fit and not first_round and self.efficient_memory_management:
                    self.completion_control.value = 0
        else:
            first_round = True
            second_round = False
            lock = None
            for i in range(rounds):
                self.init_batch_execute(*mp_initargs)
                for batch_num in range(self.total_batches):
                    self.execute_batch(batch_num, print_batch_info, self.exact_sum_control, config_output, prepare_fit,
                                       self.total_batches, self.x, self.d, self.n_cp, self.batchsize, self.prediction_horizon,
                                       self.z_array, self.MC_cp_configurations, self.n_MC_samples, lock, first_round, second_round)
                first_round = False
                second_round = True
        if self.efficient_memory_management and not prepare_fit:
            if not prepare_fit:
                self.prob_cp = 1. / self.normalizing_Z_factor.value * np.exp(self.prob_cp[:] + np.log(self.cp_prior_pdf))
            else:
                self.prob_cp = np.frombuffer(self.prob_cp)
        elif not self.efficient_memory_management:
            self.calculate_marginal_cp_pdf(integration_method=integration_method)
            self.calculate_prob_cp(integration_method=integration_method)
            self.MC_cp_configurations = np.copy(storage_configs)

    def fit(self, sigma_multiples=3, print_progress=True,  print_batch_info = False, integration_method='Riemann sum', config_output=False,
            print_sum_control=True, multiprocessing=True, num_processes='half', print_CPU_count=False):
        """
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

		:param print_batch_info: If ``True``, computed to total batches are printed. Default is ``False''.
		:type print_batch_info: bool

		:param config_output: If ``True``, the CP configurations of the current batch are printed. Default is ``False''.
		:type config_output: bool

		:param print_sum_control: If ``print_sum_control == True`` it prints whether the exact
                or the approximate MC sum is computed. Default is ``False``.
		:type print_sum_control: bool

		:param multiprocessing: If ``True``, the batches are computed by ``num_processes`` workers in parallel. Default is ``True``.
		:type multiprocessing: bool

		:param num_processes: Default is ``'half'``. If ``half``, almost half of the CPU kernels are used. If  ``'all'``, all CPU kernels
		        are used. If integer number, the defined number of CPU kernels is used for multiprocessing.

		:type num_processes: str, int

		:param print_CPU_count: If ``True``, the total number of available CPU kernels is printed. Default is ``False``.
		:type print_CPU_count: bool
		"""

        prediction_flag = False
        upper_flag = False
        lower_flag = False
        if not self.efficient_memory_management:
            self.batched_D_factor = mp.RawArray('d', self.z_array.size * self.n_MC_samples)
            self.batched_DELTA_D2_factor = mp.RawArray('d', self.z_array.size * self.n_MC_samples)
        self.cp_scan(print_sum_control=print_sum_control, print_batch_info=print_batch_info,
                     integration_method=integration_method, config_output=config_output, prepare_fit=True,
                     multiprocessing=multiprocessing, num_processes=num_processes, print_CPU_count=print_CPU_count)
        if not self.efficient_memory_management:
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
        elif self.efficient_memory_management:
            self.z_prediction_summands = np.array(self.z_prediction_summands).reshape((self.z_array_size, 2))
            for i in range(self.z_array_size):
                if self.z_prediction_summands[i, 0] >= 0 and prediction_flag == False:
                    self.transition_time = self.z_array[i]
                    prediction_flag = True
                if self.z_prediction_summands[i, 0] + sigma_multiples * np.sqrt(
                        self.z_prediction_summands[i, 1]) >= 0 and upper_flag == False:
                    self.upper_uncertainty_bound = self.z_array[i]
                    upper_flag = True
                if self.z_prediction_summands[i, 0] - sigma_multiples * np.sqrt(
                        self.z_prediction_summands[i, 1]) >= 0 and lower_flag == False:
                    self.lower_uncertainty_bound = self.z_array[i]
                    lower_flag = True

def custom_error_callback(error):
    print(error, flush=True)