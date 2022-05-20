import numpy as np
from scipy.special import binom

from antiCPy.trend_extrapolation.cp_segment_fit import CPSegmentFit


class BatchedCPSegmentFit(CPSegmentFit):
    """
    The ``BatchedCPSegmentFit`` is a child class of ``CPSegmentFit``. It can be used to calculate
    the change point configuations with the corresponding segment fit in a batch-wise manner to avoid
    memory errors in the case of high amount of data and change point configurations.
    """

    def __init__(self, x_data, y_data, number_expected_changepoints, num_MC_cp_samples, batchsize=5000,
                 predict_up_to=None, z_array_size=100):

        super().__init__(x_data, y_data, number_expected_changepoints, num_MC_cp_samples, predict_up_to, z_array_size)

        self.batched_D_factor = np.zeros((z_array_size, num_MC_cp_samples))
        self.batched_DELTA_D2_factor = np.zeros((z_array_size, num_MC_cp_samples))
        self.batchsize = batchsize
        self.total_batches = None
        self.batch_D = None
        self.batch_DELTA_D2 = None

    def cp_scan(self, print_sum_control=False, integration_method = 'Riemann sum', print_batch_info=True, config_output = False, prepare_fit=False):
        self.initialize_MC_cp_configurations(print_sum_control=print_sum_control, config_output = config_output)
        storage_configs = np.copy(self.MC_cp_configurations)
        while self.n_MC_samples % self.batchsize != 0:
            self.batchsize -= 1
            if print_batch_info:
                print('Adapt batchsize!')
        self.total_batches = int(self.n_MC_samples / self.batchsize)
        if print_batch_info:
            print('Final batch size: ' + str(self.batchsize))
            print('Total batches: ' + str(self.total_batches))
        for i in range(self.total_batches):
            if print_batch_info:
                print('Batch: ' + str(i + 1) + '/' + str(self.total_batches))
            one_batch_helper = CPSegmentFit(self.x, self.d, self.n_cp, self.batchsize, self.prediction_horizon,
                                            self.z_array.size)
            one_batch_helper.MC_cp_configurations = np.roll(self.MC_cp_configurations, shift=- i * self.batchsize,
                                                            axis=0)[:self.batchsize]
            one_batch_helper.initialize_A_matrices()
            try:
                one_batch_helper.Q_matrix_and_inverse_Q()
            except np.linalg.LinAlgError as err:
                if 'Singular matrix' in str(err):
                    one_batch_helper.MC_cp_configurations = np.roll(self.MC_cp_configurations,
                                                                    shift=- i * self.batchsize, axis=0)[:self.batchsize]
                    one_batch_helper.initialize_A_matrices()
            one_batch_helper.calculate_f0()
            one_batch_helper.calculate_residue()
            one_batch_helper.calculate_marginal_likelihood()
            if prepare_fit:
                for j in range(self.z_array.size):
                    one_batch_helper.initialize_prediction_factors(self.z_array[j])
                    self.batched_D_factor[j, i * self.batchsize: (i + 1) * self.batchsize] = one_batch_helper.D_factor
                    self.batched_DELTA_D2_factor[j,
                    i * self.batchsize: (i + 1) * self.batchsize] = one_batch_helper.DELTA_D2_factor

            self.marginal_log_likelihood[
            i * self.batchsize:(i + 1) * self.batchsize] = one_batch_helper.marginal_log_likelihood

        self.calculate_marginal_cp_pdf(integration_method = integration_method)
        self.calculate_prob_cp(integration_method=integration_method)
        self.MC_cp_configurations = np.copy(storage_configs)

    def fit(self, sigma_multiples=3, print_progress=True, integration_method = 'Riemann sum', config_output = False, print_sum_control=True):
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

        self.cp_scan(print_sum_control=print_sum_control, integration_method = integration_method, config_output = config_output, prepare_fit=True)
        prediction_flag = False
        upper_flag = False
        lower_flag = False
        self.D_array = np.zeros(self.z_array.size)
        self.DELTA_D2_array = np.zeros(self.z_array.size)
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