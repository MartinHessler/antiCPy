import numpy as np
import matplotlib
import matplotlib.pylab as plt
from matplotlib.animation import FuncAnimation
from scipy import optimize
from scipy.ndimage import gaussian_filter
import scipy.stats as cpy
import emcee
import multiprocessing as mp
import ipyparallel as ipp
import math as mh

from antiCPy.early_warnings.drift_slope.rocket_fast_resilience_estimation import RocketFastResilienceEstimation
from .summary_statistics_helper import summary_statistics_helper

class LangevinEstimation(RocketFastResilienceEstimation):
    '''
    The ``LangevinEstimation`` class includes tools to estimate a polynomial Langevin equation with various
    drift and diffusion terms and provides the drift slope :math:`\zeta` as a resilience measure.

    :param data: A one dimensional numpy array containing the times series to analyse.
    :type data: One dimensional numpy array of floats
    :param time: A one dimensional numpy array containing the time samples of the time series data.
    :type time: One dimensional numpy array of floats
    :param drift_model: Defines the drift model. Default is ``'3rd order polynomial'``.
                        Additionally, a ``'first order polynomial'`` can be chosen.
    :type drift_model: str
    :param diffusion_model: Defines the diffusion model. Default is ``'constant'``.
                        Additionally, a ``'first order polynomial'`` can be chosen.
    :type diffusion_model: str
    :param prior_type: Defines the used prior to calculate the posterior distribution of the data.
                        Default is ``'Non-informative linear and Gaussian Prior'``. A flat prior can be chosen
                        with ``'Flat Prior'``.
    :type drift_type: str
    :param prior_range: Customize the prior range of the estimated polynomial parameters :math:`\theta_i`
                        starting for :math:`\theta_0` in row with index 0 and upper limits in the first column.
                        Default is ``None`` which means prior ranges of -50 to 50 for the drift parameters and
                        0 to 50 for constant diffusion terms.
    :type prior_range: Two-dimensional numpy array of floats, optional
    :param scales: Two tailed percentiles to create credibility bands of the estimated measures.
    :type scales: One-dimensional numpy array with two float entries
    :param theta: Array that contains the estimated drift and diffusion parameters with drift and ending with diffusion.
                    The lowest order is mentioned first.
    :type theta: One-dimensional numpy array of floats
    :param ndim: Total number of parameters to estimate.
    :type ndim: int
    :param nwalkers: Number of MCMC chains.
    :type nwalkers: int
    :param nsteps: Length of Markov chains.
    :type nsteps: int
    :param nburn: Length of burn in period of the MCMC chains.
    :type nburn: int
    :param data_size: Length of the data array.
    :type data_size: int
    :param dt: Time intervall of the equidistant time array.
    :type dt: float
    :param drift_slope: Array with the current drift slope estimate in the 0th component.
                        Component 1, 2 and 3, 4 contain the upper and lower bound of the credibility intervals
                        defined by the ``scales`` variable.
    :type drift_slope: One-dimensional numpy array of floats
    :param noise_level_estimate: Array with the noise level estimate in the 0th component.
                        Component 1, 2 and 3, 4 contain the upper and lower bound of the credibility intervals
                        defined by the ``scales`` variable.
    :type noise_level_estimate: One-dimenional numpy array of floats
    :param loop_range: Array with the start indices of the time array for each rolled time window.
    :type loop_range: One-dimensional (``data_size - window_size / window_shift + 1``) numpy array of integers
    :param window_size: Size of rolling windows.
    :type window_size: int
    :param data_window: Array that contains data of a certain interval of the dataset in ``data``.
    :type data_window: One-dimensional numpy array of floats
    :param time_window: Array that contains time of a certain interval of the dataset in ``data``.
    :type time_window: One-dimensional numpy array of floats
    :param increments: Array that contains the increments from :math:`t` to :math:`t+1`.
    :type increments: One-dimensional numpy array of floats
    :param joint_samples: Array that contains drawed parameter samples from their joint posterior probability
                         density function that is estimated from the data in the current ``data_window``.
    :type joint_samples: Two-dimensional (``ndim, num_of_joint_samples``) numpy array of floats
    :param slope_estimates: Estimates of the drift slope on the current ``data_window``.
    :type slope_estimates: One-dimensional numpy array of floats
    :param fixed_point_estimate: Fixed point estimate calculated as the mean of the current ``data_window``.
    :type fixed_point_estimate: float
    :param starting_guesses: Array that contains the initial guesses of the MCMC sampling.
    :type starting_guesses: Two-dimensional numpy array of floats
    :param theta_array: Array that contains the drift and diffusion parameters sampled with MCMC.
    :type theta_array: Two-dimensional numpy array of floats
    :param slope_storage: Array that contains the drift slope estimates :math:`\hat{\zeta}`
                        of the rolling windows. The columns encode each rolling window with the drift
                        slope estimate :math:`\hat{\zeta}` in the 0th row and in rows 1,2 and 3,4
                        the upper and lower bound of the credibility intervals defined by ``scales``.
    :type slope_storage: Two-dimensional (``5, loop_range.size``) numpy array of floats
    :param noise_level_storage:  Array that contains the noise level estimates :math:`\hat{\sigma}`
                        of the rolling windows. The columns encode each rolling window with the noise level
                        estimate :math:`\hat{\sigma}` in the 0th row and in rows 1,2 and 3,4
                        the upper and lower bound of the credibility intervals defined by ``scales``.
    :type noise_level_storage: Two-dimensional (``5, loop_range.size``) numpy array of floats
    :param slope_kernel_density_obj: Kernel density object of ``scipy.gaussian_kde(...)`` of the slope
                        posterior created with a Gaussian kernel and a bandwidth computed by silverman's rule.
    :type slope_kernel_density_obj: ``scipy`` kernel density object of the ``scipy.gaussian_kde(...)`` function.
    :param noise_kernel_density_obj: Kernel density object of ``scipy.gaussian_kde(...)`` of the noise level
                        posterior created with a Gaussian kernel and a bandwidth computed by silverman's rule.
    :type noise_level_density_obj: ``scipy`` kernel density object of the ``scipy.gaussian_kde(...)`` function.
    :param slow_trend: Contains the subtracted slow trend if a detrending is applied to the whole data
                        or each window separately.
    :type slow_trend: One-dimensional numpy array of floats.
    :param detrending_of_whole_dataset: Default is ``None``. If ``'linear'`` the whole ``data`` is detrended linearly. If
                        ``Gaussian kernel smoothed`` a Gaussian smoothed curve is subtracted from the whole ``data``.
                        The parameters of the kernel smoothing can be set by ``gauss_filter_mode``, ``gauss_filter_sigma``,
                        ``gauss_filter_order``, ``gauss_filter_cval`` and ``gauss_filter_truncate``.
    :type detrending_of_whole_dataset: str
    :param gauss_filter_mode: According to the ``scipy.ndimage.filters.gaussian_filter`` option.
    :type gauss_filter_mode: str
    :param gauss_filter_sigma: According to the ``scipy.ndimage.filters.gaussian_filter`` option.
    :type gauss_filter_sigma: float
    :param gauss_filter_order: According to the ``scipy.ndimage.filters.gaussian_filter`` option.
    :type gauss_filter_order: int
    :param gauss_filter_cval: According to the ``scipy.ndimage.filters.gaussian_filter`` option.
    :type gauss_filter_cval: float
    :param gauss_filter_truncate: According to the ``scipy.ndimage.filters.gaussian_filter`` option.
    :type gauss_filter_truncate: float
    :param plot_detrending: Default is ``False``. If ``True``, the ``self.data`` as well as the
                        ``self.slow_trend`` and the detrended version are shown.
    :type plot_detrending: bool
    '''

    def __init__(self, data, time, drift_model='3rd order polynomial', diffusion_model='constant',
                 prior_type='Non-informative linear and Gaussian Prior', prior_range=None, scales=np.array([4,8]),
                 detrending_of_whole_dataset = None, gauss_filter_mode = 'reflect',
                    gauss_filter_sigma = 6, gauss_filter_order = 0, gauss_filter_cval = 0.0,
                    gauss_filter_truncate = 4.0, plot_detrending = False):

        super().__init__('LangevinEstimation')

        self.drift_model = drift_model
        self.diffusion_model = diffusion_model
        if drift_model == '3rd order polynomial':
            self.num_drift_params = 4
        elif drift_model == 'first order polynomial':
            self.num_drift_params = 2
        else:
            print('ERROR: Drift model is not defined!')
        if diffusion_model == 'constant':
            self.num_diff_params = 1
        elif diffusion_model == 'first order polynomial':
            self.num_diff_params = 2
        else:
            print('ERROR: Diffusion model is not defined!')
        self.ndim = self.num_drift_params + self.num_diff_params
        self.nwalkers = None
        self.nsteps = None
        self.nburn = None
        if np.all(prior_range == None) and drift_model == '3rd order polynomial' and diffusion_model == 'constant':
            self.prior_range = np.array([[50., -50.], [50., -50.], [50., -50.], [50., -50.], [50., 0.]])
        elif np.all(prior_range == None) and drift_model == 'first order polynomial' and diffusion_model == 'constant':
            self.prior_range = np.array([[50., -50.], [50., -50.], [50., 0.]])
        else:
            self.prior_range = prior_range
        if prior_type == 'Non-informative linear and Gaussian Prior':
            self.scales = scales
        self.prior_type = prior_type
        self.theta = np.zeros(self.ndim)
        self.slow_trend = None
        if detrending_of_whole_dataset == None:
            self.data = data
        else:
            self.data, self.slow_trend = self.detrend(time, data, detrending_of_whole_dataset,
                                                           gauss_filter_mode,gauss_filter_sigma,
                                                           gauss_filter_order,gauss_filter_cval,
                                                           gauss_filter_truncate, plot_detrending)
        self.data_size = data.size
        self.time = time
        self.dt = time[1] - time[0]
        self.drift_slope = np.zeros(5)
        self.noise_level_estimate = np.zeros(5)
        self.loop_range = None
        self.window_size = None
        self.data_window = None
        self.time_window = None
        self.increments = None
        self.joint_samples = None
        self.slope_estimates = None
        self.fixed_point_estimate = None
        self.starting_guesses = None
        self.theta_array = None
        self.slope_storage = None
        self.noise_level_storage = None
        self.slope_kernel_density_obj = None
        self.noise_kernel_density_obj = None

    @staticmethod
    def detrend(time, data, detrend_mode = 'linear', gauss_filter_mode = 'reflect',
                    gauss_filter_sigma = 6, gauss_filter_order = 0, gauss_filter_cval = 0.0,
                    gauss_filter_truncate = 4.0, plot_detrending = False):
        """
        Method to apply a linear or Gaussian kernel smoothed detrending to the
        whole data or to each data window separately.

        :param time: Time points of the time series.
        :type time: One-dimensional numpy array of floats.
        :param data: Time series that is to be detrended.
        :type data: One-dimensional numpy array of floats.
        :param detrend_mode: Each window is linearly detrended for ``detrend = 'linear'`` and with a Gaussian kernel filter via ``detrend = 'gauss_kernel'``. The linear detrending is the default option.
        :type detrend_mode: str
        :param gauss_filter_mode: According to the ``scipy.ndimage.filters.gaussian_filter`` option.
        :type gauss_filter_mode: str
        :param gauss_filter_sigma: According to the ``scipy.ndimage.filters.gaussian_filter`` option.
        :type gauss_filter_sigma: float
        :param gauss_filter_order: According to the ``scipy.ndimage.filters.gaussian_filter`` option.
        :type gauss_filter_order: int
        :param gauss_filter_cval: According to the ``scipy.ndimage.filters.gaussian_filter`` option.
        :type gauss_filter_cval: float
        :param gauss_filter_truncate: According to the ``scipy.ndimage.filters.gaussian_filter`` option.
        :type gauss_filter_truncate: float
        :return: The first return object contains the detrended data, the second one the subtracted slow trend.
        :rtype: One-dimensional numpy arrays.
        """
        if detrend_mode == 'linear':
            degree = 1
            popt = np.polyfit(time, data, deg = degree)
            slow_trend = popt[0] * time + popt[1]
            detrended_data = data - slow_trend
        elif detrend_mode == 'Gaussian kernel smoothed':
            slow_trend = gaussian_filter(data, gauss_filter_sigma, order = gauss_filter_order,
                                         output=None, mode= gauss_filter_mode, cval= gauss_filter_cval,
                                         truncate= gauss_filter_truncate)
            detrended_data = data - slow_trend
        else:
            print('ERROR: Type of detrending unknown.')
        if plot_detrending:
            plt.close()
            check_detrending_figure, ax = plt.subplots()
            ax.plot(time, data, label='data')
            ax.plot(time, slow_trend, label='slow trend')
            ax.plot(time, detrended_data, label='detrended data')
            ax.legend()
            plt.show()
            plt.close(check_detrending_figure)
        return detrended_data, slow_trend

    def third_order_polynom_slope_in_fixed_point(self):
        '''
        Calculate the slope_estimates of a third order polynomial drift function with joint_samples of the
        posterior distribution around a fixed point estimate given as the data mean of the current window.
        '''

        self.fixed_point_estimate = np.mean(self.data_window)
        self.slope_estimates = self.joint_samples[1, :] + 2 * self.joint_samples[2,
                                                              :] * self.fixed_point_estimate + 3 * self.joint_samples[3,
                                                                                                   :] * self.fixed_point_estimate ** 2

    def _MAP_third_order_polynom_slope_in_fixed_point(self):
        '''
        Calculate the `drift_slope` of a third order polynomial drift function with maximum a posteriori
        solution around a fixed point estimate given as the data mean of the current window.
        '''

        self.fixed_point_estimate = np.mean(self.data_window)
        self.drift_slope[0] = (self.MAP_theta[1, 0] + 2 * self.MAP_theta[2, 0] * self.fixed_point_estimate
                               + 3 * self.MAP_theta[3, 0] * self.fixed_point_estimate ** 2)


    @staticmethod
    def calc_D1(theta, data, drift_model):
        '''
        Returns the drift parameterized by a first or third order polynomial for input data.
        '''

        if drift_model == '3rd order polynomial':
            return theta[0] + theta[1] * data + theta[2] * data ** 2 + theta[3] * data ** 3
        elif drift_model == 'first order polynomial':
            return theta[0] + theta[1] * data
        elif drift_model == 'first order polynomial':
            return theta[0] + theta[1] * data
        elif drift_model == 'no offset first order':
            return - theta[0] ** 2 * data

    def D1(self):
        '''
        Calls the static ``calc_D1`` function and returns its value.
        '''

        return self.calc_D1(self.theta, self.data_window[:-1], self.drift_model)

    @staticmethod
    def calc_D2(theta, data, diffusion_model):
        '''
        Returns the diffusion parameterized by a constant or first order polynomial for input data.
        '''

        if diffusion_model == 'constant':
            return theta[-1] * np.ones(data.size)
        elif diffusion_model == 'first order polynomial':
            return theta[-2] + theta[-1] * data

    def D2(self):
        '''
        Calls the static ``calc_D2`` function and returns its value.
        '''
        return self.calc_D2(self.theta, self.data_window[:-1], self.diffusion_model)

    @staticmethod
    def log_prior(theta, drift_model, diffusion_model, prior_type, prior_range, scales):
        '''
        Returns the logarithmic prior probability for a given set of parameters based on the short time
        propagator depending on the drift and diffusion models and the given `prior_range` of the
        parameters `theta`. A flat prior for the parameters apart from restriction to positive constant
        diffusion and a non-informative prior for linear parts and Gaussian priors for higher orders are
        implemented. The standard deviation of the Gaussian priors is given by the `scales` variable.
        '''

        if drift_model == '3rd order polynomial' and diffusion_model == 'constant':
            if prior_type == 'Flat Prior':
                if (prior_range[0, 0] > theta[0] > prior_range[0, 1]) and (
                        prior_range[1, 0] > theta[1] > prior_range[1, 1]) and (
                        prior_range[2, 0] > theta[2] > prior_range[2, 1]) and (
                        prior_range[3, 0] > theta[3] > prior_range[3, 1]) and (
                        prior_range[4, 0] > theta[4] > prior_range[4, 1]):
                    return 0
                else:
                    return - np.inf

            elif prior_type == 'Non-informative linear and Gaussian Prior':
                if (prior_range[0, 0] > theta[0] > prior_range[0, 1]) and (
                        prior_range[1, 0] > theta[1] > prior_range[1, 1]) and (
                        prior_range[2, 0] > theta[2] > prior_range[2, 1]) and (
                        prior_range[3, 0] > theta[3] > prior_range[3, 1]) and (
                        prior_range[4, 0] > theta[4] > prior_range[4, 1]):
                    # print('Check_prior!')
                    return ((-3. / 2.) * np.log(1 + theta[1] ** 2) - np.log(theta[4])) + np.log(
                        cpy.norm.pdf(theta[2], loc=0, scale=scales[0])) + np.log(
                        cpy.norm.pdf(theta[3], loc=0, scale=scales[1]))
                else:
                    return - np.inf
        elif drift_model == 'first order polynomial' and diffusion_model == 'constant':
            if prior_type == 'Flat Prior':
                if (prior_range[0, 0] > theta[0] > prior_range[0, 1]) and (
                        prior_range[1, 0] > theta[1] > prior_range[1, 1]) and (
                        prior_range[2, 0] > theta[2] > prior_range[2, 1]):
                    return 0
                else:
                    return - np.inf
            elif prior_type == 'Non-informative linear and Gaussian Prior':
                if (prior_range[0, 0] > theta[0] > prior_range[0, 1]) and (
                        prior_range[1, 0] > theta[1] > prior_range[1, 1]) and (
                        prior_range[2, 0] > theta[2] > prior_range[2, 1]):
                    return ((-3. / 2.) * np.log(1 + theta[1] ** 2) - np.log(theta[2]))
                else:
                    return - np.inf

    @staticmethod
    def log_posterior(theta):
        '''
        Returns the logarithmic posterior probability of the data for the given model parametrization.
        '''

        lg_prior = __class__.log_prior(theta, shared_memory_dict['drift_model'], shared_memory_dict['diffusion_model'],
                                       shared_memory_dict['prior_type'], shared_memory_dict['prior_range'],
                                       shared_memory_dict['scales'])
        if not np.isfinite(lg_prior):
            return - np.inf
        return lg_prior + __class__.log_likelihood(theta, shared_memory_dict['data_window'], shared_memory_dict['dt'],
                                                   shared_memory_dict['drift_model'],
                                                   shared_memory_dict['diffusion_model'])

    @staticmethod
    def log_likelihood(theta, data, dt, drift_model, diffusion_model):
        '''
        Returns the logarithmic likelihood of the data for the given model parametrization.
        '''
        increments = data[1:] - data[:-1]
        return np.sum(-0.5 * np.log(2 * np.pi * __class__.calc_D2(theta, data[:-1], diffusion_model) ** 2 * dt) - (
                    increments - __class__.calc_D1(theta, data[:-1], drift_model) * dt) ** 2 / (
                                  2. * __class__.calc_D2(theta, data[:-1], diffusion_model) ** 2 * dt))

    @staticmethod
    def neg_log_posterior(theta):
        '''
        Returns the negative logarithmic posterior distribution of the data
        for the given model parametrization.
        '''

        return (-1) * __class__.log_posterior(theta)

    def declare_MAP_starting_guesses(self, nwalkers=None, nsteps=None):
        '''
        Declare the maximum a posterior (MAP) starting guesses for a MCMC sampling with ``nwalkers``
        and ``nsteps``.
        '''
        if nwalkers != None and nsteps != None:
            self.nwalkers = nwalkers
            self.nsteps = nsteps
        if self.nwalkers == None and nwalkers == None or self.nsteps == None and nsteps == None:
            print('ERROR: nwalkers and/or nsteps is not yet defined!')
        self.init_parallel_EnsembleSampler(self.data_window, self.dt, self.drift_model, self.diffusion_model,
                                           self.prior_type, self.prior_range, self.scales)
        res = optimize.minimize(self.neg_log_posterior, x0=np.ones(self.ndim), method='Nelder-Mead')
        MAP_results = res['x']
        self.starting_guesses = np.ones((self.nwalkers, self.ndim))
        for i in range(self.ndim):
            self.starting_guesses[:, i] = MAP_results[i] * (0.5 + np.random.rand(self.nwalkers))
            if self.drift_model == '3rd order polynomial' and self.diffusion_model == 'constant':
                self.starting_guesses[self.starting_guesses[:, i] > self.prior_range[i, 0], i] = self.prior_range[
                                                                                                     i, 0] - 1
                self.starting_guesses[self.starting_guesses[:, i] < self.prior_range[i, 1], i] = self.prior_range[
                                                                                                     i, 1] + 1

    def compute_posterior_samples(self, print_AC_tau, ignore_AC_error, thinning_by, print_progress,
                                  MCMC_parallelization_method=None, num_processes=None, num_chop_chains=None):
        '''
        Compute the `theta_array` with :math:`nwalkers \cdot nsteps` Markov Chain Monte Carlo (MCMC) samples.
        If `ignore_AC_error = False` the calculation will terminate with error if
        the autocorrelation of the sampled chains is too high compared to the chain length.
        Otherwise the highest autocorrelation length will be used to thin the sampled chains.
        In order to run tests in shorter time you can set "ignore_AC_error = True" and
        define a `thinning_by` n steps. If `print_AC_tau = True` the autocorrelation lengths of the
        sampled chains is printed.
        '''
        print('Calculate posterior samples')

        if MCMC_parallelization_method == None:
            sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_posterior)
            sampler.run_mcmc(self.starting_guesses, self.nsteps, progress=print_progress)
        elif MCMC_parallelization_method == 'multiprocessing':
            if num_processes != None and not isinstance(num_processes, str):
                with mp.Pool(processes=num_processes, initializer=self.init_parallel_EnsembleSampler, initargs=(
                self.data_window, self.dt, self.drift_model, self.diffusion_model, self.prior_type, self.prior_range,
                self.scales)) as pool:
                    sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_posterior, pool=pool)
                    sampler.run_mcmc(self.starting_guesses, self.nsteps, progress=print_progress)
            elif num_processes == 'all':
                with mp.Pool(processes=mp.cpu_count(), initializer=self.init_parallel_EnsembleSampler, initargs=(
                self.data_window, self.dt, self.drift_model, self.diffusion_model, self.prior_type, self.prior_range,
                self.scales)) as pool:
                    sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_posterior, pool=pool)
                    sampler.run_mcmc(self.starting_guesses, self.nsteps, progress=print_progress)
            else:
                with mp.Pool(processes=int(mp.cpu_count() / 2.), initializer=self.init_parallel_EnsembleSampler,
                             initargs=(
                             self.data_window, self.dt, self.drift_model, self.diffusion_model, self.prior_type,
                             self.prior_range, self.scales)) as pool:
                    sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_posterior, pool=pool)
                    sampler.run_mcmc(self.starting_guesses, self.nsteps, progress=print_progress)
        elif MCMC_parallelization_method == 'chop_chain':
            stepspernode = (self.nsteps - self.nburn) / num_chop_chains + self.nburn
            chop_chain_callables = {'parallel_log_prior': self.log_prior,
                                    'parallel_log_likelihood': self.log_likelihood,
                                    'parallel_log_posterior': self.log_posterior,
                                    'ndim': self.ndim}
            sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_posterior)
            cluster = ipp.Cluster(n=num_chop_chains)
            rc = cluster.start_and_connect_sync()
            dview = rc[:]
            dview.block = True
            dview.use_dill()
            chop_chain_samples = self.run_chop_chain(dview, self.starting_guesses, sampler, chop_chain_callables,
                                                     self.nsteps, self.nburn, self.ndim, self.nwalkers, print_progress,
                                                     ignore_AC_error, thinning_by, print_AC_tau)
        if MCMC_parallelization_method == None or MCMC_parallelization_method == 'multiprocessing':
            if ignore_AC_error == False:
                tau = sampler.get_autocorr_time()
            elif ignore_AC_error:
                tau = thinning_by
            print(sampler.get_chain(discard=self.nburn, flat=True).shape)
            flat_samples = sampler.get_chain(discard=self.nburn, thin=int(np.max(tau)), flat=True)
            if print_AC_tau:
                print('tau: ', tau)
        elif MCMC_parallelization_method == 'chop_chain':
            if ignore_AC_error == False:
                flat_samples = chop_chain_samples
            elif ignore_AC_error:
                tau = thinning_by
                print('Overhead: True!')
                thinned_chop_chain_samples = chop_chain_samples[::tau, :, :]
                flattened_samples = thinned_chop_chain_samples.reshape((-1, self.ndim))
                correct_num_of_thinned_tuples = int(self.nwalkers * int((self.nsteps - self.nburn) / thinning_by))
                print(correct_num_of_thinned_tuples)
                flat_samples = flattened_samples[:correct_num_of_thinned_tuples, :]
        self.theta_array = np.zeros((self.ndim, flat_samples[:, 0].size))
        for i in range(self.ndim):
            self.theta_array[i, :] = np.transpose(flat_samples[:, i])

    @staticmethod
    def run_chop_chain(dview, starting_guesses, sampler, chop_chain_callables, nsteps, nburn, ndim, nwalkers,
                       print_progress, ignore_AC_error, thinning_by, print_AC_tau):
        for (key, val) in chop_chain_callables.items():
            dview[key] = val
        # dview['ndim'] = int(ndim)
        dview['nburn'] = int(nburn)
        # print(dview["nburn"])
        print(len(dview))
        if ignore_AC_error:
            dview["stepspernode"] = int(mh.ceil((nsteps - nburn) / len(dview) + nburn))
        else:
            dview["stepspernode"] = int(nsteps / len(dview))
        dview['sampler'] = sampler
        dview['starting_guesses'] = starting_guesses
        dview['print_progress'] = print_progress
        dview["ignore_AC_error"] = ignore_AC_error
        dview["thinning_by"] = thinning_by
        dview["print_AC_tau"] = print_AC_tau
        dview["tau"] = 0
        # tau = 1
        # dview['tau'] = tau
        dview.execute("import numpy as np\n" +
                      "import scipy.stats as cpy\n" +
                      "sampler.run_mcmc(starting_guesses, stepspernode, rstate0=np.random.get_state())\n" +
                      "if ignore_AC_error == False:\n" +
                      "	tau = sampler.get_autocorr_time()\n" +
                      "	samples = sampler.get_chain(discard=nburn, thin=int(np.max(tau)), flat=True)\n" +
                      "elif ignore_AC_error:\n" +
                      "	tau = thinning_by\n" +
                      "	samples = sampler.get_chain()")
        if print_AC_tau:
            print('tau: ', dview["tau"])
        #		"samples = sampler.chain[::int(np.max(tau)), nburn:, :].reshape((-1, ndim))")
        return dview.gather("samples")

    @staticmethod
    def init():
        '''
        Defines the animation background for easy visualization of a rolling window scan of a given time
        series. It prepares plots for the time series and the time evolution of the drift slope
        estimate :math:`\hat{\zeta}` and noise level estimate :math:`\hat{\sigma}` and its probability
        density estimate. The method needs to be static. Some variables have to be declared global in
        to guarantee the functionality of the animation procedure.
        '''
        global fig, axs, sl_gr_animation, noise_gr_animation, animation_count, animation_time, animation_data  # vspan_window, noise_pdf_line, drift_slope_line, noise_line, CB_slope_I, CB_slope_II, CB_noise_I, CB_noise_II
        axs[0, 0].plot(animation_time, animation_data, color='C0')  # , color = 'b'
        axs[0, 0].set_xlim(animation_time[0], animation_time[-1])
        if crit_poi != None:
            axs[0, 0].axvline(crit_poi, ls=':', color='r')
        axs[0, 0].set_ylim(np.min(animation_data) - 1, np.max(animation_data) + 1)
        axs[0, 0].set_ylabel(r'variable $x$', fontsize=15)
        axs[0, 0].set_xlabel(r'time $t$', fontsize=15)

        axs[0, 1].set_xlim(noise_gr_animation[0], noise_gr_animation[-1])
        axs[0, 1].set_ylim(0, 100)
        if noi_le != None:
            axs[0, 1].axvline(noi_le, ls=':', color='g')
        axs[0, 1].set_ylabel(r'Noise Posterior $P(\hat{\theta_4})$', fontsize=15)
        axs[0, 1].set_xlabel(r'noise level $\hat{\theta_4}$', fontsize=15)

        axs[1, 0].set_xlim(animation_time[0], animation_time[-1])
        axs[1, 0].plot(animation_time, np.zeros(animation_time.size), ls=':', color='r')
        if crit_poi != None:
            axs[1, 0].axvline(crit_poi, ls=':', color='r')
        axs[1, 0].set_ylabel(r'max posterior slope $\zeta^{\rm max}$', fontsize=15)
        axs[1, 0].set_xlabel(r'time $t$', fontsize=15)

        axs[1, 1].set_xlim(animation_time[0], animation_time[-1])
        axs[1, 1].set_ylim(noise_gr_animation[0], noise_gr_animation[-1])
        if noi_le != None:
            axs[1, 1].plot(animation_time, noi_le * np.ones(animation_time.size), ls=':', color='g')
        axs[1, 1].set_ylabel(r'max posterior noise $\theta^{\rm max}_4$', fontsize=15)
        axs[1, 1].set_xlabel(r'time $t$', fontsize=15)

        axs[0, 0].tick_params(axis='both', which='major', labelsize=15)
        axs[0, 1].tick_params(axis='both', which='major', labelsize=15)
        axs[1, 0].tick_params(axis='both', which='major', labelsize=15)
        axs[1, 1].tick_params(axis='both', which='major', labelsize=15)

        fig.tight_layout()

        vspan_window = axs[0, 0].axvspan(0, 0, alpha=0.7, color='g')
        noise_pdf_line.set_data([], [])
        drift_slope_line.set_data([], [])
        noise_line.set_data([], [])
        CB_slope_I = axs[1, 0].fill_between([], [], [], alpha=0.7, color='orange')
        CB_slope_II = axs[1, 0].fill_between([], [], [], alpha=0.4, color='orange')
        CB_noise_I = axs[1, 1].fill_between([], [], [], alpha=0.7, color='orange')
        CB_noise_II = axs[1, 1].fill_between([], [], [], alpha=0.4, color='orange')
        return vspan_window, noise_pdf_line, drift_slope_line, noise_line, CB_slope_I, CB_slope_II, CB_noise_I, CB_noise_II

    # performs the animation

    def animation(self, i, slope_grid, noise_grid, nwalkers, nsteps, nburn, n_joint_samples, n_slope_samples,
                  n_noise_samples, cred_percentiles, print_AC_tau, ignore_AC_error, thinning_by, print_progress,
                  detrending_per_window, gauss_filter_mode,gauss_filter_sigma, gauss_filter_order,
                  gauss_filter_cval, gauss_filter_truncate, plot_detrending, MCMC_parallelization_method, num_processes,
                  num_chop_chains):
        '''
        Function that is called iteratively by the ``matplotlib.animation.FuncAnimation(...)`` tool
        to generate an animation of the rolling window scan of a time series. The time series, the rolling
        windows, the time evolution of the drift slope estimate :math:`\hat{\zeta}` and the noise level
        estimate :math:`\hat{\sigma}` and its posterior probality density is shown in the animation.
        '''
        global animation_count
        self.window_shift = i
        self.calc_driftslope_noise(slope_grid, noise_grid, nwalkers, nsteps, nburn, n_joint_samples, n_slope_samples,
                                   n_noise_samples, cred_percentiles, print_AC_tau, ignore_AC_error, thinning_by,
                                   print_progress, detrending_per_window,gauss_filter_mode,gauss_filter_sigma,
                                   gauss_filter_order, gauss_filter_cval, gauss_filter_truncate, plot_detrending,
                                   MCMC_parallelization_method, num_processes, num_chop_chains)
        self.slope_storage[:, animation_count] = self.drift_slope
        self.noise_level_storage[:, animation_count] = self.noise_level_estimate
        axs[0, 0].collections.clear()

        if i == 0:
            path = vspan_window.get_xy()
            path[:, 0] = [self.time_window[0], self.time_window[0], self.time_window[-1], self.time_window[-1]]
            vspan_window.set_xy(path)
        if i > 0:
            path = vspan_window.get_xy()
            path[:, 0] = [self.time_window[0], self.time_window[0], self.time_window[-1], self.time_window[-1],
                          self.time_window[0]]
            vspan_window.set_xy(path)
        noise_pdf_line.set_data(noise_gr_animation, self.noise_kernel_density_obj(noise_gr_animation))
        drift_slope_line.set_data(self.time[self.window_size - 1 + self.loop_range[:animation_count + 1]],
                                  self.slope_storage[0, :animation_count + 1])
        noise_line.set_data(self.time[self.window_size - 1 + self.loop_range[:animation_count + 1]],
                            self.noise_level_storage[0, :animation_count + 1])
        axs[1, 0].collections.clear()
        axs[1, 1].collections.clear()
        CB_slope_I = axs[1, 0].fill_between(self.time[self.window_size - 1 + self.loop_range[:animation_count + 1]],
                                            self.slope_storage[1, :animation_count + 1],
                                            self.slope_storage[2, :animation_count + 1], alpha=0.7, color='orange')
        CB_slope_II = axs[1, 0].fill_between(self.time[self.window_size - 1 + self.loop_range[:animation_count + 1]],
                                             self.slope_storage[3, :animation_count + 1],
                                             self.slope_storage[4, :animation_count + 1], alpha=0.4, color='orange')
        CB_noise_I = axs[1, 1].fill_between(self.time[self.window_size - 1 + self.loop_range[:animation_count + 1]],
                                            self.noise_level_storage[1, :animation_count + 1],
                                            self.noise_level_storage[2, :animation_count + 1], alpha=0.7,
                                            color='orange')
        CB_noise_II = axs[1, 1].fill_between(self.time[self.window_size - 1 + self.loop_range[:animation_count + 1]],
                                             self.noise_level_storage[3, :animation_count + 1],
                                             self.noise_level_storage[4, :animation_count + 1], alpha=0.4,
                                             color='orange')
        animation_count += 1
        return vspan_window, noise_pdf_line, drift_slope_line, noise_line, CB_slope_I, CB_slope_II, CB_noise_I, CB_noise_II

    def calc_driftslope_noise(self, slope_grid, noise_grid, nwalkers=50, nsteps=10000, nburn=200,
                              n_joint_samples=50000, n_slope_samples=50000, n_noise_samples=50000,
                              cred_percentiles=[16, 1], print_AC_tau=False,
                              ignore_AC_error=False, thinning_by=60, print_progress=False,
                              detrending_per_window = None, gauss_filter_mode = 'reflect',
                              gauss_filter_sigma = 6, gauss_filter_order = 0, gauss_filter_cval = 0.0,
                              gauss_filter_truncate = 4.0, plot_detrending = False, MCMC_parallelization_method=None,
                              num_processes=None, num_chop_chains=None):
        '''
        Calculates the drift slope estimate :math:`\hat{\zeta}` and the noise level :math:`\hat{\sigma}`
        from the MCMC sampled parameters of a Langevin model for a given ``window_size`` and given ``window_shift``.
        In the course of the computations the parameters ``joint_samples``, a ``noise_kernel_density_obj``,
        a ``slope_kernel_density_obj``, the ``drift_slope`` with credibility intverals, the ``noise_level`` with
        credibility intervals and in case of a third order polynomial drift the estimated `fixed_point_estimate`
        of the window will be stored.

        :param slope_grid: Array on which the drift slope kernel density estimate is evaluated.
        :type slope_grid: One-dimensional numpy array of floats
        :param noise_grid: Array on which the noise level kernel density estimate is evaluated.
        :type noise_grid: One-dimensional numpy array of floats
        :param nwalkers: Number of walkers that are initialized for the MCMC sampling via the package ``emcee``.
                        Default is 50.
        :type nwalkers: int
        :param nsteps: Length of the sampled MCMC chains. Default is 10000.
        :type nsteps: int
        :param nburn: Number of data points at the beginning of the Markov chains which are discarded
                        in terms of a burn in period. Default is 200.
        :type nburn: int
        :param n_joint_samples: Number of joint samples that are drawn from the estimated joint posterior
                        probability in order to calculate the drift slope estimate :math:`\zeta` and
                        corresponding credibility bands. Default is 50000.
        :type n_joint_samples: int
        :param n_slope_samples: Number of drift slope samples that are drawn from the estimated drift slope
                        posterior computed based on the sampling results of the joint posterior probability
                        of the Langevin parameters :math:`\theta_i`. Default is 50000.
        :type n_slope_samples: int
        :param n_noise_samples: Number of constant noise level samples that are drawn from the estimated
                        marignal probility distribution of the diffusion model is chosen constant.
                        Default is 50000.
        :type n_noise_samples: int
        :param cred_percentiles: Two entries to define the percentiles of the calculated credibility bands
                        of the estimated parameters. Default is ``numpy.array([16,1])``.
        :type cred_percentiles: One-dimensional numpy array of integers
        :param print_AC_tau: If ``True`` the estimated autocorrelation lengths of the Markov chains is shown.
                        The maximum length is used for thinning of the chains. Default is ``False``.
        :type prind_AC_tau: bool
        :param ignore_AC_error: If ``True`` the autocorrelation lengths of the Markov chains is not estimated
                        and thus, it is not checked whether the chains are too short to give an reliable
                        autocorrelation estimate. This avoids error interruption of the procedure, but
                        can lead to unreliable results if the chains are too short. The option should
                        be chosen in order to save computation time, e.g. when debugging your code
                        with short Markov chains. Default is ``False``.
        :type ignore_AC_error: bool
        :param thinning_by: If ``ignore_AC_error = True`` the chains are thinned by ``thinning_by``. Every
                        ``thinning_by``-th data point is stored. Default is 60.
        :type thinning_by: int
        :param print_progress: If ``True`` the progress of the MCMC sampling is shown. Default is ``False``.
        :type print_progress: bool
        :param detrending_per_window: Default is ``None``. If ``'linear'``, the ``self.data_window`` is detrended linearly. If
                        ``Gaussian kernel smoothed``, a Gaussian smoothed curve is subtracted from ``self.data_window``.
                        The parameters of the kernel smoothing can be set by ``gauss_filter_mode``, ``gauss_filter_sigma``,
                        ``gauss_filter_order``, ``gauss_filter_cval`` and ``gauss_filter_truncate``. Drift slope and
                        noise level are estimated after detrending in these cases.
        :type detrending_per_window: str
        :param gauss_filter_mode: According to the ``scipy.ndimage.filters.gaussian_filter`` option.
        :type gauss_filter_mode: str
        :param gauss_filter_sigma: According to the ``scipy.ndimage.filters.gaussian_filter`` option.
        :type gauss_filter_sigma: float
        :param gauss_filter_order: According to the ``scipy.ndimage.filters.gaussian_filter`` option.
        :type gauss_filter_order: int
        :param gauss_filter_cval: According to the ``scipy.ndimage.filters.gaussian_filter`` option.
        :type gauss_filter_cval: float
        :param gauss_filter_truncate: According to the ``scipy.ndimage.filters.gaussian_filter`` option.
        :type gauss_filter_truncate: float
        :param plot_detrending: Default is ``False``. If ``True``, the ``self.data`` as well as the
                            ``self.slow_trend`` and the detrended version are shown.
        :type plot_detrending: bool
        '''
        cred_percentiles = np.array(cred_percentiles)
        self.data_window = np.roll(self.data, shift=- self.window_shift)[:self.window_size]
        self.time_window = np.roll(self.time, shift=- self.window_shift)[:self.window_size]
        if detrending_per_window != None:
            self.data_window, self.slow_trend = self.detrend(self.time_window, self.data_window,
                                                                  detrending_per_window,gauss_filter_mode,
                                                                  gauss_filter_sigma,gauss_filter_order,
                                                                  gauss_filter_cval, gauss_filter_truncate,
                                                                  plot_detrending)
        self.increments = self.data_window[1:] - self.data_window[:-1]
        self.declare_MAP_starting_guesses()
        self.compute_posterior_samples(print_AC_tau=print_AC_tau, ignore_AC_error=ignore_AC_error,
                                       thinning_by=thinning_by, print_progress=print_progress,
                                       MCMC_parallelization_method=None, num_processes=None, num_chop_chains=None)
        if self.drift_model == '3rd order polynomial':
            self.joint_kernel_density_obj = cpy.gaussian_kde(self.theta_array, bw_method='silverman')
            self.joint_samples = self.joint_kernel_density_obj.resample(size=n_joint_samples)
            self.third_order_polynom_slope_in_fixed_point()
            self.noise_kernel_density_obj = cpy.gaussian_kde(self.theta_array[4, :], bw_method='silverman')
        elif self.drift_model == 'first order polynomial':
            self.slope_estimates = self.theta_array[:, 1]
            self.noise_kernel_density_obj = cpy.gaussian_kde(self.theta_array[2, :], bw_method='silverman')
        self.slope_kernel_density_obj = cpy.gaussian_kde(self.slope_estimates, bw_method='silverman')
        slope_samples = self.slope_kernel_density_obj.resample(size=n_slope_samples)
        max_slope = slope_grid[
            self.slope_kernel_density_obj(slope_grid) == np.max(self.slope_kernel_density_obj(slope_grid))]
        if max_slope.size == 1:
            self.drift_slope[0] = max_slope
        else:
            self.drift_slope[0] = max_slope[0]
        slope_credibility_percentiles = np.percentile(slope_samples, [cred_percentiles[0], 100 - cred_percentiles[0]])
        self.drift_slope[1] = slope_credibility_percentiles[0]
        self.drift_slope[2] = slope_credibility_percentiles[1]
        slope_credibility_percentiles = np.percentile(slope_samples, [cred_percentiles[1], 100 - cred_percentiles[1]])
        self.drift_slope[3] = slope_credibility_percentiles[0]
        self.drift_slope[4] = slope_credibility_percentiles[1]
        noise_samples = self.noise_kernel_density_obj.resample(size=n_noise_samples)
        noise_level = noise_grid[
            self.noise_kernel_density_obj(noise_grid) == np.max(self.noise_kernel_density_obj(noise_grid))]
        if noise_level.size == 1:
            self.noise_level_estimate[0] = noise_level
        else:
            self.noise_level_estimate[0] = noise_level[0]
        noise_credibility_percentiles = np.percentile(noise_samples, [cred_percentiles[0], 100 - cred_percentiles[0]])
        self.noise_level_estimate[1] = noise_credibility_percentiles[0]
        self.noise_level_estimate[2] = noise_credibility_percentiles[1]
        noise_credibility_percentiles = np.percentile(noise_samples, [cred_percentiles[1], 100 - cred_percentiles[1]])
        self.noise_level_estimate[3] = noise_credibility_percentiles[0]
        self.noise_level_estimate[4] = noise_credibility_percentiles[1]

    @staticmethod
    def init_parallel_EnsembleSampler(data_window, dt, drift_model, diffusion_model, prior_type, prior_range, scales):
        global shared_memory_dict
        shared_memory_dict = {}
        shared_memory_dict['data_window'] = data_window
        shared_memory_dict['dt'] = dt
        shared_memory_dict['drift_model'] = drift_model
        shared_memory_dict['diffusion_model'] = diffusion_model
        shared_memory_dict['prior_type'] = prior_type
        shared_memory_dict['prior_range'] = prior_range
        shared_memory_dict['scales'] = scales

    def perform_resilience_scan(self, window_size, window_shift, slope_grid, noise_grid,
                                             nwalkers=50, nsteps=10000, nburn=200, n_joint_samples=50000,
                                             n_slope_samples=50000, n_noise_samples=50000,
                                             cred_percentiles=[16, 1], print_AC_tau=False,
                                             ignore_AC_error=False, thinning_by=60, print_progress=False,
                                             slope_save_name='default_save_slopes',
                                             noise_level_save_name='default_save_noise', save=True,
                                             create_animation=False, ani_save_name='default_animation_name',
                                             animation_title='', mark_critical_point=None,
                                             mark_noise_level=None, detrending_per_window = None,
                                             gauss_filter_mode = 'reflect',gauss_filter_sigma = 6,
                                             gauss_filter_order = 0, gauss_filter_cval = 0.0,
                                             gauss_filter_truncate = 4.0, plot_detrending = False,
                                             MCMC_parallelization_method = None, num_processes = None,
                                             num_chop_chains = None):
        '''
        Performs an automated window scan with defined ``window_shift`` over the whole time series. In each
        window the drift slope and noise level estimates with corresponding credibility bands are computed
        and saved in the ``slope_storage`` and the ``noise_level_storage``. It can also be used to create an
        animation of the sliding window approach plotting the time series, the moving window, and the
        time evolution of the drift slope estimates :math:`\hat{\zeta}`, the noise level :math:`\hat{\sigma}`
        and the noise kernel density estimate. The start indices of the shifted windows is also saved
        in order to facilitate customized plots.

        :param window_size: Time window size.
        :type window_size: int
        :param window_shift: The rolling time window is shifted about ``window_shift`` data points.
        :type window_shift: int
        :param slope_grid: Array on which the drift slope kernel density estimate is evaluated.
        :type slope_grid: One-dimensional numpy array of floats
        :param noise_grid: Array on which the noise level kernel density estimate is evaluated.
        :type noise_grid: One-dimensional numpy array of floats
        :param nwalkers: Number of walkers that are initialized for the MCMC sampling via the package ``emcee``.
                        Default is 50.
        :type nwalkers: int
        :param nsteps: Length of the sampled MCMC chains. Default is 10000.
        :type nsteps: int
        :param nburn: Number of data points at the beginning of the Markov chains which are discarded in terms of
                        a burn in period. Default is 200.
        :type nburn: int
        :param n_joint_samples: Number of joint samples that are drawn from the estimated joint posterior
                        probability in order to calculate the drift slope estimate :math:`\zeta` and
                        corresponding credibility bands. Default is 50000.
        :type n_joint_samples: int
        :param n_slope_samples: Number of drift slope samples that are drawn from the estimated drift slope
                        posterior computed based on the sampling results of the joint posterior probability
                        of the Langevin parameters :math:`\theta_i`. Default is 50000.
        :type n_slope_samples: int
        :param n_noise_samples: Number of constant noise level samples that are drawn from the estimated
                        marignal probility distribution of the diffusion model is chosen constant.
                        Default is 50000.
        :type n_noise_samples: int
        :param cred_percentiles: Two entries to define the percentiles of the calculated credibility bands
                        of the estimated parameters. Default is ``numpy.array([16,1])``.
        :type cred_percentiles: One-dimensional numpy array of integers
        :param print_AC_tau: If ``True`` the estimated autocorrelation lengths of the Markov chains is shown.
                        The maximum length is used for thinning of the chains. Default is ``False``.
        :type prind_AC_tau: bool
        :param ignore_AC_error: If ``True`` the autocorrelation lengths of the Markov chains is not estimated
                        and thus, it is not checked whether the chains are too short to give an reliable
                        autocorrelation estimate. This avoids error interruption of the procedure, but
                        can lead to unreliable results if the chains are too short. The option should
                        be chosen in order to save computation time, e.g. when debugging your code
                        with short Markov chains. Default is ``False``.
        :type ignore_AC_error: bool
        :param thinning_by: If ``ignore_AC_error == True`` the chains are thinned by ``thinning_by``. Every
                        ``thinning_by``-th data point is stored. Default is 60.
        :type thinning_by: int
        :param print_progress: If ``True`` the progress of the MCMC sampling is shown. Default is ``False``.
        :type print_progress: bool
        :param slope_save_name: Name of the file in which the ``slope_storage`` array will be saved. Default is
                        ``default_save_slopes``.
        :type slope_save_name: str
        :param  noise_level_save_name: Name of the file in which the ``noise_level_storage`` array will
                        be saved. Default is ``default_save_noise``.
        :type noise_level_save_name: str
        :param save: If ``True`` the ``slope_storage`` and the ``noise_level_storage`` arrays are saved in the
                        end of the scan in an `.npy` file. Default is ``True``.
        :type save: bool
        :param create_animation: If ``True`` an automated animation of the time evolution of the drift slope
                        estimate :math:\hat{\zeta}`, the noise level :math:`\hat{\sigma}` and the
                        noise kernel density is shown together with the time series and rolling
                        windows. Default is ``False``.

                        .. warning::

                            It makes use of global variables to incorporate the movie animation tool in the class.
                            The ``create_animation`` parameter is only intended to easily reconstruct
                            the plot results of the related publication. In other circumstances it should not
                            be used in order to avoid conflicts with the global variable names.

        :type create_animation: bool
        :param ani_save_name: If ``create_animation = True`` the animation is saved in a `.mp4` file with this name.
                        Default is ``default_animation_name``.
        :type ani_save_name: str
        :param animation_title: A title can be given to the animation.
        :type animation_title: str
        :param mark_critical_point: A red dotted vertical line is shown at time ``mark_critical_point``.
                        Default is ``None``.
        :type mark_critical_point: float
        :param mark_noise_level: A green dotted line is shown at the noise level ``mark_noise_level`` in the
                        noise kernel density plot and the time evolution plot of the noise level.
                        Default is ``None``.
        :type mark_noise_level: float
        :param detrending_per_window: Default is ``None``. If ``'linear'`` the ``data_window`` is detrended linearly. If
                        ``Gaussian kernel smoothed`` a Gaussian smoothed curve is subtracted from the ``data_window``.
                        The parameters of the kernel smoothing can be set by ``gauss_filter_mode``, ``gauss_filter_sigma``,
                        ``gauss_filter_order``, ``gauss_filter_cval`` and ``gauss_filter_truncate``.
        :type detrending_of_whole_dataset: str
        :param gauss_filter_mode: According to the ``scipy.ndimage.filters.gaussian_filter`` option.
        :type gauss_filter_mode: str
        :param gauss_filter_sigma: According to the ``scipy.ndimage.filters.gaussian_filter`` option.
        :type gauss_filter_sigma: float
        :param gauss_filter_order: According to the ``scipy.ndimage.filters.gaussian_filter`` option.
        :type gauss_filter_order: int
        :param gauss_filter_cval: According to the ``scipy.ndimage.filters.gaussian_filter`` option.
        :type gauss_filter_cval: float
        :param gauss_filter_truncate: According to the ``scipy.ndimage.filters.gaussian_filter`` option.
        :type gauss_filter_truncate: float
        :param plot_detrending: Default is ``False``. If ``True``, the ``self.data_window`` as well as the
                        ``self.slow_trend`` and the detrended version are shown.
        :type plot_detrending: bool
        :param MCMC_parallelization_method: Default is `None`. If `None` the basic serial MCMC computation is performed. If
                        ``MCMC_parallelization_method = 'multiprocessing'``, a multiprocessing pool with `num_processes`
                        is used to accelerate MCMC sampling. If ``MCMC_parallelization_method = 'chop_chain'`` is used, the
                        total length of the desired Markov chain is divided into ``'chop_chain'`` parts each of which is
                        sampled in parallel and joined together in the end.
        :type MCMC_parallelization_method: str
        :param num_processes: Default is ``None``. If ``'half``, almost half of the CPU kernels are used. If ``'all'``,
                        all CPU kernels are used. If integer number, the defined number of CPU kernels is used for
                        multiprocessing.
        :type num_processes: str or int
        :param num_chop_chains: Number by which the total length of the Markov chain is divided. Each slice is sampled
                        in parallel and joined together in the end of the calculations.
        :type num_chop_chains: int

        '''

        cred_percentiles = np.array(cred_percentiles)
        self.window_size = window_size
        self.window_shift = window_shift
        self.data_window = np.zeros(window_size)
        self.time_window = np.zeros(window_size)
        self.increments = np.zeros(window_size - 1)
        self.nwalkers = nwalkers
        self.nsteps = nsteps
        self.nburn = nburn
        self.loop_range = np.arange(0, self.data_size - self.window_size, self.window_shift)
        self.slope_storage = np.zeros((5, self.loop_range.size))
        self.noise_level_storage = np.zeros((5, self.loop_range.size))
        if create_animation == False:
            for i in range(self.loop_range.size):
                self.window_shift = self.loop_range[i]
                self.calc_driftslope_noise(slope_grid, noise_grid, nwalkers, nsteps, nburn, n_joint_samples,
                                           n_slope_samples, n_noise_samples, cred_percentiles, print_AC_tau,
                                           ignore_AC_error, thinning_by, print_progress, detrending_per_window,
                                           gauss_filter_mode,gauss_filter_sigma, gauss_filter_order,
                                           gauss_filter_cval, gauss_filter_truncate, plot_detrending,
                                           MCMC_parallelization_method, num_processes, num_chop_chains)
                self.slope_storage[:, i] = self.drift_slope
                self.noise_level_storage[:, i] = self.noise_level_estimate
        elif create_animation == True:
            global fig, axs, sl_gr_animation, noise_gr_animation, animation_count, animation_time, animation_data, noi_le, crit_poi, vspan_window, noise_pdf_line, drift_slope_line, noise_line, CB_slope_I, CB_slope_II, CB_noise_I, CB_noise_II
            animation_time = self.time
            animation_data = self.data
            noi_le = mark_noise_level
            crit_poi = mark_critical_point
            sl_gr_animation = slope_grid
            noise_gr_animation = noise_grid
            animation_count = 0
            fig, axs = plt.subplots(2, 2, figsize=(19, 8))
            plt.suptitle(animation_title, fontsize=15)
            vspan_window = axs[0, 0].axvspan(0, 0, alpha=0.7, color='g')
            noise_pdf_line, = axs[0, 1].plot([], [], linewidth=2, color='r')
            drift_slope_line, = axs[1, 0].plot([], [], linewidth=2, color='b')
            noise_line, = axs[1, 1].plot([], [], linewidth=2, color='b')
            CB_slope_I = axs[1, 0].fill_between([], [], [], alpha=0.7, color='orange')
            CB_slope_II = axs[1, 0].fill_between([], [], [], alpha=0.4, color='orange')
            CB_noise_I = axs[1, 1].fill_between([], [], [], alpha=0.7, color='orange')
            CB_noise_II = axs[1, 1].fill_between([], [], [], alpha=0.4, color='orange')
            Writer = matplotlib.animation.writers['ffmpeg']
            writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=1800)
            ani = FuncAnimation(fig, self.animation, frames=self.loop_range, init_func=self.init, blit=True,
                                repeat=False, fargs=[slope_grid, noise_grid, nwalkers, nsteps, nburn, n_joint_samples,
                                                     n_slope_samples, n_noise_samples, cred_percentiles, print_AC_tau,
                                                     ignore_AC_error, thinning_by, print_progress,detrending_per_window,
                                                     gauss_filter_mode,gauss_filter_sigma, gauss_filter_order,
                                                     gauss_filter_cval, gauss_filter_truncate, plot_detrending,
                                                     MCMC_parallelization_method, num_processes, num_chop_chains])
            ani.save(ani_save_name + '.mp4', writer=writer)  # , writer = writer
        if save:
            np.save(slope_save_name + '.npy', self.slope_storage)
            np.save(noise_level_save_name + '.npy', self.noise_level_storage)

    def perform_MAP_resilience_scan(self, window_size, window_shift,
                                    cred_percentiles=np.array([16, 1]), error_propagation='summary statistics',
                                    summary_window_size = 10, sigma_multiples = np.array([1,3]),
                                    print_progress=True, print_details=False,
                                    slope_save_name='default_save_slopes',
                                    noise_level_save_name='default_save_noise', save=True,
                                    create_plot=False, ani_save_name='default_animation_name',
                                    animation_title='', mark_critical_point=None,
                                    mark_noise_level=None, print_hint = True, fastMAPflag = False):
        """
        Performs an automated MAP window scan with defined `window_shift` over the whole time series. In each
        window the drift slope and noise level estimates with corresponding credibility bands are computed
        and saved in the `slope_storage` and the `noise_level_storage`. It can also be used to create an
        plot of the sliding window approach plotting the time series, the moving window, and the
        time evolution of the drift slope estimates :math:`\hat{\zeta}` and the noise level :math:`\hat{\sigma}`.
        The start indices of the shifted windows are also saved in order to facilitate customized plots.

        :param window_size: Time window size.
        :type window_size: int
        :param window_shift: The rolling time window is shifted about `window_shift` data points.
        :type window_shift: int
        :param cred_percentiles: One or two entries to define the percentiles of the calculated credibility bands
                        of the estimated parameters. It is stored in the attribute `credibility_bands`.
                        Default is `numpy.array([16,1])`.

        :type cred_percentiles: One-dimensional numpy array of integers.
        :param error_propagation: Defines the method that is used to compute the confidence bands. Default is
                        ``'summary statistics``. In that case drift slope samples of size ``summary_window_size`` are
                        used to compute the drift slope mean and its standard error. The parameter ``sigma_multiples``
                        defines the width of the summary statistics' symmetric error bands.
                        If ``'error bound'`` or ``'uncorrelated Gaussian'`` is chosen, the marginal uncertainties
                        corresponding to ``cred_percentiles`` are computed employing Wilks' theorem. With ``'error bound'``
                        the highest uncertainties per parameter and ``cred_percentiles`` level is interpreted as a symmetric
                        worst case bound of the error. If  ``'uncorrelated Gaussian'`` is chosen, the uncertainties are
                        treated corresponding to the slight asymmetric results of the Wilks' theorem confidence intervals.
                        Both options give trustworthy results in the case of a first order polynomial drift. In case of the
                        third order polynomial drift they are too optimistic, i.e. narrow, since the error bounds
                        (``'error bound'``) or errors (``'uncorrelated Gaussian'``) are propagated without regard of
                        correlations in the model parameters. This issue might be solvable by introducing an estimate of the
                        covariance matrix in future, but is not planned for now. Furthermore, the ``'uncorrelated Gaussian'``
                        option keeps the slightly asymmetric errors proposed by Wilks' theorem for the marginal parameter
                        distributions. This small formal incorrectness is to maintain information about asymmetry in the
                        estimates in a first Wilks' theorem guess. The ``cred_percentiles`` are fixed to
                        ``cred_percentiles = numpy.array([5,1])`` to transform the half confidence bands under the assumption
                        of Gaussian distributions by the factors 1.96 and 2.5525, respectively. The propagated result is
                        transformed back.

                        .. hint::
                            The options ``'error bound'`` and ``'uncorrelated Gaussian'`` are only trustworthy for a linear
                            drift term. They might also be correct for a third order polynomial drift in the case ofvery
                            high amounts of data per window. Otherwise, the default ``'summary statistics'`` should be used,
                            unless the window shift is very high. In that case it might introduce a too strong delay due
                            to the averaging procedure.

        :type error_propagation: str

        :param print_progress: If `True` the progress of the MAP estimation per window is shown.
                        Default is `False`.

        :type print_progress: Boolean
        :param print_details: If `True` a more detailed print output in the case the binning procedure is
                        provided. Default is `False`.

        :type print_details: Boolean
        :param slope_save_name: Name of the file in which the `slope_storage` array will be saved. Default is
                        `default_save_slopes`.

        :type slope_save_name: string
        :param  noise_level_save_name: Name of the file in which the `noise_level_storage` array will
                        be saved. Default is `default_save_noise`.

        :type noise_level_save_name: string
        :param save: If `True` the `slope_storage` and the `noise_level_storage` arrays are saved in the
                        end of the scan in an .npy file. Default is `True`.

        :type save: Boolean
        :param create_plot: If ``True``, the MAP resilience scan results are plotted. Default is ``False``.
                        NOTE: Not implemented yet.
        :type create_plot: Boolean
        :param summary_window_size: If ``error_propagation = 'summary statistics'`` is chosen, the parameter defines
                        the number of drift slope estimates to use in a window summary statistic. The windows are shifted
                        by one.
        :type summary_window_size: int
        :param sigma_multiples: The array hast two entries. If ``error_propagation = 'summary statistics'`` is chosen,
                        the entries define the drift slope standard error multiples which are used to calculate the
                        uncertainty bands.
        :type sigma_multiples: One dimensional numpy array of float .
        """

        if error_propagation == 'uncorrelated Gaussian' and self.drift_model == '3rd order polynomial':
            credibility_bands = np.array([5,1])
            if print_hint:
                print('HINT: Fixed cred_percentiles = numpy.array[5,1]) for the uncorrelated Gaussian error propagation '
                      'of the drift slope is used for drift_model == `3rd_order_polynomial`.')
        elif error_propagation == 'uncorrelated Gaussian' and self.drift_model == 'first order polynomial':
            credibility_bands = cred_percentiles
        elif error_propagation == 'error bound':
            credibility_bands = cred_percentiles
        elif not error_propagation == 'summary statistics':
            print('ERROR: No suitable error_propagation option defined.')
        self.window_size = window_size
        self.window_shift = window_shift
        self.data_window = np.zeros(window_size)
        self.time_window = np.zeros(window_size)
        self.loop_range = np.arange(0, self.data_size - self.window_size, self.window_shift)
        self.slope_storage = np.zeros((5, self.loop_range.size))
        self.noise_level_storage = np.zeros((5, self.loop_range.size))
        if create_plot == False:
            for i in range(self.loop_range.size):
                if print_progress:
                    print('Calculate MAP resilience for window ' + str(i + 1) + ' of ' + str(self.loop_range.size) + '.')
                self.window_shift = self.loop_range[i]
                self._calc_MAP_resilience(cred_percentiles, error_propagation, summary_window_size,
                                          sigma_multiples, print_details)
                self.slope_storage[:, i] = self.drift_slope
                self.noise_level_storage[:, i] = self.noise_level_estimate
        elif create_plot:
            print('The MAP plot feature is not implemented yet.')
        if error_propagation == 'summary statistics' and not fastMAPflag:
            self.slope_storage = summary_statistics_helper(self.slope_storage, summary_window_size, sigma_multiples)
        if save:
            np.save(slope_save_name, self.slope_storage)
            np.save(noise_level_save_name, self.noise_level_storage)


    def _prepare_data(self, printbool):
        """
        Helper function to calculate the increments in case of a ``LangevinEstimation`` object.
        The ``printbool`` parameter is just a placeholder for the overloaded function of the
        ``BinningLangevinEstimation`` class.
        """
        self.increments = self.data_window[1:] - self.data_window[:-1]


    def _calc_MAP_resilience(self, cred_percentiles, error_propagation, summary_window_size,
                            sigma_multiples, print_details):
        """
        Helper function that computes the MAP estimates of drift slope :math:`\hat{\zeta}` and noise level
        :math:`\hat{\sigma}` with corresponding confidence bands created with Wilks' theorem and
        Gaussian propagation of unertainty for a given ``window_size`` and ``window_shift`` stored
        in the corresponding attributes.
        """
        self.data_window = np.roll(self.data, shift=- self.window_shift)[:self.window_size]
        self.time_window = np.roll(self.time, shift=- self.window_shift)[:self.window_size]
        self._prepare_data(printbool=print_details)
        self._compute_MAP_estimates(printbool=print_details)
        if print_details:
            print('__________________________')
            print('Calculate MAP drift slope!')
            print('__________________________')
        if self.drift_model == '3rd order polynomial':
            self._MAP_third_order_polynom_slope_in_fixed_point()
        elif self.drift_model == 'linear model':
            self.drift_slope[0] = self.MAP_theta[1, 0]
        if print_details:
            print('_____')
            print('Done!')
            print('_____')
        self._determine_confidence_bands(sigRatio=cred_percentiles, printbool=print_details)
        if error_propagation == 'error bound':
            self._compute_slope_error_margin(printbool=print_details)
        elif error_propagation == 'uncorrelated Gaussian':
            self._compute_slope_errors(printbool=print_details)


    def _compute_MAP_estimates(self, printbool=False, print_time_scale_info = None):
        """
        Helper function that determines the ``MAP_theta``.

        :param printbool: If ``True`` a detailed output is printed.
        :type printbool: Boolean
        """
        if printbool:
            print('______________________')
            print('Perform MAP estimation!')
            print('______________________')
        par0 = np.ones(self.ndim)
        self.MAP_theta = np.zeros((self.ndim, 5))
        if self.antiCPyObject == 'LangevinEstimation':
            self.init_parallel_EnsembleSampler(self.data_window, self.dt, self.drift_model, self.diffusion_model,
                                                self.prior_type, self.prior_range, self.scales)
        elif self.antiCPyObject == 'NonMarkovEstimation':
            self.init_parallel_EnsembleSampler(self.data_window, self.X_drift_model, self.Y_drift_model, self.Y_diffusion_model,
                                       self.X_coupling_term, self.Y_model, self.dt, self.prior_type, self.prior_range,
                                       self.scales, self.activate_time_scale_separation_prior,
                                       self.time_scale_separation_factor, self.slow_process, print_time_scale_info)
            par0 = self.max_likelihood_starting_guesses
        res = optimize.minimize(self.neg_log_posterior, x0=par0,
                                options={'maxiter': 10 ** 5, 'disp': False},
                                method='Nelder-Mead')
        self.MAP_theta[:, 0] = res['x']
        if printbool:
            print('MAP theta: ', self.MAP_theta)
            print('______________________')
            print('Done!')
            print('______________________')

    def _confidence_helper(self, map_bpci_bisection, theta_index):
        """
        Helper function to find the roots at the boundaries of the 1D confidence intervals
        of the ``theta_index``-th ``MAP_theta`` estimate at a confidence level given by the
        attribute``_sigmaRatio``.

        :param map_bpci_bisection: Deviation of the ``theta_index``-th parameter that is varied by
                        the bisection algorithm in order to find the confidence bound for a given
                        confidence level.

        :type map_bpci_bisection: float
        :param theta_index: Index of the ``MAP_theta`` estimate of which the confidence band is computed.
        :type theta_index: int
        """

        dev = np.zeros(self.ndim)
        dev[theta_index] = map_bpci_bisection
        return (self._optimum_solution
                - self.neg_log_posterior(self.MAP_theta[:, 0] + dev)
                - (np.log(self._sigmaRatio)))


    def _calc_confidence_band(self, printbool):
        """
        Helper function that determines the ``MAP_theta`` confidence band for a given
        confidence level stored in the attribute ``_sigmaRatio`` by bisection of the
        ``_confidence_helper(...)`` method.

        :param printbool: Determines whether a detailed output is printed or not.
        :type printbool: Boolean
        """
        if printbool:
            print('_______________________________________________________')
            print('Compute Bayesian confidence bands with Wilks´ theorem!')
            print('_______________________________________________________')
        self._optimum_solution = self.neg_log_posterior(self.MAP_theta[:, 0])
        MAP_CI = np.zeros((self.ndim, 2))
        for i in range(self.ndim):
            if printbool:
                print('Parameter ' + str(i + 1) + ' : ' + str(self.MAP_theta[i, 0]) + ' BPCI : ( ')
            # find lower bound for bisection root finding algorithm
            map_bpci_bisection = -1
            while self._confidence_helper(map_bpci_bisection, i) >= 0:
                map_bpci_bisection -= 1.
            # b i s e c t i o n
            MAP_CI[i, 0] = optimize.bisect(f=self._confidence_helper, a=map_bpci_bisection, b=0.,
                                           args=(i), xtol=1e-3)
            # find upper bound for bisection root finding algorithm
            map_bpci_bisection = 1
            while self._confidence_helper(map_bpci_bisection, i) > 0:
                map_bpci_bisection += 1.
            # b i s e c t i o n
            MAP_CI[i, 1] = optimize.bisect(f=self._confidence_helper, a=0., b=map_bpci_bisection,
                                           args=(i), xtol=1e-3)
            if printbool:
                print(str(MAP_CI[i, 0]) + ', ' + str(MAP_CI[i, 1]) + ')')
        if printbool:
            print('_______________________________________________________')
            print('Done!')
            print('_______________________________________________________')
        return MAP_CI


    def _determine_confidence_bands(self, sigRatio, printbool):
        """
        Helper function that is used to calculate the confidence bands for up to two different
        confidence levels each of which is computed via the ``_calc_confidence_band(...)`` method.

        :param sigRatio: Contains the first and second confidence level each of which is used to determine
                        the confidence bands of the ``MAP_theta`` estimates.

        :type sigRatio: One-dimensional numpy array of float.

        :param printbool: Determines whether a detailed output is printed or not.
        :type printbool: Boolean
        """
        if sigRatio.size == 2:
            self.MAP_CI = np.zeros((self.ndim, 4))
            self._sigmaRatio = np.exp(-cpy.chi2.ppf(1-sigRatio[0]/100.,1)/2.)
            self.MAP_CI[:, 0:2] = self._calc_confidence_band(printbool=printbool)
            self.MAP_theta[:, 1] = self.MAP_theta[:, 0] + self.MAP_CI[:, 0]
            self.MAP_theta[:, 2] = self.MAP_theta[:, 0] + self.MAP_CI[:, 1]
            self._sigmaRatio = np.exp(-cpy.chi2.ppf(1-sigRatio[1]/100.,1)/2.)
            self.MAP_CI[:, 2:4] = self._calc_confidence_band(printbool=printbool)
            self.MAP_theta[:, 3] = self.MAP_theta[:, 0] + self.MAP_CI[:, 2]
            self.MAP_theta[:, 4] = self.MAP_theta[:, 0] + self.MAP_CI[:, 3]
        elif sigRatio.size == 1:
            self.MAP_CI = np.zeros((self.ndim, 2))
            self._sigmaRatio = np.exp(-cpy.chi2.ppf(1-sigRatio[0]/100., 1)/2.)
            self.MAP_CI[:, 0:2] = self._calc_confidence_band(printbool=printbool)
            self.MAP_theta[:, 1] = self.MAP_theta[:, 0] + self.MAP_CI[:, 0]
            self.MAP_theta[:, 2] = self.MAP_theta[:, 0] + self.MAP_CI[:, 1]
        else:
            print('ERROR: The sigRatio argument needs one or two numpy array entries.')
        self.noise_level_estimate = self.MAP_theta[-1, :]

    def _compute_slope_errors(self, printbool):
        """
        Helper function that determines the asymmetric marginal uncertainties of the ``MAP_theta`` via Wilks' theorem.
        The method yields trustable confidence bands for the drift slope :math:`\hat{\zeta}` for the first order polynomial
        of the Langevin equation. In case of the third order polynomial drift, the uncertainty of the drift lope :math:`\hat{\zeta}`
        is determined via simple Gaussian propagation of uncertainties. Several assumptions make the confidence bands too
        optimistic, i.e. too narrow. The model parameters are assumed to be uncorrelated and Gaussian distributed. The model
        sample size of the MAP estimates is one. If this error propagation option is chosen for the third order drift polynomial,
        the confidence levels are fixed to cred_percentiles = numpy.array([5,1]) to transform the marginal confidence bands
        of the parameters into error via fixed transformation factors, i.e. 1.96 and 2.5525 for half-sided intervals.
        After error propagation the drift slope error is transformed back via the same factors. The normally slightly asymmetric error
        bounds are maintained and treated as if they would be half confidence intervals of a Gaussian. This formally not correct,
        but maintains the information about asymmetry of the error bands of the Wilks' theorem approximation.
        Only in the rare case of high amounts of data per window, the confidence bands computed by this procedure might be
        trustworthy in the case of a third order polynomial drift term. This issue might be solvable by introducing an estimate
        of the covariance matrix of the parameters in future.

        :param printbool: Determines whether a detailed output is printed or not.
        :type printbool: Boolean
        """

        if printbool:
            print('______________________________________________')
            print('Compute asymmetric slope confidence intervals.')
            print('______________________________________________')
        self.MAP_slope_errors = np.zeros(4)
        theta_error_sigLevel1 = np.zeros((self.ndim, 2))
        theta_error_sigLevel2 = np.zeros((self.ndim, 2))
        theta_error_sigLevel1[:, 0] = self.MAP_CI[:, 0]
        theta_error_sigLevel1[:, 1] = self.MAP_CI[:, 1]
        theta_error_sigLevel1[:, 0] = self.MAP_CI[:, 0]
        theta_error_sigLevel1[:, 1] = self.MAP_CI[:, 1]

        theta_error_sigLevel2[:, 0] = self.MAP_CI[:, 2]
        theta_error_sigLevel2[:, 1] = self.MAP_CI[:, 3]
        theta_error_sigLevel2[:, 0] = self.MAP_CI[:, 2]/2.5525
        theta_error_sigLevel2[:, 1] = self.MAP_CI[:, 3]/2.5525

        # compute worst case lower bound
        if self.drift_model == '3rd order polynomial':
            data_window_error = np.sqrt(1./(self.data_window.size) * np.var(self.data_window))
            X_error = (2 * self.MAP_theta[2, 0] + 6 * self.fixed_point_estimate *
                       self.MAP_theta[3, 0]) * data_window_error
            param1_error = 1 * theta_error_sigLevel1[1, 0]
            param2_error = 2 * self.fixed_point_estimate * theta_error_sigLevel1[2, 0]
            param3_error = 3 * self.fixed_point_estimate ** 2 * theta_error_sigLevel1[3, 0]
            self.MAP_slope_errors[0] = np.sqrt(param1_error**2 + param2_error**2 + param3_error**2 + X_error**2)

            # compute worst case upper bound
            param1_error = 1 * theta_error_sigLevel1[1, 1]
            param2_error = 2 * self.fixed_point_estimate * theta_error_sigLevel1[2, 1]
            param3_error = 3 * self.fixed_point_estimate ** 2 * theta_error_sigLevel1[3, 1]
            self.MAP_slope_errors[1] = np.sqrt(param1_error ** 2 + param2_error ** 2 + param3_error ** 2 + X_error**2)

            # compute worst case lower bound
            param1_error = 1 * theta_error_sigLevel2[1, 0]
            param2_error = 2 * self.fixed_point_estimate * theta_error_sigLevel2[2, 0]
            param3_error = 3 * self.fixed_point_estimate ** 2 * theta_error_sigLevel2[3, 0]
            self.MAP_slope_errors[2] = np.sqrt(param1_error ** 2 + param2_error ** 2 + param3_error ** 2 + X_error**2) * 2.5525

            # compute worst case upper bound
            param1_error = 1 * theta_error_sigLevel2[1, 1]
            param2_error = 2 * self.fixed_point_estimate * theta_error_sigLevel2[2, 1]
            param3_error = 3 * self.fixed_point_estimate ** 2 * theta_error_sigLevel2[3, 1]
            self.MAP_slope_errors[3] = np.sqrt(param1_error ** 2 + param2_error ** 2 + param3_error ** 2 + X_error**2) * 2.5525

        elif self.drift_model == 'linear model':
            self.MAP_slope_errors = np.array([theta_error_sigLevel1[1, 0], theta_error_sigLevel1[1, 1],
                                              theta_error_sigLevel2[1, 0], theta_error_sigLevel2[1, 1]])
        self.drift_slope[1] = self.drift_slope[0] - (self.MAP_slope_errors[0])
        self.drift_slope[2] = self.drift_slope[0] + (self.MAP_slope_errors[1])
        self.drift_slope[3] = self.drift_slope[0] - (self.MAP_slope_errors[2])
        self.drift_slope[4] = self.drift_slope[0] + (self.MAP_slope_errors[3])
        if printbool:
            print('MAP_slope_errors: ', self.MAP_slope_errors)
            print('drift_slope array: ', self.drift_slope)
            print('_____')
            print('Done!')
            print('_____')

    def _compute_slope_error_margin(self, printbool):
        """
        Helper function that determines the symmetric uncertainty bands of the MAP slope estimates via
        Gaussian propagation of uncertainties. The underlying marginal uncertainties of the ``MAP_theta``
        are estimated via Wilks' theorem and the higher deviation of each estimated ``MAP_theta`` is interpreted
        as symmetric error bound. The deviations are propagated in terms of error bounds.
        The error bound of the fixed point estimate is assumed to be equal to a Gaussian :math:`3\sigma` -confidence
        band.
        The estimation tends to be too optimistic, i.e. the uncertainty bounds are too narrow. The MAP estimation tends
        to vary to strong for subsequent windows. It is only useful in the very rare case of very high amounts of
        stationary data per window. It is recommended to use the summary statistics uncertainty bands.

        :param printbool: Determines whether a detailed output is printed or not.
        :type printbool: Boolean
        """

        if printbool:
            print('______________________________________________')
            print('Compute symmetric slope error margins.')
            print('______________________________________________')
        self.fixed_point_estimate = np.mean(self.data_window)
        data_window_error = np.sqrt(1. / (self.data_window.size) * np.var(self.data_window)) * 2.5525
        X_error_bound = abs(2 * self.MAP_theta[2, 0] + 6 * self.fixed_point_estimate *
                   self.MAP_theta[3, 0]) * data_window_error
        self.MAP_slope_margin = np.zeros(2)
        theta_error_sigLevel1 = np.zeros((self.ndim, 2))
        theta_error_sigLevel2 = np.zeros((self.ndim, 2))
        theta_error_sigLevel1[:, 0] = self.MAP_CI[:, 0]
        theta_error_sigLevel1[:, 1] = self.MAP_CI[:, 1]
        theta_margin1 = np.maximum(np.absolute(theta_error_sigLevel1[1:, 0]), theta_error_sigLevel1[1:, 1])
        theta_error_sigLevel2[:, 0] = self.MAP_CI[:, 2]
        theta_error_sigLevel2[:, 1] = self.MAP_CI[:, 3]
        theta_margin2 = np.maximum(np.absolute(theta_error_sigLevel2[1:, 0]), theta_error_sigLevel2[1:, 1])
        param1_margin = 1 * theta_margin1[0]
        param2_margin = 2 * np.absolute(self.fixed_point_estimate) * theta_margin1[1]
        param3_margin = 3 * self.fixed_point_estimate ** 2 * theta_margin1[2]
        self.MAP_slope_margin[0] = param1_margin + param2_margin + param3_margin + X_error_bound
        param1_margin = 1 * theta_margin2[0]
        param2_margin = 2 * np.absolute(self.fixed_point_estimate) * theta_margin2[1]
        param3_margin = 3 * self.fixed_point_estimate ** 2 * theta_margin2[2]
        self.MAP_slope_margin[1] = param1_margin + param2_margin + param3_margin + X_error_bound
        self.drift_slope[1] = self.drift_slope[0] - self.MAP_slope_margin[0]
        self.drift_slope[2] = self.drift_slope[0] + self.MAP_slope_margin[0]
        self.drift_slope[3] = self.drift_slope[0] - self.MAP_slope_margin[1]
        self.drift_slope[4] = self.drift_slope[0] + self.MAP_slope_margin[1]
        if printbool:
            print('MAP_slope_margin: ', self.MAP_slope_margin)
            print('drift_slope array: ', self.drift_slope)
            print('_____')
            print('Done!')
            print('_____')