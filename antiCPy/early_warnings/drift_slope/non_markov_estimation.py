import numpy as np
import scipy.stats as cpy
from scipy import optimize
import emcee
import multiprocessing as mp
import matplotlib
import matplotlib.pylab as plt
from matplotlib.animation import FuncAnimation
import celerite
from celerite import terms

from antiCPy.early_warnings.drift_slope.langevin_estimation import LangevinEstimation
from .summary_statistics_helper import summary_statistics_helper

class NonMarkovEstimation(LangevinEstimation):
    """
    Child class of ``LangevinEstimation``. Inherits its functions to guarantee similar function structure.

    :param data: A one dimensional numpy array containing the times series to analyse.
    :type data: One dimensional numpy array of floats
    :param time: A one dimensional numpy array containing the time samples of the time series data.
    :type time: One dimensional numpy array of floats
    :param X_drift_model: Defines the drift model. Default is ``'3rd order polynomial'``.
        Additionally, a ``'first order polynomial'`` or a ``'3rd order odds correlated'`` can be chosen.
        In ``'3rd order odds correlated'`` the first and third order coefficients are the same to reduce
        complexity. Corresponds to ``drift_model`` in the ``LangevinEstimation`` case.

    :type X_drift_model: str
    :param X_coupling_term: Defines the X_coupling_term. Default is ``'constant'``. Additionally, a ``'first order polynomial'``
        can be chosen.  Corresponds to ``diffusion_model`` in the ``LangevinEstimation`` case.

    :type X_coupling_term: str
    :param Y_model: Default is ``'Ornstein-Uhlenbeck noise'``. It defines automatically the Y drift model to be linear without offset
        and the Y diffusion to be constant. The drift coefficient is given by :math:`-\\frac{1}{\\theta_5^2}` .
        The diffusion coefficient is given by :math:`\\frac{1}{\\theta_5}` .

    :type Y_model: str
    :param Y_drift_model: Other model settings than ``Y_model = 'Ornstein-Uhlenbeck noise'`` are not tested yet. In principle,
        ``Y_drift_model`` can be chosen to be ``'first order polynomial'``, ``'3rd order polynomial'`` and
        ``'3rd order odds correlated'``.

    :type Y_drift_model: str
    :param Y_diffusion_model:  Other model settings than ``Y_model = 'Ornstein-Uhlenbeck noise'`` are not tested yet. In principle,
        ``'constant'`` and ``'first order polynomial'`` models can be chosen.

    :type Y_diffusion_model: str
    :param prior_type: Defines the used prior to calculate the posterior distribution of the data.
        Default is ``'Non-informative linear and Gaussian Prior'``. A flat prior can be chosen
        with ``'Flat Prior'``.

    :type prior_type: str
    :param prior_range: Customize the prior range of the estimated polynomial parameters :math:`\\theta_i`
        starting for :math:`\\theta_0` in row with index 0 and upper limits in the first column.
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
    :param activate_time_scale_separation_prior: If ``True``, a time scale separation of the observed process :math:`X` and the
        unobserved process :math:`Y` is assumed. Default is ``False``.

    :type activate_time_scale_separation_prior: bool
    :param slow_process: If ``activate_time_scale_separation_prior = True``, define the slow process via ``'X'`` or ``'Y'``, respectively.
    :type slow_process: str
    :param time_scale_separation_factor: If ``activate_time_scale_separation_prior = True``, define the factor by which you assume
        the time scales of :math:`X` and :math:`Y` to differ.

    :type time_scale_separation_factor: int
    :param max_likelihood_starting_guesses: If ``None``, the MAP starting guesses (essentially maximum likelihood because of flat priors) are
        are computed starting with all parameters set to one. In general, you can pass a one-dimensional numpy array with
        ``ndim`` entries to this argument to define different starting values.

        .. note::

            This is especially necessary, if you enable the time scale separation, since the default array of ones will
            contradict normally to the assumed two time scale model.

    :type max_likelihood_starting_guesses: One-dimensional numpy array of float

    """
    def __init__(self, data, time, X_drift_model='3rd order polynomial',
                 Y_model='Ornstein-Uhlenbeck noise', Y_drift_model='no offset first order',
                 X_coupling_term='constant', Y_diffusion_model='constant',
                 prior_type='Non-informative linear and Gaussian Prior', prior_range=None,
                 scales=np.array([4,8]), activate_time_scale_separation_prior=False, slow_process=None,
                 time_scale_separation_factor=None, max_likelihood_starting_guesses=None, detrending_of_whole_dataset = None,
                 gauss_filter_mode = 'reflect', gauss_filter_sigma = 6, gauss_filter_order = 0, gauss_filter_cval = 0.0,
                 gauss_filter_truncate = 4.0, plot_detrending = False):
        self.antiCPyObject = 'NonMarkovEstimation'
        self.Y_model = Y_model
        if X_drift_model == '3rd order polynomial':
            self.num_X_drift_params = 4
        elif X_drift_model == '3rd order odds correlated':
            self.num_X_drift_params = 3
        elif X_drift_model == 'first order polynomial':
            self.num_X_drift_params = 2
        elif X_drift_model == 'no offset first order':
            self.num_X_drift_params = 1
        else:
            print('ERROR: Drift model of X is not defined!')
        if Y_model == 'Ornstein-Uhlenbeck noise':
            self.Y_drift_model = 'no offset first order'
            self.Y_diffusion_model = 'constant'
            self.num_Y_model_params = 1
        else:
            if Y_drift_model == '3rd order polynomial':
                self.num_Y_drift_params = 4
            elif Y_drift_model == 'first order polynomial':
                self.num_Y_drift_params = 2
            elif Y_drift_model == 'no offset first order':
                self.num_Y_drift_params = 1
            else:
                print('ERROR: Drift model of Y is not defined!')
            self.num_Y_model_params = self.num_Y_drift_params + self.num_Y_diff_params
            if Y_diffusion_model == 'constant':
                self.num_Y_diff_params = 1
            elif Y_diffusion_model == 'first order polynomial':
                self.num_Y_diff_params = 2
            else:
                print('ERROR: Diffusion model of X is not defined!')
        if X_coupling_term == 'constant':
            self.num_X_coupling_term_params = 1
        elif X_coupling_term == 'first order polynomial':
            self.num_X_coupling_term_params = 2
        else:
            print('ERROR: Diffusion model of X is not defined!')

        self.ndim = self.num_X_drift_params + self.num_X_coupling_term_params + self.num_Y_model_params
        if np.all(max_likelihood_starting_guesses == None):
            self.max_likelihood_starting_guesses = np.ones(self.ndim)
        else:
            self.max_likelihood_starting_guesses = max_likelihood_starting_guesses
        self.nwalkers = None
        self.nsteps = None
        self.nburn = None
        if np.all(
                prior_range == None) and X_drift_model == '3rd order polynomial' and X_coupling_term == 'constant' and Y_model == 'Ornstein-Uhlenbeck noise':
            self.prior_range = np.array([[50., -50.], [50., -50.], [50., -50.], [50., -50.], [50., 0.], [50., 0.]])
        elif np.all(
                prior_range == None) and X_drift_model == 'first order polynomial' and X_coupling_term == 'constant' and Y_model == 'Ornstein-Uhlenbeck noise':
            self.prior_range = np.array([[50., -50.], [50., -50.], [50., 0.], [50., 0.]])
        elif np.all(prior_range != None):
            self.prior_range = prior_range
        else:
            print('ERROR: No prior range defined for the chosen model parameters.')
        if prior_type == 'Non-informative linear and Gaussian Prior':
            self.scales = scales
        self.prior_type = prior_type
        self.activate_time_scale_separation_prior = activate_time_scale_separation_prior
        if activate_time_scale_separation_prior:
            if not time_scale_separation_factor == None:
                self.time_scale_separation_factor = time_scale_separation_factor
            else:
                print(
                    'ERROR: Time scale separation prior is activated without defining a time scale separation factor!')
            if not slow_process == None:
                self.slow_process = slow_process
            else:
                print('ERROR: Time scale separation prior is activated without defining the slow process!')
        else:
            self.slow_process = slow_process
            self.time_scale_separation_factor = time_scale_separation_factor
        self.theta = np.zeros(self.ndim)
        if detrending_of_whole_dataset == None:
            self.data = data
        else:
            self.data, self.slow_trend = self.detrend(time, data, detrending_of_whole_dataset,
                                                      gauss_filter_mode, gauss_filter_sigma,
                                                      gauss_filter_order, gauss_filter_cval,
                                                      gauss_filter_truncate, plot_detrending)
        self.data_size = data.size
        self.time = time
        self.dt = time[1] - time[0]
        self.drift_slope = np.zeros(5)
        self.noise_level_estimate = np.zeros(5)
        self.X_coupling_estimate = np.zeros(5)
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
        self.X_coupling_density_obj = None
        self.X_coupling_storage = None
        self.X_coupling_samples = None
        self.MAP_theta = None
        self.MAP_CI = None
        self.MAP_noise_errors = None
        self.MAP_noise_margin = None
        self._sigmaRatio = None
        self._optimum_solution = None
        self.MAP_slope_errors = None
        self.MAP_slope_margin = None
        self.credibility_bands = None

        self.X_drift_model = X_drift_model
        self.drift_model = X_drift_model  # to make it compatible to MarkovModel object
        self.diffusion_model = X_coupling_term  # to make it compatible to MarkovModel object
        self.Y_drift_model = Y_drift_model
        self.X_coupling_term = X_coupling_term
        self.Y_diffusion_model = Y_diffusion_model
        self.OU_param_estimate = np.zeros(5)
        self.OU_param_storage = None
        self.OU_kernel_density_obj = None
        self.OU_param_samples = None
        self.D2Xi = None

    @staticmethod
    def D1(data, theta, Y_model, drift_model, component):
        '''
        Calls the static `calc_D1` function and returns its value. Passing of parameters is adapted to the NonMarkovEstimation model.
        '''
        if Y_model == 'Ornstein-Uhlenbeck noise':
            if component == 'X':
                if drift_model == '3rd order polynomial':
                    return __class__.calc_D1(theta[:4], data, drift_model)
                if drift_model == '3rd order odds correlated':
                    return __class__.calc_D1(theta[:3], data, drift_model)
                elif drift_model == 'first order polynomial':
                    return __class__.calc_D1(theta[:2], data, drift_model)
                elif drift_model == 'no offset first order':
                    return __class__.calc_D1(theta, data, drift_model)
            elif component == 'Y':
                return __class__.calc_D1(np.array([1. / theta[-1]]), data, drift_model)
        else:
            print('ERROR: Y model is unknown.')

    @staticmethod
    def D2(data, theta, Y_model, diffusion_model, component):
        '''
        Calls the static `calc_D2` function and returns its value. Passing of parameters is adapted to the NonMarkovEstimation model.
        '''
        if Y_model == 'Ornstein-Uhlenbeck noise':
            if component == 'X':
                if diffusion_model == 'constant':
                    return __class__.calc_D2(theta[:-1], data, diffusion_model)
                elif diffusion_model == 'first order polynomial':
                    return __class__.calc_D2(theta[-3:-1], data, diffusion_model)
            elif component == 'Y':
                return __class__.calc_D2(np.array([1, 1. / theta[-1]]), data, diffusion_model)

    @staticmethod
    def log_prior(theta, X_drift_model, X_coupling_term, Y_model, prior_type, prior_range, scales, data_window,
                  activate_time_scale_separation_prior, slow_process, time_scale_separation_factor,
                  print_time_scale_info):
        '''
        Returns the logarithmic prior probability for a given set of parameters depending on the drift model and coupling
        term of the observed variable X and the unobserved drift-diffusion process Y that couples into X via the specified
        coupling term. The prior probability is computed within the given `prior_range` of the parameters `theta`.
        A flat prior for the parameters apart from restriction to positive constant diffusion and a non-informative prior for
        linear parts and Gaussian priors for higher orders are implemented. The standard deviation of the Gaussian priors
        is given by the `scales` variable.
        If two separate time scales are assumed via ``activate_time_scale_separation_prior=True``, the ``slow_process``
        and the ``time_scale_separation_factor`` can be specified. The time scales for each sampled parameter set are
        approximated by :math:`\left\lvert \\frac{1}{f'(x)}\\right\\rvert` and :math:`\left\lvert\\frac{1}{f'(y)}\\right\\rvert` with prime denoting derivative with
        respect to the variables :math:`x` and :math:`y`, respectively.
        '''

        if X_drift_model == '3rd order polynomial' and X_coupling_term == 'constant' and Y_model == 'Ornstein-Uhlenbeck noise':
            if prior_type == 'Flat Prior':
                if (prior_range[0, 0] > theta[0] > prior_range[0, 1]) and (
                        prior_range[1, 0] > theta[1] > prior_range[1, 1]) and (
                        prior_range[2, 0] > theta[2] > prior_range[2, 1]) and (
                        prior_range[3, 0] > theta[3] > prior_range[3, 1]) and (
                        prior_range[4, 0] > theta[4] > prior_range[4, 1]) and (
                        prior_range[5, 0] > theta[5] > prior_range[5, 1]):
                    return 0
                else:
                    return - np.inf
            elif prior_type == 'Non-informative linear and Gaussian Prior':
                if (prior_range[0, 0] > theta[0] > prior_range[0, 1]) and (
                        prior_range[1, 0] > theta[1] > prior_range[1, 1]) and (
                        prior_range[2, 0] > theta[2] > prior_range[2, 1]) and (
                        prior_range[3, 0] > theta[3] > prior_range[3, 1]) and (
                        prior_range[4, 0] > theta[4] > prior_range[4, 1]) and (
                        prior_range[5, 0] > theta[5] > prior_range[5, 1]):
                    if activate_time_scale_separation_prior:
                        fixed_point_estimate = np.mean(data_window)
                        X_time_scale = abs(1. / (theta[1] + 2 * theta[2] * fixed_point_estimate + 3 * theta[
                            3] * fixed_point_estimate ** 2))
                        Y_time_scale = theta[5] ** 2
                        if slow_process == 'X':
                            if print_time_scale_info:
                                print('time_scale_separation_factor * fast_time_scale: ', time_scale_separation_factor * Y_time_scale)
                                print('Defined slow time scale Y: ', X_time_scale)
                                print('Time scale separation condition fulfilled: ', np.abs(time_scale_separation_factor * Y_time_scale) < X_time_scale)
                            if time_scale_separation_factor * Y_time_scale < X_time_scale:
                                return (-3. / 2.) * np.log(1 + theta[1] ** 2) + (
                                            (-3. / 2.) * np.log(1 + 1. / theta[5] ** 2) - np.log(
                                        1. / theta[5])) + np.log(
                                    cpy.norm.pdf(theta[4], loc=0, scale=scales[0])) + np.log(
                                    cpy.norm.pdf(theta[2], loc=0, scale=scales[0])) + np.log(
                                    cpy.norm.pdf(theta[3], loc=0, scale=scales[1]))
                            else:
                                return - np.inf
                        elif slow_process == 'Y':
                            if print_time_scale_info:
                                print('time_scale_separation_factor * X_time_scale: ', time_scale_separation_factor * X_time_scale)
                                print('Defined slow time scale Y: ', Y_time_scale)
                                print('Time scale separation condition fulfilled: ', np.abs(time_scale_separation_factor * X_time_scale) < Y_time_scale)
                            if time_scale_separation_factor * X_time_scale < Y_time_scale:
                                return (-3. / 2.) * np.log(1 + theta[1] ** 2) + (
                                            (-3. / 2.) * np.log(1 + 1. / theta[5] ** 2) - np.log(
                                        1. / theta[5])) + np.log(
                                    cpy.norm.pdf(theta[4], loc=0, scale=scales[0])) + np.log(
                                    cpy.norm.pdf(theta[2], loc=0, scale=scales[0])) + np.log(
                                    cpy.norm.pdf(theta[3], loc=0, scale=scales[1]))
                            else:
                                return - np.inf
                        else:
                            print('ERROR: slow_process unknown.')
                    return (-3. / 2.) * np.log(1 + theta[1] ** 2) + (
                                (-3. / 2.) * np.log(1 + 1. / theta[5] ** 2) - np.log(1. / theta[5])) + np.log(
                        cpy.norm.pdf(theta[4], loc=0, scale=scales[0])) + np.log(
                        cpy.norm.pdf(theta[2], loc=0, scale=scales[0])) + np.log(
                        cpy.norm.pdf(theta[3], loc=0, scale=scales[1]))
                else:
                    return - np.inf
        elif X_drift_model == 'first order polynomial' and X_coupling_term == 'constant' and Y_model == 'Ornstein-Uhlenbeck noise':
            if prior_type == 'Flat Prior':
                if (prior_range[0, 0] > theta[0] > prior_range[0, 1]) and (
                        prior_range[1, 0] > theta[1] > prior_range[1, 1]) and (
                        prior_range[2, 0] > theta[2] > prior_range[2, 1]) and (
                        prior_range[3, 0] > theta[3] > prior_range[3, 1]):
                    return 0
                else:
                    return - np.inf
            elif prior_type == 'Non-informative linear and Gaussian Prior':
                if (prior_range[0, 0] > theta[0] > prior_range[0, 1]) and (
                        prior_range[1, 0] > theta[1] > prior_range[1, 1]) and (
                        prior_range[2, 0] > theta[2] > prior_range[2, 1]) and (
                        prior_range[3, 0] > theta[3] > prior_range[3, 1]):
                    return (-3. / 2.) * np.log(1 + theta[1] ** 2) + (
                                (-3. / 2.) * np.log(1 + 1. / theta[3] ** 2) - np.log(1. / theta[3])) + np.log(
                        cpy.norm.pdf(theta[2], loc=0, scale=scales[0]))
                else:
                    return - np.inf
        elif X_drift_model == '3rd order odds correlated' and X_coupling_term == 'constant' and Y_model == 'Ornstein-Uhlenbeck noise':
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
                    if activate_time_scale_separation_prior:
                        fixed_point_estimate = np.mean(data_window)
                        X_time_scale = abs(1. / (theta[1] + 2 * theta[2] * fixed_point_estimate + 3 * (
                                    1 - theta[1]) * fixed_point_estimate ** 2))
                        Y_time_scale = theta[4] ** 2
                        if slow_process == 'X':
                            if print_time_scale_info:
                                print('time_scale_separation_factor * Y_time_scale: ', time_scale_separation_factor * Y_time_scale)
                                print('Defined slow time scale X: ', X_time_scale)
                                print('Time scale separation condition fulfilled: ', np.abs(time_scale_separation_factor * Y_time_scale) < X_time_scale)
                            if time_scale_separation_factor * Y_time_scale < X_time_scale:
                                return ((-3. / 2.) * np.log(1 + 1. / theta[4] ** 2) - np.log(1. / theta[4])) + np.log(
                                    cpy.norm.pdf(theta[3], loc=0, scale=scales[0])) + np.log(
                                    cpy.norm.pdf(theta[1], loc=0, scale=scales[0])) + np.log(
                                    cpy.norm.pdf(theta[2], loc=0, scale=scales[0]))
                            else:
                                return - np.inf
                        elif slow_process == 'Y':
                            if print_time_scale_info:
                                print('time_scale_separation_factor * X_time_scale: ',time_scale_separation_factor * X_time_scale)
                                print('Defined slow time scale Y: ', Y_time_scale)
                                print('Time scale separation condition fulfilled: ', np.abs(time_scale_separation_factor * X_time_scale) < Y_time_scale)
                            if time_scale_separation_factor * X_time_scale < Y_time_scale:
                                return ((-3. / 2.) * np.log(1 + 1. / theta[4] ** 2) - np.log(1. / theta[4])) + np.log(
                                    cpy.norm.pdf(theta[3], loc=0, scale=scales[0])) + np.log(
                                    cpy.norm.pdf(theta[1], loc=0, scale=scales[0])) + np.log(
                                    cpy.norm.pdf(theta[2], loc=0, scale=scales[0]))
                            else:
                                return - np.inf
                        else:
                            print('ERROR: slow_process unknown.')
                else:
                    return - np.inf
        elif X_drift_model == 'first order polynomial' and X_coupling_term == 'constant' and Y_model == 'Ornstein-Uhlenbeck noise':
            if prior_type == 'Flat Prior':
                if (prior_range[0, 0] > theta[0] > prior_range[0, 1]) and (
                        prior_range[1, 0] > theta[1] > prior_range[1, 1]) and (
                        prior_range[2, 0] > theta[2] > prior_range[2, 1]) and (
                        prior_range[3, 0] > theta[3] > prior_range[3, 1]):
                    return 0
                else:
                    return - np.inf
            elif prior_type == 'Non-informative linear and Gaussian Prior':
                if (prior_range[0, 0] > theta[0] > prior_range[0, 1]) and (
                        prior_range[1, 0] > theta[1] > prior_range[1, 1]) and (
                        prior_range[2, 0] > theta[2] > prior_range[2, 1]) and (
                        prior_range[3, 0] > theta[3] > prior_range[3, 1]):
                    return (-3. / 2.) * np.log(1 + theta[1] ** 2) + (
                                (-3. / 2.) * np.log(1 + theta[3] ** 2) - np.log(theta[3])) + np.log(
                        cpy.norm.pdf(theta[2], loc=0, scale=scales[0]))
                else:
                    return - np.inf

    @staticmethod
    def log_likelihood(theta, data, X_drift_model, Y_drift_model, X_coupling_term, Y_diffusion_model, Y_model, dt):
        '''
        Returns the logarithmic likelihood of the data for the given model parametrization. It is given by the modified
        short-time propagator of [Willers2021]_ .
        '''
        x_iplus1 = data[2:]
        x_i = data[1:-1]
        x_iminus1 = data[0:-2]
        y_iminus1 = (x_i - x_iminus1 - __class__.D1(x_iminus1, theta, Y_model, X_drift_model, 'X') * dt) / (
                    __class__.D2(x_iminus1, theta, Y_model, X_coupling_term, 'X') * dt)
        D2Xi = __class__.D2(x_i, theta, Y_model, X_coupling_term, 'X')
        sigma2_theta = __class__.D2(y_iminus1, theta, Y_model, Y_diffusion_model, 'Y') ** 2 * D2Xi ** 2 * dt ** 3
        mu_theta = x_i + __class__.D1(x_i, theta, Y_model, X_drift_model,
                                      'X') * dt + D2Xi * y_iminus1 * dt + __class__.D1(y_iminus1, theta, Y_model,
                                                                                       Y_drift_model,
                                                                                       'Y') * D2Xi * dt ** 2
        LLH = np.sum(-0.5 * np.log(2 * np.pi * sigma2_theta) - (x_iplus1 - mu_theta) ** 2 / (2 * sigma2_theta))
        return LLH

    @staticmethod
    def log_posterior(theta):
        '''
        Returns the logarithmic posterior probability of the data for the given model parametrization.
        '''
        lg_prior = __class__.log_prior(theta, shared_memory_dict['X_drift_model'],
                                       shared_memory_dict['X_coupling_term'], shared_memory_dict['Y_model'],
                                       shared_memory_dict['prior_type'], shared_memory_dict['prior_range'],
                                       shared_memory_dict['scales'], shared_memory_dict['data_window'],
                                       shared_memory_dict['activate_time_scale_separation_prior'],
                                       shared_memory_dict['slow_process'],
                                       shared_memory_dict['time_scale_separation_factor'],
                                       shared_memory_dict['print_time_scale_info'])
        if not np.isfinite(lg_prior):
            return - np.inf
        return lg_prior + __class__.log_likelihood(theta, shared_memory_dict['data_window'],
                                                   shared_memory_dict['X_drift_model'],
                                                   shared_memory_dict['Y_drift_model'],
                                                   shared_memory_dict['X_coupling_term'],
                                                   shared_memory_dict['Y_diffusion_model'],
                                                   shared_memory_dict['Y_model'], shared_memory_dict['dt'])

    @staticmethod
    def neg_log_posterior(theta):
        """
        Calculates the negative logarithmic posterior for a given tuple of parameters :math:`\theta`.
        """
        return (-1) * __class__.log_posterior(theta)

    @staticmethod
    def init_parallel_EnsembleSampler(data_window, X_drift_model, Y_drift_model, Y_diffusion_model, X_coupling_term,
                                      Y_model, dt, prior_type, prior_range, scales,
                                      activate_time_scale_separation_prior,
                                      time_scale_separation_factor, slow_process, print_time_scale_info):
        """
        Initializes the workers for the MCMC sampling if ``MCMC_parallelization_method = 'multiprocessing'`` in ``perform_resilience_scan(...)``.
        """
        global shared_memory_dict
        shared_memory_dict = {}
        shared_memory_dict['data_window'] = data_window
        shared_memory_dict['X_drift_model'] = X_drift_model
        shared_memory_dict['Y_drift_model'] = Y_drift_model
        shared_memory_dict['Y_diffusion_model'] = Y_diffusion_model
        shared_memory_dict['X_coupling_term'] = X_coupling_term
        shared_memory_dict['Y_model'] = Y_model
        shared_memory_dict['dt'] = dt
        shared_memory_dict['prior_type'] = prior_type
        shared_memory_dict['prior_range'] = prior_range
        shared_memory_dict['scales'] = scales
        shared_memory_dict['activate_time_scale_separation_prior'] = activate_time_scale_separation_prior
        shared_memory_dict['time_scale_separation_factor'] = time_scale_separation_factor
        shared_memory_dict['slow_process'] = slow_process
        shared_memory_dict['print_time_scale_info'] = print_time_scale_info

    def compute_posterior_samples(self, print_AC_tau, ignore_AC_error, thinning_by, print_progress,
                                  MCMC_parallelization_method=None, num_processes=None, num_chop_chains=None,
                                  MCMC_AC_estimate = 'standard'):
        '''
        Compute the `theta_array` with :math:`nwalkers \cdot nsteps` Markov Chain Monte Carlo (MCMC) samples.
        If `ignore_AC_error = False` the calculation will terminate with error if
        the autocorrelation of the sampled chains is too high compared to the chain length.
        Otherwise, the highest autocorrelation length will be used to thin the sampled chains.
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
                with mp.Pool(processes=num_processes, initializer=self.init_parallel_EnsembleSampler,
                             initargs=(self.data_window, self.X_drift_model, self.Y_drift_model, self.Y_diffusion_model,
                                       self.X_coupling_term, self.Y_model, self.dt, self.prior_type, self.prior_range,
                                       self.scales, self.activate_time_scale_separation_prior,
                                       self.time_scale_separation_factor, self.slow_process, False)) as pool:
                    sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_posterior, pool=pool)
                    sampler.run_mcmc(self.starting_guesses, self.nsteps, progress=print_progress)
            elif num_processes == 'all':
                with mp.Pool(processes=mp.cpu_count(), initializer=self.init_parallel_EnsembleSampler,
                             initargs=(self.data_window, self.X_drift_model, self.Y_drift_model, self.Y_diffusion_model,
                                       self.X_coupling_term, self.Y_model, self.dt, self.prior_type, self.prior_range,
                                       self.scales, self.activate_time_scale_separation_prior,
                                       self.time_scale_separation_factor, self.slow_process, False)) as pool:
                    sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_posterior, pool=pool)
                    sampler.run_mcmc(self.starting_guesses, self.nsteps, progress=print_progress)
            else:
                with mp.Pool(processes=int(mp.cpu_count() / 2.), initializer=self.init_parallel_EnsembleSampler,
                             initargs=(self.data_window, self.X_drift_model, self.Y_drift_model, self.Y_diffusion_model,
                                       self.X_coupling_term, self.Y_model, self.dt, self.prior_type, self.prior_range,
                                       self.scales, self.activate_time_scale_separation_prior,
                                       self.time_scale_separation_factor, self.slow_process, False)) as pool:
                    sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_posterior, pool=pool)
                    sampler.run_mcmc(self.starting_guesses, self.nsteps, progress=print_progress)
        elif MCMC_parallelization_method == 'chop_chain':
            print('ERROR: chop chain solution for the NonMarkovEstimation is not implemented yet.')
        if MCMC_parallelization_method == None or MCMC_parallelization_method == 'multiprocessing':
            if ignore_AC_error == False:
                if MCMC_AC_estimate == 'standard':
                    tau = sampler.get_autocorr_time()
                elif MCMC_AC_estimate == 'alternative':
                    chains = sampler.get_chain(discard = self.nburn).T
                    tau = np.zeros(self.ndim)
                    for k in range(self.ndim):
                        thin = max(1, int(0.05 * self.autocorr_new(chains[:, :, k])))
                        tau[k] = self.autocorr_ml(chains[:,:,k], thin = thin)
            elif ignore_AC_error:
                tau = thinning_by
                thin = thinning_by
            if ignore_AC_error == False:
                if tau.any() >= 1:
                    thin = int(np.max(tau))
                else:
                    thin = 1
            flat_samples = sampler.get_chain(discard=self.nburn, thin=np.max([1,thin]), flat=True)  # thin= int(np.max(tau))
            if print_AC_tau:
                print('tau: ', tau)
        self.theta_array = np.zeros((self.ndim, flat_samples[:, 0].size))
        for i in range(self.ndim):
            self.theta_array[i, :] = np.transpose(flat_samples[:, i])

    def declare_MAP_starting_guesses(self, nwalkers=None, nsteps=None, print_time_scale_info = False):
        '''
        Declare the maximum a posterior (MAP) starting guesses for a MCMC sampling with `nwalkers`
        and `nsteps`.
        '''
        if nwalkers != None and nsteps != None:
            self.nwalkers = nwalkers
            self.nsteps = nsteps
        if self.nwalkers == None and nwalkers == None or self.nsteps == None and nsteps == None:
            print('ERROR: nwalkers and/or nsteps is not yet defined!')
        self.init_parallel_EnsembleSampler(self.data_window, self.X_drift_model, self.Y_drift_model,
                                           self.Y_diffusion_model, self.X_coupling_term, self.Y_model, self.dt,
                                           self.prior_type, self.prior_range, self.scales,
                                           self.activate_time_scale_separation_prior, self.time_scale_separation_factor,
                                           self.slow_process, print_time_scale_info)
        res = optimize.minimize(self.neg_log_posterior, x0=self.max_likelihood_starting_guesses, method='Nelder-Mead')
        MAP_results = res['x']
        # print('MAP results: ', MAP_results)
        self.starting_guesses = np.ones((self.nwalkers, self.ndim))
        for i in range(self.ndim):
            self.starting_guesses[:, i] = MAP_results[i] * (0.5 + np.random.rand(self.nwalkers))
            self.starting_guesses[self.starting_guesses[:, i] > self.prior_range[i, 0], i] = self.prior_range[i, 0] - 1
            self.starting_guesses[self.starting_guesses[:, i] < self.prior_range[i, 1], i] = self.prior_range[i, 1] + 1
        if self.activate_time_scale_separation_prior:
            for i in range(self.starting_guesses.shape[0]):
                fixed_point_estimate = np.mean(self.data_window)
                if self.X_drift_model == '3rd order odds correlated':
                    X_time_scale = abs(1. / (self.starting_guesses[i, 1] + 2 * self.starting_guesses[
                        i, 2] * fixed_point_estimate + 3 * (
                                                         1 - self.starting_guesses[i, 1]) * fixed_point_estimate ** 2))
                    Y_time_scale = self.starting_guesses[i, 4] ** 2
                elif self.X_drift_model == '3rd order polynomial':
                    X_time_scale = abs(1. / (self.starting_guesses[i, 1] + 2 * self.starting_guesses[
                        i, 2] * fixed_point_estimate + 3 * self.starting_guesses[i, 3] * fixed_point_estimate ** 2))
                    Y_time_scale = self.starting_guesses[i, 5] ** 2
                else:
                    print('ERROR: Time scale separation prior not implemented yet for the chosen X_drift_model.')
                if self.slow_process == 'X':
                    while (self.time_scale_separation_factor * Y_time_scale > X_time_scale):
                        for j in range(self.ndim):
                            random_theta = MAP_results[j] * (0.5 + np.random.rand(1))
                            if (self.prior_range[j, 0] > random_theta) and (random_theta > self.prior_range[j, 1]):
                                self.starting_guesses[i, j] = random_theta
                            elif random_theta > self.prior_range[j, 0]:
                                self.starting_guesses[i, j] = self.prior_range[j, 0] - 1
                            else:
                                self.starting_guesses[i, j] = self.prior_range[j, 1] + 1
                        if self.X_drift_model == '3rd order odds correlated':
                            X_time_scale = abs(1. / (self.starting_guesses[i, 1] + 2 * self.starting_guesses[
                                i, 2] * fixed_point_estimate + 3 * (1 - self.starting_guesses[
                                i, 1]) * fixed_point_estimate ** 2))
                            Y_time_scale = self.starting_guesses[i, 4] ** 2
                        elif self.X_drift_model == '3rd order polynomial':
                            X_time_scale = abs(1. / (self.starting_guesses[i, 1] + 2 * self.starting_guesses[
                                i, 2] * fixed_point_estimate + 3 * self.starting_guesses[
                                                         i, 3] * fixed_point_estimate ** 2))
                            Y_time_scale = self.starting_guesses[i, 5] ** 2
                elif self.slow_process == 'Y':
                    while self.time_scale_separation_factor * X_time_scale > Y_time_scale:
                        for j in range(self.ndim):
                            random_theta = MAP_results[j] * (0.5 + np.random.rand(1))
                            if (self.prior_range[j, 0] > random_theta) and (random_theta > self.prior_range[j, 1]):
                                self.starting_guesses[i, j] = random_theta
                            elif random_theta > self.prior_range[j, 0]:
                                self.starting_guesses[i, j] = self.prior_range[j, 0] - 1
                            else:
                                self.starting_guesses[i, j] = self.prior_range[j, 1] + 1
                        if self.X_drift_model == '3rd order odds correlated':
                            X_time_scale = abs(1. / (self.starting_guesses[i, 1] + 2 * self.starting_guesses[
                                i, 2] * fixed_point_estimate + 3 * (1 - self.starting_guesses[
                                i, 1]) * fixed_point_estimate ** 2))
                            Y_time_scale = self.starting_guesses[i, 4] ** 2
                        elif self.X_drift_model == '3rd order polynomial':
                            X_time_scale = abs(1. / (self.starting_guesses[i, 1] + 2 * self.starting_guesses[
                                i, 2] * fixed_point_estimate + 3 * self.starting_guesses[
                                                         i, 3] * fixed_point_estimate ** 2))
                            Y_time_scale = self.starting_guesses[i, 5] ** 2

    def calc_drift_slope_and_noise(self, slope_grid, noise_grid, OU_grid, X_coupling_grid,
                                   n_joint_samples=50000, n_slope_samples=50000, n_noise_samples=50000,
                                   n_OU_param_samples=50000, n_X_coupling_samples=50000,
                                   cred_percentiles=np.array([16, 1]), print_AC_tau=False, print_time_scale_info = False,
                                   ignore_AC_error=False, thinning_by=60, print_progress=False, detrending_per_window = None,
                                   gauss_filter_mode='reflect',
                                   gauss_filter_sigma=6, gauss_filter_order=0, gauss_filter_cval=0.0,
                                   gauss_filter_truncate=4.0, plot_detrending=False,
                                   print_details=False, MCMC_parallelization_method=None, num_processes=None,
                                   num_chop_chains=None, MCMC_AC_estimate = 'standard'):
        '''
        Calculates the drift slope estimate :math:`\hat{\zeta}` and the noise level :math:`\hat{\psi}`
        from the MCMC sampled parameters of the non-Markovian model for a given `window_size` and given `window_shift`.
        In the course of the computations the parameters `joint_samples`, a `noise_kernel_density_obj`,
        a `slope_kernel_density_obj`, a `X_coupling_density_obj`, a `OU_kernel_density_obj`, the `drift_slope` with credibility intervals,
        the `noise_estimates`, the modified `noise_level` :math:`\hat{\psi}` with credibility intervals as well as the
        `X_coupling_estimate` and `OU_param_estimate` with credibility invervals and in case of a third order polynomial
        drift the estimated `fixed_point_estimate` of the window will be stored.

        :param slope_grid: Array on which the drift slope kernel density estimate is evaluated.
        :type slope_grid: One-dimensional numpy array of floats.
        :param noise_grid: Array on which the noise level :math:`\hat{\psi}` kernel density estimate is evaluated.
        :type noise_grid: One-dimensional numpy array of floats.
        :param OU_grid: Array on which the Ornstein Uhlenbeck (OU) parameter's kernel density estimate is evaluated.
        :type OU_grid: One-dimensional numpy array of floats.
        :param X_coupling_grid: Array on which the X coupling strength kernel density estimate is evaluated.
        :type X_coupling_grid: One-dimensional numpy array of floats.
        :param nwalkers: Number of walkers that are initialized for the MCMC sampling via the package `emcee`.
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
                        marginal probability distribution of the diffusion model is chosen constant.
                        Default is 50000.
        :type n_noise_samples: int
        :param n_X_coupling_samples: Number of X coupling samples that are drawn from the estimated posterior
                        probability. Default is 50000.
        :type n_X_coupling_samples: int
        :param n_OU_param_samples: Number of OU parameter samples that are drawn from the estimated posterior
                        probability. Default is 50000.
        :type n_OU_param_samples: int
        :param cred_percentiles: Two entries to define the percentiles of the calculated credibility bands
                        of the estimated parameters. Default is `numpy.array([16,1])`.
        :type cred_percentiles: One-dimensional numpy array of int
        :param print_AC_tau: If `True` the estimated autocorrelation lengths of the Markov chains is shown.
                        The maximum length is used for thinning of the chains. Default is `False`.
        :type prind_AC_tau: Boolean
        :param ignore_AC_error: If `True` the autocorrelation lengths of the Markov chains is not estimated
                        and thus, it is not checked whether the chains are too short to give an reliable
                        autocorrelation estimate. This avoids error interruption of the procedure, but
                        can lead to unreliable results if the chains are too short. The option should
                        be chosen in order to save computation time, e.g. when debugging your code
                        with short Markov chains. Default is `False`.
        :type ignore_AC_error: Boolean
        :param thinning_by: If `ignore_AC_error = True` the chains are thinned by `thinning_by`. Every
                        `thinning_by`-th data point is stored. Default is 60.
        :type thinning_by: int
        :param print_progress: If `True` the progress of the MCMC sampling is shown. Default is `False`.
        :type print_progress: Boolean
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
        :param print_time_scale_info: If `True`, control values are plotted during each prior evaluation. First, the
                        fast time scale estimate times the ``time_scale_separation_factor`` is printed. Second, the slow
                        time scale estimate and third, a check up whether the desired time scale separation is fulfilled
                        are printed.
        :type print_time_scale_info: bool
        :param MCMC_parallelization_method: Default is `None`. If `None` the basic serial MCMC computation is performed. If
                        `MCMC_parallelization_method = 'multiprocessing'`, a multiprocessing pool with `num_processes`
                        is used to accelerate MCMC sampling. If `MCMC_parallelization_method = 'chop_chain'` is used, the
                        total length of the desired Markov chain is divided into `'chop_chain'` parts each of which is
                        sampled in parallel and joined together in the end.
        :type MCMC_parallelization_method: str
        :param num_processes: Default is ``None``. If ``'half'``, almost half of the CPU kernels are used. If  ``'all'`` ,
                        all CPU kernels are used. If integer number, the defined number of CPU kernels is used for
                        multiprocessing.
        :type num_processes: str or int
        :param num_chop_chains: Number by which the total length of the Markov chain is divided. Each slice is sampled in parallel and
                        joined together in the end of the calculations.
        :type num_chop_chains: int
        :param MCMC_AC_estimate: If default `'standard'` is used, emcee's ``.get_autocorr_time()`` is applied to estimate th
                        sampled Markov chain's autocorrelation length for thinning. In some cases the estimation procedure requires longer chains
                        than you can run and does not converge at all. In such situations you can try to estimate the autocorrelation length with
                        parametric models following the suggestions of the `emcee documentation <https://emcee.readthedocs.io/en/stable/tutorials/autocorr/>`_
                        via ``MCMC_AC_estimate = 'alternative'``.
        :type MCMC_AC_estimate: str
        '''
        self.data_window = np.roll(self.data, shift=- self.window_shift)[:self.window_size]
        self.time_window = np.roll(self.time, shift=- self.window_shift)[:self.window_size]
        if detrending_per_window != None:
            self.data_window, self.slow_trend = self.detrend(self.time_window, self.data_window,
                                                                  detrending_per_window, gauss_filter_mode,
                                                                  gauss_filter_sigma,gauss_filter_order,
                                                                  gauss_filter_cval, gauss_filter_truncate,
                                                                  plot_detrending)
        self._prepare_data(printbool=print_details)
        self.declare_MAP_starting_guesses(print_time_scale_info=print_time_scale_info)
        self.compute_posterior_samples(print_AC_tau=print_AC_tau, ignore_AC_error=ignore_AC_error,
                                       thinning_by=thinning_by, print_progress=print_progress,
                                       MCMC_parallelization_method=MCMC_parallelization_method,
                                       num_processes=num_processes, num_chop_chains=num_chop_chains,
                                       MCMC_AC_estimate=MCMC_AC_estimate)
        if self.drift_model == '3rd order polynomial' or self.drift_model == '3rd order odds correlated':
            self.joint_kernel_density_obj = cpy.gaussian_kde(self.theta_array, bw_method='silverman')
            self.joint_samples = self.joint_kernel_density_obj.resample(size=n_joint_samples)
            self.third_order_polynom_slope_in_fixed_point()
        elif self.drift_model == 'first order polynomial':
            self.slope_estimates = self.theta_array[:, 1]
        self.X_coupling_density_obj = cpy.gaussian_kde(self.theta_array[self.num_X_drift_params, :],
                                                       bw_method='silverman')
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
        X_coupling_samples = self.X_coupling_density_obj.resample(size=n_X_coupling_samples)
        # print('X coupling term estimates: ', X_coupling_samples)
        X_coupling_level = X_coupling_grid[
            self.X_coupling_density_obj(X_coupling_grid) == np.max(self.X_coupling_density_obj(X_coupling_grid))]
        # print('X coupling level estimate: ', X_coupling_level)
        if X_coupling_level.size == 1:
            self.X_coupling_estimate[0] = X_coupling_level
        else:
            self.X_coupling_estimate[0] = X_coupling_level[0]
        X_coupling_credibility_percentiles = np.percentile(X_coupling_samples,
                                                           [cred_percentiles[0], 100 - cred_percentiles[0]])
        self.X_coupling_estimate[1] = X_coupling_credibility_percentiles[0]
        self.X_coupling_estimate[2] = X_coupling_credibility_percentiles[1]
        X_coupling_credibility_percentiles = np.percentile(X_coupling_samples,
                                                           [cred_percentiles[1], 100 - cred_percentiles[1]])
        self.X_coupling_estimate[3] = X_coupling_credibility_percentiles[0]
        self.X_coupling_estimate[4] = X_coupling_credibility_percentiles[1]
        if self.Y_model == 'Ornstein-Uhlenbeck noise':
            self.OU_kernel_density_obj = cpy.gaussian_kde(self.theta_array[-1, :], bw_method='silverman')
            self.OU_param_samples = self.OU_kernel_density_obj.resample(size=n_OU_param_samples)
            # print('OU param estimates: ', self.OU_param_samples)
            OU_param = OU_grid[self.OU_kernel_density_obj(OU_grid) == np.max(self.OU_kernel_density_obj(OU_grid))]
            # print('OU param estimate: ', OU_param)
            if OU_param.size == 1:
                self.OU_param_estimate[0] = OU_param
            else:
                self.OU_param_estimate[0] = OU_param[0]
            OU_param_credibility_percentiles = np.percentile(self.OU_param_samples,
                                                             [cred_percentiles[0], 100 - cred_percentiles[0]])
            self.OU_param_estimate[1] = OU_param_credibility_percentiles[0]
            self.OU_param_estimate[2] = OU_param_credibility_percentiles[1]
            OU_param_credibility_percentiles = np.percentile(self.OU_param_samples,
                                                             [cred_percentiles[1], 100 - cred_percentiles[1]])
            self.OU_param_estimate[3] = OU_param_credibility_percentiles[0]
            self.OU_param_estimate[4] = OU_param_credibility_percentiles[1]
            self.noise_estimates = self.theta_array[self.num_X_drift_params, :] * 1. / self.theta_array[-1, :] * self.dt
            self.noise_kernel_density_obj = cpy.gaussian_kde(self.noise_estimates, bw_method='silverman')
            self.noise_samples = self.noise_kernel_density_obj.resample(size=n_noise_samples)
            noise_level = noise_grid[
                self.noise_kernel_density_obj(noise_grid) == np.max(self.noise_kernel_density_obj(noise_grid))]
            if noise_level.size == 1:
                self.noise_level_estimate[0] = noise_level
            else:
                self.noise_level_estimate[0] = noise_level[0]
            noise_cred_percentiles = np.percentile(self.noise_samples, [cred_percentiles[0], 100 - cred_percentiles[0]])
            self.noise_level_estimate[1] = noise_cred_percentiles[0]
            self.noise_level_estimate[2] = noise_cred_percentiles[1]
            noise_cred_percentiles = np.percentile(self.noise_samples, [cred_percentiles[1], 100 - cred_percentiles[1]])
            self.noise_level_estimate[3] = noise_cred_percentiles[0]
            self.noise_level_estimate[4] = noise_cred_percentiles[1]

    def perform_resilience_scan(self, window_size, window_shift, slope_grid, noise_grid, OU_grid, X_coupling_grid,
                                nwalkers=50, nsteps=10000, nburn=200, n_joint_samples=50000,
                                n_slope_samples=50000, n_noise_samples=50000, n_OU_param_samples=50000,
                                n_X_coupling_samples=50000, cred_percentiles=np.array([16, 1]),
                                print_AC_tau=False, ignore_AC_error=False, thinning_by=60,
                                print_progress=False, print_details=False, print_time_scale_info = False,
                                slope_save_name='default_save_slopes',
                                noise_level_save_name='default_save_noise', save=True,
                                create_animation=False, ani_save_name='default_animation_name',
                                animation_title='', mark_critical_point=None,
                                mark_noise_level=None, detrending_per_window = None,
                                gauss_filter_mode='reflect',
                                gauss_filter_sigma=6, gauss_filter_order=0, gauss_filter_cval=0.0,
                                gauss_filter_truncate=4.0, plot_detrending=False,
                                MCMC_parallelization_method=None,
                                num_processes=None, num_chop_chains=None, MCMC_AC_estimate = 'standard'):
        '''
        Performs an automated window scan with defined `window_shift` over the whole time series. In each
        window the drift slope, noise level, X coupling strength and Ornstein-Uhlenbeck (OU) parameter estimates with
        corresponding credibility bands are computed and saved in the `slope_storage`,`noise_level_storage`,`X_coupling_storage`
        and `OU_param_storage`. It can also be used to create an
        animation of the sliding window approach plotting the time series, the moving window, and the time evolution of
        the drift slope estimates :math:`\hat{\zeta}`, the noise level :math:`\hat{\psi}` and the noise kernel density estimate.
        The start indices of the shifted windows are also saved in order to facilitate customized plots.

        :param window_size: Time window size.
        :type window_size: int
        :param window_shift: The rolling time window is shifted about `window_shift` data points.
        :type window_shift: int
        :param slope_grid: Array on which the drift slope kernel density estimate is evaluated.
        :type slope_grid: One-dimensional numpy array of floats.
        :param noise_grid: Array on which the noise level kernel density estimate is evaluated.
        :type noise_grid: One-dimensional numpy array of floats.
        :param OU_grid: Array on which the Ornstein Uhlenbeck (OU) parameter's kernel density estimate is evaluated.
        :type OU_grid: One-dimensional numpy array of floats.
        :param X_coupling_grid: Array on which the X coupling strength kernel density estimate is evaluated.
        :type X_coupling_grid: One-dimensional numpy array of floats.
        :param nwalkers: Number of walkers that are initialized for the MCMC sampling via the package `emcee`.
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
                        marginal probability distribution of the diffusion model is chosen constant.
                        Default is 50000.
        :type n_noise_samples: int
        :param n_X_coupling_samples: Number of X coupling samples that are drawn from the estimated posterior
                        probability. Default is 50000.
        :type n_X_coupling_samples: int
        :param n_OU_param_samples: Number of OU parameter samples that are drawn from the estimated posterior
                        probability. Default is 50000.
        :type n_OU_param_samples: int
        :param cred_percentiles: Two entries to define the percentiles of the calculated credibility bands
                        of the estimated parameters. It is stored in the attribute `credibility_bands`.
                        Default is `numpy.array([16,1])`.
        :type cred_percentiles: One-dimensional numpy array of integers.
        :param print_AC_tau: If `True` the estimated autocorrelation lengths of the Markov chains is shown.
                        The maximum length is used for thinning of the chains. Default is `False`.
        :type prind_AC_tau: Boolean
        :param ignore_AC_error: If `True` the autocorrelation lengths of the Markov chains is not estimated
                        and thus, it is not checked whether the chains are too short to give an reliable
                        autocorrelation estimate. This avoids error interruption of the procedure, but
                        can lead to unreliable results if the chains are too short. The option should
                        be chosen in order to save computation time, e.g. when debugging your code
                        with short Markov chains. Default is `False`.
        :type ignore_AC_error: Boolean
        :param thinning_by: If `ignore_AC_error = True` the chains are thinned by `thinning_by`. Every
                        `thinning_by`-th data point is stored. Default is 60.
        :type thinning_by: int
        :param print_progress: If `True` the progress of the MCMC sampling is shown. Default is `False`.
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
        :param create_animation: If `True` an automated animation of the time evolution of the drift slope
                        estimate :math:\hat{\zeta}`, the noise level :math:`\hat{\sigma}` and the
                        noise kernel density is shown together with the time series and rolling
                        windows. Default is `False`.

                        .. warning::

                            The implementation is same as in ``LangevinEstimation``. It might work to generate animations
                            with the option, but it is not tested. If errors occur, they are almost certainly bugs for that
                            reason.

        :type create_animation: Boolean
        :param ani_save_name: If `create_animation = True` the animation is saved in a .mp4 file with this name.
                        Default is `default_animation_name`.
        :type ani_save_name: string
        :param animation_title: A title can be given to the animation.
        :type animation_title: string
        :param mark_critical_point: A red dotted vertical line is shown at time `mark_critical_point`.
                        Default is `None`.
        :type mark_critical_point: float
        :param mark_noise_level: A green dotted line is shown at the noise level `mark_noise_level` in the
                        noise kernel density plot and the time evolution plot of the noise level.
                        Default is `None`.
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
        :param print_time_scale_info: If `True`, control values are plotted during each prior evaluation. First, the
                        fast time scale estimate times the ``time_scale_separation_factor`` is printed. Second, the slow
                        time scale estimate and third, a check up whether the desired time scale separation is fulfilled
                        are printed.
        :type print_time_scale_info: bool
        :param MCMC_parallelization_method: Default is `None`. If `None` the basic serial MCMC computation is performed. If
                        `MCMC_parallelization_method = 'multiprocessing'`, a multiprocessing pool with `num_processes`
                        is used to accelerate MCMC sampling. If `MCMC_parallelization_method = 'chop_chain'` is used, the
                        total length of the desired Markov chain is divided into `'chop_chain'` parts each of which is
                        sampled in parallel and joined together in the end.
        :type MCMC_parallelization_method: str
        :param num_processes: Default is ``None``. If ``'half'``, almost half of the CPU kernels are used. If  ``'all'``, all CPU kernels
                        are used. If integer number, the defined number of CPU kernels is used for multiprocessing.
        :type num_processes: str or int
        :param num_chop_chains: Number by which the total length of the Markov chain is divided. Each slice is sampled in parallel and
                        joined together in the end of the calculations.
        :type num_chop_chains: int
        :param MCMC_AC_estimate: If default `'standard'` is used, emcee's ``.get_autocorr_time()`` is applied to estimate the
                        sampled Markov chain's autocorrelation length for thinning. In some cases the estimation procedure requires longer chains
                        than you can run and does not converge at all. In such situations you can try to estimate the autocorrelation length with
                        parametric models following the suggestions of the `emcee documentation <https://emcee.readthedocs.io/en/stable/tutorials/autocorr/>`_
                        via ``MCMC_AC_estimate = 'alternative'``.
        :type MCMC_AC_estimate: str
        '''

        self.window_size = window_size
        self.window_shift = window_shift
        self.data_window = np.zeros(window_size)
        self.time_window = np.zeros(window_size)
        self.increments = np.zeros(window_size - 1)
        self.nwalkers = nwalkers
        self.nsteps = nsteps
        self.nburn = nburn
        if self.data_size - self.window_size != 0:
            self.loop_range = np.arange(0, self.data_size - self.window_size, self.window_shift)
        else:
            self.loop_range = np.array([0])
        self.slope_storage = np.zeros((5, self.loop_range.size))
        self.noise_level_storage = np.zeros((5, self.loop_range.size))
        self.OU_param_storage = np.zeros((5, self.loop_range.size))
        self.X_coupling_storage = np.zeros((5, self.loop_range.size))
        if create_animation == False:
            for i in range(self.loop_range.size):
                if print_progress:
                    print('Calculate resilience for window ' + str(i + 1) + ' of ' + str(self.loop_range.size) + '.')
                self.window_shift = self.loop_range[i]
                self.calc_drift_slope_and_noise(slope_grid, noise_grid, OU_grid, X_coupling_grid, nwalkers, nsteps,
                                                nburn,
                                                n_joint_samples, n_slope_samples, n_noise_samples,
                                                n_OU_param_samples, n_X_coupling_samples,
                                                cred_percentiles, print_AC_tau, print_time_scale_info,
                                                ignore_AC_error, thinning_by, print_progress, detrending_per_window,
                                                gauss_filter_mode, gauss_filter_sigma, gauss_filter_order, gauss_filter_cval,
                                                gauss_filter_truncate, plot_detrending, print_details, MCMC_parallelization_method,
                                                num_processes, num_chop_chains, MCMC_AC_estimate)
                self.slope_storage[:, i] = self.drift_slope
                self.noise_level_storage[:, i] = self.noise_level_estimate
                if self.Y_model == 'Ornstein-Uhlenbeck noise':
                    self.OU_param_storage[:, i] = self.OU_param_estimate
                    self.X_coupling_storage[:, i] = self.X_coupling_estimate
                else:
                    print('ERROR: Y model is unknown.')

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
                                repeat=False,
                                fargs=[slope_grid, noise_grid, OU_grid, X_coupling_grid, nwalkers, nsteps, nburn,
                                       n_joint_samples, n_slope_samples, n_noise_samples, n_OU_param_samples,
                                       n_X_coupling_samples, cred_percentiles, print_AC_tau, ignore_AC_error,
                                       thinning_by, print_progress, print_details, MCMC_parallelization_method,
                                       num_processes, num_chop_chains])
            ani.save(ani_save_name + '.mp4', writer=writer)  # , writer = writer
        if save:
            np.save(slope_save_name + '.npy', self.slope_storage)
            np.save(noise_level_save_name + '.npy', self.noise_level_storage)

    def animation(self, i, slope_grid, noise_grid, OU_grid, X_coupling_grid, nwalkers, nsteps, nburn, n_joint_samples,
                  n_slope_samples, n_noise_samples, n_OU_param_samples, n_X_coupling_samples, cred_percentiles,
                  print_AC_tau, ignore_AC_error, thinning_by, print_progress, print_details,
                  MCMC_parallelization_method, num_processes, num_chop_chains):
        '''
        Function that is called iteratively by the `matplotlib.animation.FuncAnimation(...)` tool
        to generate an animation of the rolling window scan of a time series. The time series, the rolling
        windows, the time evolution of the drift slope estimate :math:`\hat{\zeta}` and the noise level
        estimate :math:`\hat{\psi}` and its posterior probality density is shown in the animation.
        '''
        global animation_count
        if print_progress:
            print('Calculate resilience for window ' + str(animation_count + 1) + ' of ' + str(
                self.loop_range.size) + '.')
        self.window_shift = i
        self.calc_drift_slope_and_noise(slope_grid, noise_grid, OU_grid, X_coupling_grid, nwalkers, nsteps, nburn,
                                        n_joint_samples, n_slope_samples, n_noise_samples,
                                        n_OU_param_samples, n_X_coupling_samples,
                                        cred_percentiles, print_AC_tau,
                                        ignore_AC_error, thinning_by, print_progress,
                                        print_details, MCMC_parallelization_method, num_processes, num_chop_chains)
        self.slope_storage[:, animation_count] = self.drift_slope
        self.noise_level_storage[:, animation_count] = self.noise_level_estimate
        self.OU_param_storage[:, animation_count] = self.OU_param_estimate
        self.X_coupling_storage[:, animation_count] = self.X_coupling_estimate
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

    def perform_MAP_resilience_scan(self, window_size, window_shift,
                                    cred_percentiles=np.array([16, 1]), error_propagation='summary statistics',
                                    summary_window_size = 10, sigma_multiples = np.array([1,3]),
                                    print_progress=True, print_details=False,
                                    slope_save_name='default_save_slopes',
                                    noise_level_save_name='default_save_noise', save=True,
                                    create_plot=False, ani_save_name='default_animation_name',
                                    animation_title='', mark_critical_point=None,
                                    mark_noise_level=None, print_time_scale_info = False, print_hint = True, fastMAPflag = False):
        """
        Performs an automated MAP window scan with defined `window_shift` over the whole time series. This might be
        the first approach for a non-Markovian estimation, since the MCMC computations are time-consuming. In each
        window the drift slope, noise level, X coupling strength and OU parameter estimates with corresponding
        credibility bands are computed and saved in the `slope_storage` and the `noise_level_storage`.

        :param window_size: Time window size.
        :type window_size: int
        :param window_shift: The rolling time window is shifted about `window_shift` data points.
        :type window_shift: int
        :param cred_percentiles: Two entries to define the percentiles of the calculated credibility bands
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
                        correlations in the model parameters. Furthermore, the ``'uncorrelated Gaussian'`` option keeps
                        the slightly asymmetric errors proposed by Wilks' theorem for the marginal parameter distributions.
                        This small formal incorrectness is to maintain information about asymmetry in the estimates in a
                        first guess. The ``cred_percentiles`` are fixed to ``cred_percentiles = numpy.array([5,1])`` to
                        transform the half confidence bands under the assumption of Gaussian distributions by the factors
                        1.96 and 2.5525, respectively. The propagated result is transformed back.

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
        self.OU_param_storage = np.zeros((5, self.loop_range.size))
        self.X_coupling_storage = np.zeros((5, self.loop_range.size))
        if create_plot == False:
            for i in range(self.loop_range.size):
                if print_progress:
                    print(
                        'Calculate MAP resilience for window ' + str(i + 1) + ' of ' + str(self.loop_range.size) + '.')
                self.window_shift = self.loop_range[i]
                self._calc_MAP_resilience(cred_percentiles, error_propagation, print_details, print_time_scale_info)
                self.slope_storage[:, i] = self.drift_slope
                self.noise_level_storage[:, i] = self.noise_level_estimate
                if self.Y_model == 'Ornstein-Uhlenbeck noise':
                    self.OU_param_storage[:, i] = self.OU_param_estimate
                    self.X_coupling_storage[:, i] = self.X_coupling_estimate
                else:
                    print('ERROR: Y model is unknown.')
        if error_propagation == 'summary statistics' and not fastMAPflag:
            self.slope_storage = summary_statistics_helper(self.slope_storage, summary_window_size, sigma_multiples)
            self.noise_level_storage = summary_statistics_helper(self.noise_level_storage, summary_window_size, sigma_multiples)
        elif create_plot:
            print('The MAP plot feature is not implemented yet.')
        if save:
            np.save(slope_save_name, self.slope_storage)
            np.save(noise_level_save_name, self.noise_level_storage)

    def _calc_MAP_resilience(self, cred_percentiles, error_propagation, print_details, print_time_scale_info):
        """
        Helper function that computes the MAP estimates of drift slope :math:`\hat{\zeta}`, noise level
        :math:`\hat{\psi}`, X coupling strength and OU parameter with corresponding confidence bands created with
        Wilks' theorem and Gaussian propagation of uncertainty for a given ``window_size`` and ``window_shift`` stored
        in the corresponding attributes.

        :param cred_percentiles: Two entries to define the percentiles of the calculated credibility bands
                        of the estimated parameters. It is stored in the attribute `credibility_bands`.
                        Default is `numpy.array([16,1])`.

        :type cred_percentiles: One-dimensional numpy array of integers.
        :param symmetric_error: If ``True``, the highest uncertainty of each estimated ``MAP_theta`` is used
                        to define symmetric confidence bands that are used for the Gaussian error progagation
                        results for the drift slope that are stored in ``MAP_slope_margin``. If ``False``,
                        the asymmetric confidence bands of the ``MAP_theta`` are used.

        :type symmetric_error: Boolean
        :param print_details: If `True` a more detailed print output in the case the binning procedure is
                        provided. Default is `False`.

        :type print_details: Boolean
        :param print_time_scale_info: If `True`, control values are plotted during each prior evaluation. First, the
                        fast time scale estimate times the ``time_scale_separation_factor`` is printed. Second, the slow
                        time scale estimate and third, a check up whether the desired time scale separation is fulfilled
                        are printed.
        :type print_time_scale_info: bool
        """
        self.data_window = np.roll(self.data, shift=- self.window_shift)[:self.window_size]
        self.time_window = np.roll(self.time, shift=- self.window_shift)[:self.window_size]
        self.credibility_bands = cred_percentiles
        self._prepare_data(printbool=print_details)
        self._compute_MAP_estimates(printbool=print_details, print_time_scale_info=print_time_scale_info)
        self.noise_level_estimate = self.MAP_theta[-2, 0] * 1./self.MAP_theta[-1, 0] * self.dt
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
        if self.Y_model == 'Ornstein-Uhlenbeck noise':
            self.OU_param_estimate = self.MAP_theta[-1, :]
            self.X_coupling_estimate = self.MAP_theta[-2, :]
        if error_propagation == 'error bound':
            self._compute_slope_error_margin(printbool=print_details)
            self._compute_noise_error_margin(printbool=print_details)
        elif error_propagation == 'uncorrelated Gaussian':
            self._compute_slope_errors(printbool=print_details)
            self._compute_noise_errors(printbool=print_details)

    def _compute_noise_errors(self, printbool):
        """
        Helper function that determines the asymmetric uncertainty bands of the MAP noise estimates via
        Gaussian propagation of uncertainties. The underlying marginal uncertainties of the ``MAP_theta``
        are estimated via Wilks' theorem.

        :param printbool: Determines whether a detailed output is printed or not.
        :type printbool: Boolean
        """

        if printbool:
            print('______________________________________________')
            print('Compute asymmetric slope confidence intervals.')
            print('______________________________________________')
        self.MAP_noise_errors = np.zeros(4)
        theta_error_sigLevel1 = np.zeros((self.ndim, 2))
        theta_error_sigLevel2 = np.zeros((self.ndim, 2))
        theta_error_sigLevel1[:, 0] = self.MAP_CI[:, 0]/1.96
        theta_error_sigLevel1[:, 1] = self.MAP_CI[:, 1]/1.96

        theta_error_sigLevel2[:, 0] = self.MAP_CI[:, 2]/2.5525
        theta_error_sigLevel2[:, 1] = self.MAP_CI[:, 3]/2.5525

        if self.Y_model == 'Ornstein-Uhlenbeck noise':
            param1_error = 1./self.MAP_theta[5, 0] * self.dt * theta_error_sigLevel1[4, 0]
            param2_error = -self.MAP_theta[4, 0] * 1./self.MAP_theta[5,0]**2 * self.dt * theta_error_sigLevel1[5, 0]
            self.MAP_noise_errors[0] = np.sqrt(param1_error**2 + param2_error**2) * 1.96
            param1_error = 1./self.MAP_theta[5, 1] * self.dt * theta_error_sigLevel1[4, 1]
            param2_error = -self.MAP_theta[4, 1] * 1./self.MAP_theta[5,0]**2 * self.dt * theta_error_sigLevel1[5, 1]
            self.MAP_noise_errors[1] = np.sqrt(param1_error**2 + param2_error**2) * 1.96
            param1_error = 1./self.MAP_theta[5, 0] * self.dt * theta_error_sigLevel2[4, 0]
            param2_error = -self.MAP_theta[4, 0] * 1./self.MAP_theta[5,0]**2 * self.dt * theta_error_sigLevel2[5, 0]
            self.MAP_noise_errors[2] = np.sqrt(param1_error**2 + param2_error**2) * 2.5525
            param1_error = 1./self.MAP_theta[5, 1] * self.dt * theta_error_sigLevel2[4, 1]
            param2_error = -self.MAP_theta[4, 1] * 1./self.MAP_theta[5,0]**2 * self.dt * theta_error_sigLevel2[5, 1]
            self.MAP_noise_errors[3] = np.sqrt(param1_error**2 + param2_error**2) * 2.5525
        else:
            print('ERROR: Y_model is not known.')

        self.noise_level_estimate[1] = self.noise_level_estimate[0] - self.MAP_noise_errors[0]
        self.noise_level_estimate[2] = self.noise_level_estimate[0] + self.MAP_noise_errors[1]
        self.noise_level_estimate[3] = self.noise_level_estimate[0] - self.MAP_noise_errors[2]
        self.noise_level_estimate[4] = self.noise_level_estimate[0] + self.MAP_noise_errors[3]
        if printbool:
            print('MAP_noise_errors: ', self.MAP_noise_errors)
            print('noise_level_estimate array: ', self.noise_level_estimate)
            print('_____')
            print('Done!')
            print('_____')

    def _compute_noise_error_margin(self, printbool):
        """
        Helper function that determines the symmetric uncertainty bands of the MAP slope estimates via
        Gaussian propagation of uncertainties. The underlying marginal uncertainties of the ``MAP_theta``
        are estimated via Wilks' theorem and the higher deviation of each estimated ``MAP_theta`` is used
        to define symmetric uncertainty intervals.

        :param printbool: Determines whether a detailed output is printed or not.
        :type printbool: Boolean
        """

        if printbool:
            print('______________________________________________')
            print('Compute symmetric slope error margins.')
            print('______________________________________________')
        self.MAP_noise_margin = np.zeros(2)
        theta_error_sigLevel1 = np.zeros((self.ndim, 2))
        theta_error_sigLevel2 = np.zeros((self.ndim, 2))

        theta_error_sigLevel1[:, 0] = self.MAP_CI[:, 0]
        theta_error_sigLevel1[:, 1] = self.MAP_CI[:, 1]
        theta_margin1 = np.maximum(np.absolute(theta_error_sigLevel1[:, 0]), theta_error_sigLevel1[:, 1])

        theta_error_sigLevel2[:, 0] = self.MAP_CI[:, 2]
        theta_error_sigLevel2[:, 1] = self.MAP_CI[:, 3]
        theta_margin2 = np.maximum(np.absolute(theta_error_sigLevel2[:, 0]), theta_error_sigLevel2[:, 1])

        if self.Y_model == 'Ornstein-Uhlenbeck noise':
            param1_error = np.abs(1./self.MAP_theta[5, 0] * self.dt) * theta_margin1[4]
            param2_error = np.abs(-self.MAP_theta[4, 0] * 1./self.MAP_theta[5,0]**2 * self.dt) * theta_margin1[5]
            self.MAP_noise_margin[0] = param1_error + param2_error
            param1_error = np.abs(1./self.MAP_theta[5, 1] * self.dt) * theta_margin2[4]
            param2_error = np.abs(-self.MAP_theta[4, 1] * 1./self.MAP_theta[5,0]**2 * self.dt) * theta_margin2[5]
            self.MAP_noise_margin[1] = param1_error + param2_error

        else:
            print('ERROR: Y_model is not known.')
        self.noise_level_estimate[1] = self.noise_level_estimate[0] - self.MAP_noise_margin[0]
        self.noise_level_estimate[2] = self.noise_level_estimate[0] + self.MAP_noise_margin[0]
        self.noise_level_estimate[3] = self.noise_level_estimate[0] - self.MAP_noise_margin[1]
        self.noise_level_estimate[4] = self.noise_level_estimate[0] + self.MAP_noise_margin[1]
        if printbool:
            print('MAP_slope_margin: ', self.MAP_slope_margin)
            print('drift_slope array: ', self.drift_slope)
            print('_____')
            print('Done!')
            print('_____')

    @staticmethod
    def auto_window(taus, c):
        """
        Helper function to perform parametric Markov chain autocorrelation estimation following the suggestions
        in `emcee's documentation <https://emcee.readthedocs.io/en/stable/tutorials/autocorr/>`_ .
        """
        m = np.arange(len(taus)) < c * taus
        if np.any(m):
            return np.argmin(m)
        return len(taus) - 1

    @staticmethod
    def autocorr_ml(y, thin=1, c=5.0):
        """
        Helper function to perform parametric Markov chain autocorrelation estimation following the suggestions
        in `emcee's documentation <https://emcee.readthedocs.io/en/stable/tutorials/autocorr/>`_ .
        """
        init = __class__.autocorr_new(y, c=c)
        z = y[:, ::thin]
        N = z.shape[1]

        # Build the GP model
        tau = max(1.0, init / thin)

        kernel = terms.RealTerm(
            np.log(0.9 * np.var(z)),
            -np.log(tau),
            bounds=[(-50, 50), (-np.log(N), 50)],
        )

        kernel += terms.RealTerm(
            np.log(0.1 * np.var(z)),
            -np.log(0.5 * tau),
            bounds=[(-50, 50), (-np.log(N), 50)],
        )
        gp = celerite.GP(kernel, mean=np.mean(z))
        gp.compute(np.arange(z.shape[1]))

        # Define the objective
        def nll(p):
            # Update the GP model
            gp.set_parameter_vector(p)

            # Loop over the chains and compute likelihoods
            v, g = zip(*(gp.grad_log_likelihood(z0, quiet=True) for z0 in z))

            # Combine the datasets
            return -np.sum(v), -np.sum(g, axis=0)

        # Optimize the model
        p0 = gp.get_parameter_vector()
        bounds = gp.get_parameter_bounds()
        soln = optimize.minimize(nll, p0, jac=True, bounds=bounds)
        gp.set_parameter_vector(soln.x)

        # Compute the maximum likelihood tau
        a, c = kernel.coefficients[:2]
        tau = thin * 2 * np.sum(a / c) / np.sum(a)
        return tau

    @staticmethod
    def autocorr_new(y, c=5.0):
        """
        Helper function to perform parametric Markov chain autocorrelation estimation following the suggestions
        in `emcee's documentation <https://emcee.readthedocs.io/en/stable/tutorials/autocorr/>`_ .
        """
        f = np.zeros(y.shape[1])
        for yy in y:
            f += __class__.autocorr_func_1d(yy)
        f /= len(y)
        taus = 2.0 * np.cumsum(f) - 1.0
        window = __class__.auto_window(taus, c)
        return taus[window]

    @staticmethod
    def autocorr_func_1d(x, norm=True):
        """
        Helper function to perform parametric Markov chain autocorrelation estimation following the suggestions
        in `emcee's documentation <https://emcee.readthedocs.io/en/stable/tutorials/autocorr/>`_ .
        """
        x = np.atleast_1d(x)
        if len(x.shape) != 1:
            raise ValueError("invalid dimensions for 1D autocorrelation function")
        n = __class__.next_pow_two(len(x))

        # Compute the FFT and then (from that) the auto-correlation function
        f = np.fft.fft(x - np.mean(x), n=2 * n)
        acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
        acf /= 4 * n

        # Optionally normalize
        if norm:
            acf /= acf[0]

        return acf

    @staticmethod
    def next_pow_two(n):
        """
        Helper function to perform parametric Markov chain autocorrelation estimation following the suggestions
        in `emcee's documentation <https://emcee.readthedocs.io/en/stable/tutorials/autocorr/>`_ .
        """
        i = 1
        while i < n:
            i = i << 1
        return i