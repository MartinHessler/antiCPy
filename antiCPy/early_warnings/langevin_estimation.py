import numpy as np
import matplotlib
import matplotlib.pylab as plt
from matplotlib.animation import FuncAnimation
from scipy import optimize
import scipy.stats as cpy
import numba
import emcee


class LangevinEstimation:
    '''
    The ``Langevin_estimation`` class includes tools to estimate a polynomial Langevin equation with various
    drift and diffusion terms and provides the drift slope :math:`\zeta` as a resilience measure.

    :param data: A one dimensional numpy array containing the times series to analyse.
    :type data: One dimensional numpy array of floats
    :param time: A one dimensional numpy array containing the time samples of the time series data.
    :type time: One dimensional numpy array of floats
    :param drift_model: Defines the drift model. Default is ``'3rd order polynomial'``.
                        Additionally, a ``'first order polynomial'`` can be chosen.
    :type drift_model: str
    :param drift_model: Defines the diffusion model. Default is ``'constant'``.
                        Additionally, a ``'first order polynomial'`` can be chosen.
    :type drift_model: str
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
    '''

    def __init__(self, data, time, drift_model='3rd order polynomial', diffusion_model='constant',
                 prior_type='Non-informative linear and Gaussian Prior', prior_range=None, scales=None):
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
        if scales == None and prior_type == 'Non-informative linear and Gaussian Prior':
            self.scales = np.array([4, 8])
        else:
            self.scales = scales
        self.prior_type = prior_type
        self.theta = np.zeros(self.ndim)
        self.data = data
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

    def third_order_polynom_slope_in_fixed_point(self):
        '''
        Calculate the slope_estimates of a third order polynomial drift function with joint_samples of the
        posterior distribution around a fixed point estimate given as the data mean of the current window.
        '''

        self.fixed_point_estimate = np.mean(self.data_window)
        self.slope_estimates = self.joint_samples[1, :] + 2 * self.joint_samples[2,
                                                              :] * self.fixed_point_estimate + 3 * self.joint_samples[3,
                                                                                                   :] * self.fixed_point_estimate ** 2

    @staticmethod
    @numba.jit(nopython=True)
    def calc_D1(theta, data, drift_model):
        '''
        Returns the drift parameterized by a first or third order polynomial for input data.
        Static function in order to speed up computation time via ``numba.jit(nopython = True)``.
        '''

        if drift_model == '3rd order polynomial':
            return theta[0] + theta[1] * data + theta[2] * data ** 2 + theta[3] * data ** 3
        elif drift_model == 'first order polynomial':
            return theta[0] + theta[1] * data

    def D1(self):
        '''
        Calls the static ``calc_D1`` function and returns its value.
        '''

        return self.calc_D1(self.theta, self.data_window[:-1], self.drift_model)

    @staticmethod
    @numba.jit(nopython=True)
    def calc_D2(theta, data, diffusion_model):
        '''
        Returns the diffusion parameterized by a constant or first order polynomial for input data.
        Static function in order to speed up computation time via ``numba.jit(nopython = True)``.
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

    def log_prior(self):
        '''
        Returns the logarithmic prior probability for a given set of parameters based on the short time
        propagator depending on the drift and diffusion models and the given ``prior_range`` of the
        parameters ``theta``. A flat prior for the parameters apart from restriction to positive constant
        diffusion and a non-informative prior for linear parts and Gaussian priors for higher orders are
        implemented. The standard deviation of the Gaussian priors is given by the ``scales`` variable.
        '''

        if self.drift_model == '3rd order polynomial' and self.diffusion_model == 'constant':
            if self.prior_type == 'Flat Prior':
                if (self.prior_range[0, 0] > self.theta[0] > self.prior_range[0, 1]) and (
                        self.prior_range[1, 0] > self.theta[1] > self.prior_range[1, 1]) and (
                        self.prior_range[2, 0] > self.theta[2] > self.prior_range[2, 1]) and (
                        self.prior_range[3, 0] > self.theta[3] > self.prior_range[3, 1]) and (
                        self.prior_range[4, 0] > self.theta[4] > self.prior_range[4, 1]):
                    return 0
                else:
                    return - np.inf

            elif self.prior_type == 'Non-informative linear and Gaussian Prior':
                if (self.prior_range[0, 0] > self.theta[0] > self.prior_range[0, 1]) and (
                        self.prior_range[1, 0] > self.theta[1] > self.prior_range[1, 1]) and (
                        self.prior_range[2, 0] > self.theta[2] > self.prior_range[2, 1]) and (
                        self.prior_range[3, 0] > self.theta[3] > self.prior_range[3, 1]) and (
                        self.prior_range[4, 0] > self.theta[4] > self.prior_range[4, 1]):
                    # print('Check_prior!')
                    return ((-3. / 2.) * np.log(1 + self.theta[1] ** 2) - np.log(self.theta[4])) + np.log(
                        cpy.norm.pdf(self.theta[2], loc=0, scale=self.scales[0])) + np.log(
                        cpy.norm.pdf(self.theta[3], loc=0, scale=self.scales[1]))
                else:
                    return - np.inf
        elif self.drift_model == 'first order polynomial' and self.diffusion_model == 'constant':
            if self.prior_type == 'Flat Prior':
                if (self.prior_range[0, 0] > self.theta[0] > self.prior_range[0, 1]) and (
                        self.prior_range[1, 0] > self.theta[1] > self.prior_range[1, 1]) and (
                        self.prior_range[2, 0] > self.theta[2] > self.prior_range[2, 1]):
                    return 0
                else:
                    return - np.inf
            elif self.prior_type == 'Non-informative linear and Gaussian Prior':
                if (self.prior_range[0, 0] > self.theta[0] > self.prior_range[0, 1]) and (
                        self.prior_range[1, 0] > self.theta[1] > self.prior_range[1, 1]) and (
                        self.prior_range[2, 0] > self.theta[2] > self.prior_range[2, 1]):
                    return ((-3. / 2.) * np.log(1 + self.theta[1] ** 2) - np.log(self.theta[2]))
                else:
                    return - np.inf

    def log_likelihood(self):
        '''
        Returns the logarithmic likelihood of the data for the given model parametrization.
        '''

        return np.sum(
            -0.5 * np.log(2 * np.pi * self.D2() ** 2 * self.dt) - (self.increments - self.D1() * self.dt) ** 2 / (
                        2. * self.D2() ** 2 * self.dt))

    def log_posterior(self, theta):
        '''
        Returns the logarithmic posterior probability of the data for the given model parametrization.
        '''

        self.theta = theta
        lg_prior = self.log_prior()
        if not np.isfinite(lg_prior):
            return - np.inf
        return lg_prior + self.log_likelihood()

    def neg_log_posterior(self, theta):
        '''
        Returns the negative logarithmic posterior distribution of the data
        for the given model parametrization.
        '''

        self.theta = theta
        return (-1) * self.log_posterior(theta)

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

    def compute_posterior_samples(self, print_AC_tau, ignore_AC_error, thinning_by, print_progress):
        '''
        Compute the ``theta_array`` with :math:`nwalkers \cdot nsteps` Markov Chain Mone Carlo samples.
        If ``ignore_AC_error = False`` the calculation will terminate with error if
        the autocorrelation of the sampled chains is to high compared to the chain length.
        Otherwise the highest autocorrelation length will be used to thin the sampled chains.
        In order to run tests in shorter time you can set ``ignore_AC_error = True`` and
        define a ``thinning_by`` n steps. If ``print_AC_tau = True`` the autocorrelation lengths of the
        sampled chains is printed.
        '''
        print('Calculate posterior samples')
        sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_posterior)
        sampler.run_mcmc(self.starting_guesses, self.nsteps,
                         progress=print_progress)  # if progress = True the remaining sample time is shown
        if ignore_AC_error == False:
            tau = sampler.get_autocorr_time()
        elif ignore_AC_error:
            tau = thinning_by
        if print_AC_tau:
            print('tau: ', tau)
        flat_samples = sampler.get_chain(discard=self.nburn, thin=int(np.max(tau)), flat=True)
        self.theta_array = np.zeros((self.ndim, flat_samples[:, 0].size))
        for i in range(self.ndim):
            self.theta_array[i, :] = np.transpose(flat_samples[:, i])

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
        axs[0, 0].axvline(crit_poi, ls=':', color='r')
        axs[0, 0].set_ylim(np.min(animation_data) - 1, np.max(animation_data) + 1)
        axs[0, 0].set_ylabel(r'variable $x$', fontsize=15)
        axs[0, 0].set_xlabel(r'time $t$', fontsize=15)

        axs[0, 1].set_xlim(noise_gr_animation[0], noise_gr_animation[-1])
        axs[0, 1].set_ylim(0, 100)
        axs[0, 1].axvline(noi_le, ls=':', color='g')
        axs[0, 1].set_ylabel(r'Noise Posterior $P(\hat{\theta_4})$', fontsize=15)
        axs[0, 1].set_xlabel(r'noise level $\hat{\theta_4}$', fontsize=15)

        axs[1, 0].set_xlim(animation_time[0], animation_time[-1])
        axs[1, 0].plot(animation_time, np.zeros(animation_time.size), ls=':', color='r')
        axs[1, 0].axvline(crit_poi, ls=':', color='r')
        axs[1, 0].set_ylabel(r'max posterior slope $\zeta^{\rm max}$', fontsize=15)
        axs[1, 0].set_xlabel(r'time $t$', fontsize=15)

        axs[1, 1].set_xlim(animation_time[0], animation_time[-1])
        axs[1, 1].set_ylim(noise_gr_animation[0], noise_gr_animation[-1])
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
                  n_noise_samples, cred_percentiles, print_AC_tau, ignore_AC_error, thinning_by, print_progress):
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
                                   print_progress)
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
                              cred_percentiles=np.array([16, 1]), print_AC_tau=False,
                              ignore_AC_error=False, thinning_by=60, print_progress=False):
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
        '''

        self.data_window = np.roll(self.data, shift=- self.window_shift)[:self.window_size]
        self.increments = self.data_window[1:] - self.data_window[:-1]
        self.time_window = np.roll(self.time, shift=- self.window_shift)[:self.window_size]
        self.declare_MAP_starting_guesses()
        self.compute_posterior_samples(print_AC_tau=print_AC_tau, ignore_AC_error=ignore_AC_error,
                                       thinning_by=thinning_by, print_progress=print_progress)
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

    def perform_resilience_scan(self, window_size, window_shift, slope_grid, noise_grid,
                                             nwalkers=50, nsteps=10000, nburn=200, n_joint_samples=50000,
                                             n_slope_samples=50000, n_noise_samples=50000,
                                             cred_percentiles=np.array([16, 1]), print_AC_tau=False,
                                             ignore_AC_error=False, thinning_by=60, print_progress=False,
                                             slope_save_name='default_save_slopes',
                                             noise_level_save_name='default_save_noise', save=True,
                                             create_animation=False, ani_save_name='default_animation_name',
                                             animation_title='', mark_critical_point=None,
                                             mark_noise_level=None):
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
        '''
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
                                           ignore_AC_error, thinning_by, print_progress)
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
                                                     ignore_AC_error, thinning_by, print_progress])
            ani.save(ani_save_name + '.mp4', writer=writer)  # , writer = writer
        if save:
            np.save(slope_save_name + '.npy', self.slope_storage)
            np.save(noise_level_save_name + '.npy', self.noise_level_storage)

