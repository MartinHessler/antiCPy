import numpy as np
import multiprocessing as mp

import antiCPy.early_warnings.drift_slope.langevin_estimation
from .summary_statistics_helper import summary_statistics_helper

class RocketFastResilienceEstimation():
    """
    Superclass of the ``LangevinEstimation``, ``BinningLangevinEstimation`` and ``NonMarkovEstimation`` class. Initializes functions
    for a strong parallelisation of the rolling window based MCMC and MAP algorithm by coordinating multiple window calculations over a
    freely eligible number of subprocesses. Instead of parallelizing the MCMC sampling itself, the whole window calculations are parallelized
    with single CPU MCMC sampling. This implementation yields significant speed up of computation times.

    :param antiCPyObject: Internal identifier of the object type.
    :type antiCPyObject: str
    """
    def __init__(self, antiCPyObject = None):
        self.antiCPyObject = antiCPyObject

    def fast_resilience_scan(self, window_size, window_shift, slope_grid, noise_grid, OU_grid = None, X_coupling_grid = None,
                             nwalkers=50, nsteps=10000, nburn=200, n_joint_samples=50000, detrending_per_window = None,
                             n_slope_samples=50000, n_noise_samples=50000, n_OU_param_samples=50000, n_X_coupling_samples=50000,
                             MCMC_AC_estimate='standard', cred_percentiles=np.array([16, 1]), print_AC_tau=False, ignore_AC_error=False,
                             thinning_by=60, print_progress=False, print_details=False, print_time_scale_info = False, slope_save_name='default_save_slopes',
                             noise_level_save_name='default_save_noise', OU_param_save_name = 'default_save_OUparam',
                             X_coupling_save_name = 'default_save_Xcoupling', num_processes=None, save=True):
        """
        The function's structure to use is almost equivalent to ``perform_resilience_scan`` of the ``LangevinEstimation`` and the
        ``NonMarkovEstimation`` class. It initializes a multiprocessing pool to calculate several rolling windows simultaneously.
        """
        self.window_size = window_size
        self.window_shift = window_shift
        self.loop_range = np.arange(0, self.data_size - self.window_size, self.window_shift, dtype=int)
        loop_range_size = self.loop_range.size
        self.slope_storage = mp.RawArray('d', 5 * loop_range_size)
        self.noise_level_storage = mp.RawArray('d', 5 * loop_range_size)
        self.data_window = np.zeros(window_size)
        self.time_window = np.zeros(window_size)
        self.increments = np.zeros(window_size - 1)
        self.nwalkers = nwalkers
        self.nsteps = nsteps
        self.nburn = nburn
        if self.antiCPyObject == 'NonMarkovEstimation':
            self.OU_param_storage = mp.RawArray('d', 5 * loop_range_size)
            self.X_coupling_storage = mp.RawArray('d', 5 * loop_range_size)
            initargs = (self.antiCPyObject, self.data, self.time,
                               self.loop_range, self.window_size, slope_grid, noise_grid, self.nwalkers,
                               self.nsteps, self.nburn, n_joint_samples, n_slope_samples, n_noise_samples,
                               cred_percentiles,  print_AC_tau, ignore_AC_error, thinning_by, print_progress, print_details,
                               self.drift_model,self.diffusion_model, self.prior_type, self.prior_range, self.scales,
                               self.slope_storage, self.noise_level_storage, detrending_per_window, self.OU_param_storage, self.X_coupling_storage,
                               n_OU_param_samples, n_X_coupling_samples, MCMC_AC_estimate, self.Y_model, self.Y_drift_model,
                               self.Y_diffusion_model, self.activate_time_scale_separation_prior, self.slow_process,
                               self.time_scale_separation_factor, self.max_likelihood_starting_guesses, OU_grid, X_coupling_grid,
                               print_time_scale_info)
        elif self.antiCPyObject != 'LangevinEstimation':
            print('ERROR: MCMC based resilience scan is not implemented for BinningLangevinEstimation.')
        else:
            initargs = (self.antiCPyObject, self.data, self.time,
                        self.loop_range, self.window_size, slope_grid, noise_grid, self.nwalkers,
                        self.nsteps, self.nburn, n_joint_samples, n_slope_samples, n_noise_samples,
                        cred_percentiles, print_AC_tau, ignore_AC_error, thinning_by, print_progress, print_details,
                        self.drift_model, self.diffusion_model, self.prior_type, self.prior_range, self.scales,
                        self.slope_storage, self.noise_level_storage, detrending_per_window)
        if num_processes == 'half':
            num_processes = int(mp.cpu_count() / 2.)
        elif num_processes == 'all':
            num_processes = int(mp.cpu_count())
        elif isinstance(num_processes, int):
            if num_processes <= mp.cpu_count():
                pass
            else:
                print('WARNING: The number of processes is greater than the number of cpu cores.')
        else:
            print(
                'ERROR: Number of processes is not properly defined. Choose between `half`, `all` or an integer number.')
        chunksize, extra = divmod(loop_range_size, num_processes * 4)
        if extra:
            chunksize += 1
        with mp.Pool(processes=num_processes, initializer=self.init_parallel_Langevin,
                     initargs=initargs) as pool:
            print('Start parallel processing!')
            pool.map_async(self.execute_window, [m for m in range(loop_range_size)],
                           error_callback=custom_error_callback, chunksize=chunksize)
            pool.close()
            pool.join()
            print('Parallel processing finished!')
            self.slope_storage = np.frombuffer(self.slope_storage).reshape((5, loop_range_size))
            self.noise_level_storage = np.frombuffer(self.noise_level_storage).reshape((5, loop_range_size))
            if self.antiCPyObject == 'NonMarkovEstimation':
                self.OU_param_storage = np.frombuffer(self.OU_param_storage).reshape((5,loop_range_size))
                self.X_coupling_storage = np.frombuffer(self.X_coupling_storage).reshape((5, loop_range_size))
            if save:
                np.save(slope_save_name + '.npy', self.slope_storage)
                np.save(noise_level_save_name + '.npy', self.noise_level_storage)
                if self.antiCPyObject == 'NonMarkovEstimation':
                    np.save(OU_param_save_name + '.npy', self.OU_param_storage)
                    np.save(X_coupling_save_name + '.npy', self.X_coupling_storage)

    @staticmethod
    def init_parallel_Langevin(antiCPyObject, data, time, loop_range, window_size, slope_grid, noise_grid, nwalkers,
                               nsteps, nburn, n_joint_samples, n_slope_samples, n_noise_samples, cred_percentiles,
                               print_AC_tau, ignore_AC_error, thinning_by, print_progress, print_details, drift_model,
                               diffusion_model, prior_type, prior_range,scales, slope_storage_connector, noise_storage_connector,
                               detrending_per_window,OU_param_storage_connector = None, X_coupling_storage_connector = None,
                               n_OU_param_samples = None, n_X_coupling_samples = None, MCMC_AC_estimate = None, Y_model = None,
                               Y_drift_model = None, Y_diffusion_model = None, activate_time_scale_separation_prior = False,
                               slow_process = False, time_scale_separation_factor = None, max_likelihood_starting_guesses = None,
                               OU_grid = None, X_coupling_grid = None, print_time_scale_info = None):
        """
        Internal function. Only to initialize the workers of ``RocketFastResilienceEstimation.fast_resilience_scan(...)``.
        """
        global init_dict, shared_memory_dict
        init_dict = {}
        init_dict['antiCPyObject'] = antiCPyObject
        init_dict['data'] = data
        init_dict['time'] = time
        init_dict['loop_range'] = loop_range
        init_dict['loop_range_size'] = loop_range.size
        init_dict['window_size'] = window_size
        init_dict['slope_grid'] = slope_grid
        init_dict['noise_grid'] = noise_grid
        init_dict['nwalkers'] = nwalkers
        init_dict['nsteps'] = nsteps
        init_dict['nburn'] = nburn
        init_dict['n_joint_samples'] = n_joint_samples
        init_dict['n_slope_samples'] = n_slope_samples
        init_dict['n_noise_samples'] = n_noise_samples
        init_dict['cred_percentiles'] = cred_percentiles
        init_dict['print_AC_tau'] = print_AC_tau
        init_dict['ignore_AC_error'] = ignore_AC_error
        init_dict['thinning_by'] = thinning_by
        init_dict['print_progress'] = print_progress
        init_dict['print_details'] = print_details
        init_dict['drift_model'] = drift_model
        init_dict['diffusion_model'] = diffusion_model
        init_dict['prior_type'] = prior_type
        init_dict['prior_range'] = prior_range
        init_dict['scales'] = scales
        init_dict['detrending_per_window'] = detrending_per_window
        if antiCPyObject == 'NonMarkovEstimation':
            init_dict['OU_grid'] = OU_grid
            init_dict['n_OU_param_samples'] = n_OU_param_samples
            init_dict['X_coupling_grid'] = X_coupling_grid
            init_dict['n_X_coupling_samples'] = n_X_coupling_samples
            init_dict['MCMC_AC_estimate'] = MCMC_AC_estimate
            init_dict['Y_model'] = Y_model
            init_dict['Y_drift_model'] = Y_drift_model
            init_dict['Y_diffusion_model'] = Y_diffusion_model
            init_dict['activate_time_scale_separation_prior'] = activate_time_scale_separation_prior
            init_dict['slow_process'] = slow_process
            init_dict['time_scale_separation_factor'] = time_scale_separation_factor
            init_dict['max_likelihood_starting_guesses'] = max_likelihood_starting_guesses
            init_dict['print_time_scale_info'] = print_time_scale_info

        shared_memory_dict = {}
        shared_memory_dict['slope_storage_connector'] = slope_storage_connector
        shared_memory_dict['noise_storage_connector'] = noise_storage_connector
        if antiCPyObject == 'NonMarkovEstimation':
            shared_memory_dict['OU_param_storage_connector'] = OU_param_storage_connector
            shared_memory_dict['X_coupling_storage_connector'] = X_coupling_storage_connector
        elif not antiCPyObject == 'LangevinEstimation':
            print('ERROR: Fast resilience scan is not defined for this antiCPyObject.')

    @staticmethod
    def execute_window(m):
        """
        Internal function. Subprocess Script of the parallel worker of ``RocketFastResilienceEstimation.fast_MAP_resilience_scan(...)``.
        """
        if init_dict['print_progress']:
            print('Execute window ' + str(m + 1) + ' of ' + str(init_dict['loop_range_size']) + ' windows.')
        data = np.roll(init_dict['data'], shift=- init_dict['loop_range'][m])[:init_dict['window_size'] + 1]
        time = np.roll(init_dict['time'], shift=- init_dict['loop_range'][m])[:init_dict['window_size'] + 1]
        slope_storage_helper = np.frombuffer(shared_memory_dict['slope_storage_connector']).reshape(
            (5, init_dict['loop_range_size']))
        noise_storage_helper = np.frombuffer(shared_memory_dict['noise_storage_connector']).reshape(
            (5, init_dict['loop_range_size']))
        if init_dict['antiCPyObject'] == 'NonMarkovEstimation':
            OU_storage_helper = np.frombuffer(shared_memory_dict['OU_param_storage_connector']).reshape(5,init_dict['loop_range_size'])
            X_coupling_storage_helper = np.frombuffer(shared_memory_dict['X_coupling_storage_connector']).reshape(5, init_dict[
                'loop_range_size'])

        if init_dict['antiCPyObject'] == 'LangevinEstimation':
            one_window_helper = antiCPy.early_warnings.drift_slope.langevin_estimation.LangevinEstimation(data, time, init_dict['drift_model'], init_dict['diffusion_model'],
                                                                                                          init_dict['prior_type'],
                                                                                                          init_dict['prior_range'], init_dict['scales'])
            one_window_helper.perform_resilience_scan(init_dict['window_size'], 2, init_dict['slope_grid'],
                                                      init_dict['noise_grid'],
                                                      nwalkers=init_dict['nwalkers'], nsteps=init_dict['nsteps'],
                                                      nburn=init_dict['nburn'],
                                                      n_joint_samples=init_dict['n_joint_samples'],
                                                      n_slope_samples=init_dict['n_slope_samples'],
                                                      n_noise_samples=init_dict['n_noise_samples'],
                                                      detrending_per_window=init_dict['detrending_per_window'],
                                                      cred_percentiles=init_dict['cred_percentiles'],
                                                      print_AC_tau=init_dict['print_AC_tau'],
                                                      ignore_AC_error=init_dict['ignore_AC_error'],
                                                      thinning_by=init_dict['thinning_by'],
                                                      print_progress=False)
        elif init_dict['antiCPyObject'] == 'NonMarkovEstimation':
            one_window_helper = antiCPy.early_warnings.drift_slope.non_markov_estimation.NonMarkovEstimation(data, time,
                                                                                                             X_drift_model=init_dict['drift_model'], Y_model=init_dict['Y_model'],
                                                                                                             Y_drift_model=init_dict['Y_drift_model'], X_coupling_term=init_dict['diffusion_model'],
                                                                                                             Y_diffusion_model=init_dict['Y_diffusion_model'],
                                                                                                             prior_type=init_dict['prior_type'], prior_range=init_dict['prior_range'],
                                                                                                             scales=init_dict['scales'], activate_time_scale_separation_prior=init_dict['activate_time_scale_separation_prior'],
                                                                                                             slow_process=init_dict['slow_process'], time_scale_separation_factor=init_dict['time_scale_separation_factor'],
                                                                                                             max_likelihood_starting_guesses=init_dict['max_likelihood_starting_guesses'])
            one_window_helper.perform_resilience_scan(init_dict['window_size'], 2, init_dict['slope_grid'],
                                                        init_dict['noise_grid'], init_dict['OU_grid'], init_dict['X_coupling_grid'],
                                                        nwalkers=init_dict['nwalkers'], nsteps=init_dict['nsteps'],
                                                        nburn=init_dict['nburn'],
                                                        n_joint_samples=init_dict['n_joint_samples'],
                                                        n_slope_samples=init_dict['n_slope_samples'],
                                                        n_noise_samples=init_dict['n_noise_samples'],
                                                        detrending_per_window=init_dict['detrending_per_window'],
                                                        n_OU_param_samples=init_dict['n_OU_param_samples'],
                                                        n_X_coupling_samples=init_dict['n_X_coupling_samples'],
                                                        cred_percentiles=init_dict['cred_percentiles'],
                                                        print_AC_tau=init_dict['print_AC_tau'],
                                                        print_time_scale_info=init_dict['print_time_scale_info'],
                                                        ignore_AC_error=init_dict['ignore_AC_error'],
                                                        thinning_by=init_dict['thinning_by'],
                                                        print_progress=False, save=False,
                                                        MCMC_AC_estimate = init_dict['MCMC_AC_estimate'])
        slope_storage_helper[:, m] = one_window_helper.slope_storage[:, 0]
        noise_storage_helper[:, m] = one_window_helper.noise_level_storage[:, 0]
        if init_dict['antiCPyObject'] == 'NonMarkovEstimation':
            OU_storage_helper[:, m] = one_window_helper.OU_param_storage[:, 0]
            X_coupling_storage_helper[:, m] = one_window_helper.X_coupling_storage[:, 0]


    def fast_MAP_resilience_scan(self, window_size, window_shift, num_processes = 'half',
                                    cred_percentiles=np.array([16, 1]), error_propagation='summary statistics',
                                    summary_window_size = 10, sigma_multiples = np.array([1,3]), print_progress=True, print_details=False,
                                    slope_save_name='default_save_slopes', noise_level_save_name='default_save_noise',
                                    OU_param_save_name='default_save_OUparam', X_coupling_save_name='default_save_Xcoupling',
                                    save=True, print_time_scale_info = False):
        """
        The function's structure to use is almost equivalent to ``perform_MAP_resilience_scan`` of the ``LangevinEstimation``, ``BinningLangevinEstimation`` and the
        ``NonMarkovEstimation`` class. It initializes a multiprocessing pool to calculate several rolling windows simultaneously.
        """
        if error_propagation == 'uncorrelated Gaussian' and self.drift_model == '3rd order polynomial':
            print('HINT: Fixed cred_percentiles = numpy.array[5,1]) for the uncorrelated Gaussian error propagation '
                  'of the drift slope is used for drift_model == `3rd_order_polynomial`.')
        self.window_size = window_size
        self.window_shift = window_shift
        self.loop_range = np.arange(0, self.data_size - self.window_size, self.window_shift)
        loop_range_size = self.loop_range.size
        self.slope_storage = mp.RawArray('d', 5 * loop_range_size)
        self.noise_level_storage = mp.RawArray('d', 5 * loop_range_size)
        if num_processes == 'half':
            num_processes = int(mp.cpu_count() / 2.)
        elif num_processes == 'all':
            num_processes = int(mp.cpu_count())
        elif isinstance(num_processes, int):
            if num_processes <= mp.cpu_count():
                pass
            else:
                print('WARNING: The number of processes is greater than the number of cpu cores.')
        else:
            print('ERROR: Number of processes is not properly defined. Choose between `half`, `all` or an integer number.')
        chunksize, extra = divmod(loop_range_size, num_processes * 4)
        if extra:
            chunksize += 1
        if self.antiCPyObject == 'LangevinEstimation':
            initargs = (self.antiCPyObject, self.data, self.time,
                               self.loop_range, self.window_size, cred_percentiles,
                               print_details, print_progress, self.drift_model, self.diffusion_model, self.slope_storage,
                               self.noise_level_storage, error_propagation, summary_window_size, sigma_multiples,
                               self.prior_range, self.prior_type, self.scales)
        elif self.antiCPyObject == 'BinningLangevinEstimation':
            initargs = (self.antiCPyObject, self.data, self.time,
                               self.loop_range, self.window_size, cred_percentiles,
                               print_details, print_progress, self.drift_model, self.diffusion_model, self.slope_storage,
                               self.noise_level_storage, error_propagation, summary_window_size, sigma_multiples,
                               self.prior_range, self.prior_type, self.scales, self._bin_num)
        elif self.antiCPyObject == 'NonMarkovEstimation':
            self.OU_param_storage = mp.RawArray('d', 5 * loop_range_size)
            self.X_coupling_storage = mp.RawArray('d', 5 * loop_range_size)
            initargs = (self.antiCPyObject, self.data, self.time,
                               self.loop_range, self.window_size, cred_percentiles,
                               print_details, print_progress, self.drift_model, self.diffusion_model, self.slope_storage,
                               self.noise_level_storage, error_propagation, summary_window_size, sigma_multiples,
                               self.prior_range, self.prior_type, self.scales,
                               None, self.OU_param_storage, self.X_coupling_storage,
                               self.Y_model, self.Y_drift_model,
                               self.Y_diffusion_model, self.activate_time_scale_separation_prior, self.slow_process,
                               self.time_scale_separation_factor, self.max_likelihood_starting_guesses, print_time_scale_info)
        with mp.Pool(processes=num_processes, initializer=self.init_parallel_MAP_Langevin,
                     initargs=initargs) as pool:
            print('Start parallel processing!')
            pool.map_async(self.execute_MAP_window, [m for m in range(loop_range_size)],
                           error_callback=custom_error_callback, chunksize=chunksize)
            pool.close()
            pool.join()
            print('Parallel processing finished!')
        self.slope_storage = np.frombuffer(self.slope_storage).reshape((5, loop_range_size))
        self.noise_level_storage = np.frombuffer(self.noise_level_storage).reshape((5, loop_range_size))
        if error_propagation == 'summary statistics':
            self.slope_storage = summary_statistics_helper(self.slope_storage, summary_window_size, sigma_multiples)
            self.noise_level_storage = summary_statistics_helper(self.noise_level_storage, summary_window_size, sigma_multiples)
        if self.antiCPyObject == 'NonMarkovEstimation':
            self.OU_param_storage = np.frombuffer(self.OU_param_storage).reshape((5,loop_range_size))
            self.X_coupling_storage = np.frombuffer(self.X_coupling_storage).reshape((5, loop_range_size))
        if save:
            np.save(slope_save_name, self.slope_storage)
            np.save(noise_level_save_name, self.noise_level_storage)
            if self.antiCPyObject == 'NonMarkovEstimation':
                np.save(OU_param_save_name + '.npy', self.OU_param_storage)
                np.save(X_coupling_save_name + '.npy', self.X_coupling_storage)

    @staticmethod
    def init_parallel_MAP_Langevin(antiCPyObject, data, time, loop_range, window_size, cred_percentiles, print_details,
                                   print_progress, drift_model, diffusion_model, slope_storage_connector, noise_storage_connector,
                                   error_propagation, summary_window_size, sigma_multiples, prior_range, prior_type, scales, bin_num = None, OU_param_storage_connector = None,
                                   X_coupling_storage_connector = None, Y_model = None, Y_drift_model = None,
                                   Y_diffusion_model = None, activate_time_scale_separation_prior = False, slow_process = False,
                                   time_scale_separation_factor = None, max_likelihood_starting_guesses = None, print_time_scale_info = None):
        """
        Internal function. Only to initialize the workers of ``RocketFastResilienceEstimation.fast_MAP_resilience_scan(...)``.
        """
        global init_dict, shared_memory_dict
        init_dict = {}
        init_dict['antiCPyObject'] = antiCPyObject
        init_dict['data'] = data
        init_dict['time'] = time
        init_dict['loop_range'] = loop_range
        init_dict['loop_range_size'] = loop_range.size
        init_dict['window_size'] = window_size
        init_dict['print_details'] = print_details
        init_dict['print_progress'] = print_progress
        init_dict['drift_model'] = drift_model
        init_dict['diffusion_model'] = diffusion_model
        init_dict['cred_percentiles'] = cred_percentiles
        init_dict['print_details'] = print_details
        init_dict['error_propagation'] = error_propagation
        init_dict['summary_window_size'] = summary_window_size
        init_dict['sigma_multiples'] = sigma_multiples
        init_dict['prior_type'] = prior_type
        init_dict['prior_range'] = prior_range
        init_dict['scales'] = scales
        init_dict['bin_num'] = bin_num
        if antiCPyObject == 'NonMarkovEstimation':
            init_dict['Y_model'] = Y_model
            init_dict['Y_drift_model'] = Y_drift_model
            init_dict['Y_diffusion_model'] = Y_diffusion_model
            init_dict['activate_time_scale_separation_prior'] = activate_time_scale_separation_prior
            init_dict['slow_process'] = slow_process
            init_dict['time_scale_separation_factor'] = time_scale_separation_factor
            init_dict['max_likelihood_starting_guesses'] = max_likelihood_starting_guesses
            init_dict['print_time_scale_info'] = print_time_scale_info

        shared_memory_dict = {}
        shared_memory_dict['slope_storage_connector'] = slope_storage_connector
        shared_memory_dict['noise_storage_connector'] = noise_storage_connector
        if antiCPyObject == 'NonMarkovEstimation':
            shared_memory_dict['OU_param_storage_connector'] = OU_param_storage_connector
            shared_memory_dict['X_coupling_storage_connector'] = X_coupling_storage_connector


    @staticmethod
    def execute_MAP_window(m):
        """
        Internal function. Subprocess Script of the parallel worker of ``RocketFastResilienceEstimation.fast_MAP_resilience_scan(...)``.
        """
        if init_dict['print_progress']:
                print('Calculate MAP resilience for window ' + str(m + 1) + ' of ' + str(init_dict['loop_range_size']) + '.')
        data = np.roll(init_dict['data'], shift=- init_dict['loop_range'][m])[:init_dict['window_size'] + 1]
        time = np.roll(init_dict['time'], shift=- init_dict['loop_range'][m])[:init_dict['window_size'] + 1]
        slope_storage_helper = np.frombuffer(shared_memory_dict['slope_storage_connector']).reshape(
            (5, init_dict['loop_range_size']))
        noise_storage_helper = np.frombuffer(shared_memory_dict['noise_storage_connector']).reshape(
            (5, init_dict['loop_range_size']))
        if init_dict['antiCPyObject'] == 'LangevinEstimation':
            one_window_helper = antiCPy.early_warnings.drift_slope.langevin_estimation.LangevinEstimation(data, time,
                                                                                                          drift_model=init_dict['drift_model'],
                                                                                                          diffusion_model=init_dict['diffusion_model'],
                                                                                                          prior_type=init_dict['prior_type'], prior_range=init_dict['prior_range'],
                                                                                                          scales = init_dict['scales'])
            one_window_helper.perform_MAP_resilience_scan(init_dict['window_size'], window_shift=2,
                                                          cred_percentiles=init_dict['cred_percentiles'],
                                                          error_propagation=init_dict['error_propagation'],
                                                          summary_window_size=init_dict['summary_window_size'],
                                                          sigma_multiples=init_dict['sigma_multiples'],
                                                          print_details=init_dict['print_details'], print_progress=False,
                                                          print_hint=False, fastMAPflag=True)
        elif init_dict['antiCPyObject'] == 'BinningLangevinEstimation':
            one_window_helper = antiCPy.early_warnings.drift_slope.binning_langevin_estimation.BinningLangevinEstimation(data, time,
                                                                                                                         init_dict['bin_num'], drift_model = init_dict['drift_model'],
                                                                                                                         diffusion_model = init_dict['diffusion_model'],
                                                                                                                         prior_type = init_dict['prior_type'],
                                                                                                                         prior_range = init_dict['prior_range'], scales = init_dict['scales'])
            one_window_helper.perform_MAP_resilience_scan(init_dict['window_size'], window_shift=2,
                                                          cred_percentiles=init_dict['cred_percentiles'],
                                                          error_propagation=init_dict['error_propagation'],
                                                          summary_window_size=init_dict['summary_window_size'],
                                                          sigma_multiples=init_dict['sigma_multiples'],
                                                          print_details=init_dict['print_details'], print_progress=False,
                                                          print_hint=False, fastMAPflag=True)
        elif init_dict['antiCPyObject'] == 'NonMarkovEstimation':
            OU_storage_helper = np.frombuffer(shared_memory_dict['OU_param_storage_connector']).reshape(5,init_dict['loop_range_size'])
            X_coupling_storage_helper = np.frombuffer(shared_memory_dict['X_coupling_storage_connector']).reshape(5, init_dict[
                'loop_range_size'])
            one_window_helper = antiCPy.early_warnings.drift_slope.non_markov_estimation.NonMarkovEstimation(data, time,
                                                                                                             X_drift_model=init_dict['drift_model'], Y_model=init_dict['Y_model'],
                                                                                                             Y_drift_model=init_dict['Y_drift_model'], X_coupling_term=init_dict['diffusion_model'],
                                                                                                             Y_diffusion_model=init_dict['Y_diffusion_model'],
                                                                                                             prior_type=init_dict['prior_type'], prior_range=init_dict['prior_range'],
                                                                                                             scales=init_dict['scales'], activate_time_scale_separation_prior=init_dict['activate_time_scale_separation_prior'],
                                                                                                             slow_process=init_dict['slow_process'], time_scale_separation_factor=init_dict['time_scale_separation_factor'],
                                                                                                             max_likelihood_starting_guesses=init_dict['max_likelihood_starting_guesses'])
            one_window_helper.perform_MAP_resilience_scan(init_dict['window_size'], window_shift=2,
                                                          cred_percentiles=init_dict['cred_percentiles'],
                                                          error_propagation=init_dict['error_propagation'],
                                                          summary_window_size=init_dict['summary_window_size'],
                                                          sigma_multiples=init_dict['sigma_multiples'],
                                                          print_details=init_dict['print_details'],
                                                          print_progress=False, save=False,
                                                          print_time_scale_info=init_dict['print_time_scale_info'],
                                                          print_hint=False, fastMAPflag=True)
        slope_storage_helper[:, m] = one_window_helper.slope_storage[:, 0]
        noise_storage_helper[:, m] = one_window_helper.noise_level_storage[:, 0]
        if init_dict['antiCPyObject'] == 'NonMarkovEstimation':
            OU_storage_helper[:, m] = one_window_helper.OU_param_storage[:, 0]
            X_coupling_storage_helper[:, m] = one_window_helper.X_coupling_storage[:, 0]


def custom_error_callback(error):
    print(error, flush=True)
