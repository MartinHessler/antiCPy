import numpy as np

def _summary_statistics_helper(slope_storage, summary_window_size=10, sigma_multiples=np.array([1, 3])):
    """
    Computes the mean of the drift slope :math:`\hat{\zeta}` and its standard error in a predefined summary
    statistics window.

    :param summary_window_size: If ``error_propagation = 'summary statistics'`` is chosen, the parameter defines
                    the number of drift slope estimates to use in a window summary statistic. The windows are shifted
                    by one.
    :type summary_window_size: int
    :param sigma_multiples: The array hast two entries. If ``error_propagation = 'summary statistics'`` is chosen,
                    the entries define the drift slope standard error multiples which are used to calculate the
                    uncertainty bands.
    :type sigma_multiples: One dimensional numpy array of float .
    """
    step_size = 1
    drift_slope_size = slope_storage.shape[1]
    drift_std_error = np.zeros(drift_slope_size)
    drift_mean = np.zeros(drift_slope_size)
    summary_loop = np.arange(0, drift_slope_size - summary_window_size + 1, step_size)
    for i in range(summary_loop.size):
        summary_window = np.roll(slope_storage[0, :], shift=- summary_loop[i])[:summary_window_size]
        drift_std_error[i + summary_window_size - 1] = np.std(summary_window)
        drift_mean[i + summary_window_size - 1] = np.mean(summary_window)
    drift_std_error[:summary_window_size] = drift_std_error[summary_window_size]
    drift_std_error /= np.sqrt(summary_window_size)
    drift_mean[:summary_window_size] = drift_mean[summary_window_size]
    slope_storage[0, :] = drift_mean
    slope_storage[1, :] = drift_mean - sigma_multiples[0] * drift_std_error
    slope_storage[2, :] = drift_mean + sigma_multiples[0] * drift_std_error
    slope_storage[3, :] = drift_mean - sigma_multiples[1] * drift_std_error
    slope_storage[4, :] = drift_mean + sigma_multiples[1] * drift_std_error
    return slope_storage