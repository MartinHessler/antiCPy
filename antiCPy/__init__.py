import sys

from .early_warnings.drift_slope import *
from .trend_extrapolation import *

from .early_warnings.dominant_eigenvalue import analysis
from .early_warnings.dominant_eigenvalue import param_opt
from .early_warnings.dominant_eigenvalue import graphics

__all__ = ['LangevinEstimation', 'BinningLangevinEstimation', 'NonMarkovEstimation', 'RocketFastResilienceEstimation',
           'CPSegmentFit', 'BatchedCPSegmentFit']


if sys.version_info < (3, 6, 0):
    import warnings

    warnings.warn(
        'The installed Python version reached its end-of-life. Please upgrade to a newer Python version for receiving '
        'further antiCPy updates.', Warning)

__version__ = '0.0.7.post2'

__author__ = 'Martin Heßler'
