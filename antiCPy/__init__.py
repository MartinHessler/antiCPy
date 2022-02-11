import sys

if sys.version_info < (3, 6, 0):
    import warnings

    warnings.warn(
        'The installed Python version reached its end-of-life. Please upgrade to a newer Python version for receiving '
        'further antiCPy updates.', Warning)

__version__ = '0.0.2'

__author__ = 'Martin Heßler'

__all__ = ['early_warnings', 'trend_extrapolation']
