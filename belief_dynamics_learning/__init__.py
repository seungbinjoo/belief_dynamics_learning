___author__ = "Lara Brudermüller"
__license__ = "BSD 3-clause"

import warnings as _warnings

try:
    import numpy
    del numpy
except ImportError:
    _warnings.warn('Install `numpy` ("pip install numpy").')


__version__ = "1.0.0"
