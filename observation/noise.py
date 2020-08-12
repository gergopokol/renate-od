import numpy
import scipy
import utility.getdata as ut


class SynthSignals:
    def __init__(self, frequency=1e6, synthetic_signals=None, data_path='signals/test.txt'):
        if isinstance(frequency, int):
            self.frequency = frequency
        else:
            raise TypeError('Expected data type for <frequency> is <int>.')
        self._check_synthetic_signals(synthetic_signals)

    def _check_synthetic_signals(self, synthetic_signals):
        pass


class Noise:
    pass

