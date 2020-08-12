import numpy
import scipy


class SynthSignals:
    def __init__(self, frequency=1e6, synthetic_signals=None, data_path='signals/test.txt'):
        self.signals = synthetic_signals
        self.frequency = frequency
        if self.signals is None:
            self._load_synthetic_signals()

    def _check_synthetic_signals(self):
        pass

    def _load_synthetic_signals(self):
        pass


class Noise:
    pass

