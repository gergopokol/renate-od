import numpy
import scipy


class RenateSyntheticDiagnostic:
    def __init__(self, synthetic_signals=None, data_path='signals/test.txt'):
        self.signals = synthetic_signals
        if self.signals is None:
            self._load_synthetic_signals()

    def _check_synthetic_signals(self):
        pass

    def _load_synthetic_signals(self):
        pass


class Noise:
    pass

