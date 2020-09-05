import numpy
from numpy.random import RandomState
from lxml import etree
import scipy
import utility.getdata as ut
import pandas


class SynthSignals:
    def __init__(self, frequency=1e6, synthetic_signals=None, data_path='signals/test.txt'):
        if isinstance(frequency, float):
            self.frequency = frequency
        else:
            raise TypeError('Expected data type for <frequency> is <int>.')
        self._check_synthetic_signals(synthetic_signals, data_path)

    def _check_synthetic_signals(self, synthetic_signals, path):
        if synthetic_signals is None:
            self._load_synthetic_signals(path)
        elif isinstance(synthetic_signals, pandas.DataFrame) or isinstance(synthetic_signals, pandas.Series):
            self.signals = synthetic_signals
        else:
            raise TypeError('The expected data type for <synthetic signals> is <pandas.DataFrame or pandas.Series>')

    def _load_synthetic_signals(self, path):
        self.signals = ut.GetData(data_path_name=path, data_format="array").data


class Parameters:
    def __init__(self):
        self._read_parameter_data

    def _read_parameter_data(self, data_path="parameters/parameters.txt"):
        parameters = numpy.loadtxt(fname=data_path)
        self.gain = parameters[0]
        self.quantum_efficiency = parameters[1]
        self.noise_index = parameters[2]
        self.bandwidth = parameters[3]
        self.load_resistance = parameters[4]
        self.load_capacity = parameters[5]
        self.dark_current = parameters[6]
        self.temperature = parameters[7]
        self.voltage_noise = parameters[8]
        self.internal_capacities = parameters[9]


class Noise(RandomState):

    def photon_noise_generator(self, signal):
        return self.poisson(signal)

    def _shot_noise_generator(self, parameters, constants):
        mean = self.signal * constants.charge_electron * parameters.gain * parameters.quantum_efficiency * parameters.load_resistance
        deviation = numpy.array(numpy.sqrt(2 * constants.charge_electron * constants.charge_electron * parameters.quantum_efficiency * parameters.gain * parameters.gain * parameters.gain ^ parameters.noise_index * parameters.bandwidth) * parameters.load_resistance * numpy.sqrt(self.signal))
        return self.signal

    def _johnson_noise_generator(self, parameters, constants):
        deviation = numpy.sqrt(4 * constants.boltzmanns_constant * parameters.temperature * parameters.bandwidth * parameters.load_resistance)
        return self.signal

    def _voltage_noise_generator(self, parameters):
        deviation = parameters.voltage_noise * numpy.sqrt(1 / (2 * numpy.pi * parameters.load_resistance * (parameters.load_capacity + parameters.internal_capacities))) + parameters.voltage_noise * (1 + parameters.internal_capacities / parameters.load_capacity) * numpy.sqrt(1.57 * 1 / (2 * numpy.pi * parameters.load_resistance * parameters.load_capacity))
        return self.signal

    def _dark_noise_generator(self, parameters, constants):
        deviation = numpy.sqrt(2 * constants.charge_electron * parameters.dark_current * parameters.bandwidth) * parameters.load_resistance
        return self.signal


class APD(Noise):

    def add_noise_to_signal(self):
        pass


class PMT(Noise):

    def add_noise_to_signal(self):
        pass


class PP(Noise):

    def add_noise_to_signal(self):
        pass


class Detector(APD, PMT, PP):
    def __init__(self, detector_type='apd', parameters=None, data_path=None):
        assert isinstance(data_path, str), 'Expected data type for data_path is str.'
        if not isinstance(parameters, etree._ElementTree):
            self.data_path = data_path
            parameters = self.__get_detector_parameters()
        else:
            self.data_path = 'From external workflows.'
        assert isinstance(type, str), 'Expected data type for <detector_type> is str.'
        self.detector_type = detector_type
        if self.detector_type is 'apd':
            APD.__init__(self, parameters)
        elif self.detector_type is 'pmt':
            PMT.__init__(self, parameters)
        elif self.detector_type is 'pp':
            PP.__init__(self, parameters)
        else:
            raise ValueError('The requested detector type:' + self.detector_type + ' is not yet supported')

    def __get_detector_parameters(self):
        parameters = ut.GetData(data_path_name=self.data_path).data
        assert isinstance(parameters, etree._ElementTree)
        return parameters
