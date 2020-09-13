import numpy
import pandas
import utility.getdata as ut
from utility.constants import Constants
from numpy.random import RandomState
from lxml import etree


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


class Noise(RandomState):
    def __init__(self, seed=None):
        RandomState.__init__(self, seed)
        self.constants = Constants()

    def photon_noise_generator(self, signal):
        signal = self.poisson(signal)
        return signal

    def shot_noise_generator(self, signal, gain, quantum_efficiency, load_resistance, noise_index, bandwidth):
        mean = signal * self.constants.charge_electron * gain * quantum_efficiency * load_resistance
        variance = numpy.array(numpy.sqrt(2 * self.constants.charge_electron ^ 2 * quantum_efficiency * gain ^ 2 *
                                           gain ^ noise_index * bandwidth) * load_resistance * numpy.sqrt(signal))
        signal = self.normal(mean, variance)
        return signal

    def johnson_noise_generator(self, signal, temperature, bandwidth, load_resistance):
        variance = numpy.sqrt(4 * self.constants.Boltzmann * temperature * bandwidth * load_resistance)
        signal = self.normal(signal, variance)
        return signal

    def voltage_noise_generator(self, signal, voltage_noise, load_resistance, load_capacity, internal_capacity):
        variance = voltage_noise * numpy.sqrt(1 / (2 * numpy.pi * load_resistance *
                                                    (load_capacity + internal_capacity))) + voltage_noise * \
                    (1 + internal_capacity / load_capacity) * numpy.sqrt(1.57 * 1 / (2 * numpy.pi * load_resistance *
                                                                                       load_capacity))
        signal = self.normal(signal, variance)
        return signal

    def dark_noise_generator(self, signal, dark_current, bandwidth, load_resistance):
        variance = numpy.sqrt(2 * self.constants.charge_electron * dark_current * bandwidth) * load_resistance
        signal = self.normal(signal, variance)
        return signal


class APD(Noise):
    def __init__(self, detector_parameters):
        Noise.__init__(self)
        self.__setup_detector_parameters(detector_parameters)

    def __setup_detector_parameters(self, detector_parameters):
        assert isinstance(detector_parameters, etree._ElementTree), 'Expected data type for <detector_parameters> ' \
                                                                    'is etree._ElementTree.'
        assert detector_parameters.getroot().find('head').find('type').text == 'apd', \
            'The detector type to be set is APD. Please check input data.'
        self.detector_temperature = float(detector_parameters.getroot().find('body').find('temperature').text)
        self.detector_gain = float(detector_parameters.getroot().find('body').find('gain').text)
        self.quantum_efficiency = float(detector_parameters.getroot().find('body').find('quantum_efficiency').text)
        self.noise_index = float(detector_parameters.getroot().find('body').find('noise_index').text)
        self.bandwidth = float(detector_parameters.getroot().find('body').find('bandwidth').text)
        self.dark_current = float(detector_parameters.getroot().find('body').find('dark_current').text)
        self.load_resistance = float(detector_parameters.getroot().find('body').find('load_resistance').text)
        self.load_capacity = float(detector_parameters.getroot().find('body').find('load_capacity').text)
        self.voltage_noise = float(detector_parameters.getroot().find('body').find('voltage_noise').text)
        self.internal_capacity = float(detector_parameters.getroot().find('body').find('internal_capacity').text)

    def add_noise_to_signal(self, signal):
        self.photon_noise_generator(signal)
        self.shot_noise_generator(signal, self.detector_gain, self.quantum_efficiency, self.load_resistance,
                                  self.noise_index, self.bandwidth)
        self.dark_noise_generator(signal, self.dark_current, self.bandwidth, self.load_resistance)
        self.voltage_noise_generator(signal, self.voltage_noise, self.load_resistance, self.load_capacity,
                                     self.internal_capacity)
        self.johnson_noise_generator(signal, self.detector_temperature, self.bandwidth, self.load_resistance)
        pass


class PMT(Noise):
    def __init__(self, detector_parameters):
        Noise.__init__(self)
        self.__setup_detector_parameters(detector_parameters)

    def __setup_detector_parameters(self, detector_parameters):
        pass

    def add_noise_to_signal(self):
        pass


class PP(Noise):
    def __init__(self, detector_parameters):
        Noise.__init__(self)
        self.__setup_detector_parameters(detector_parameters)

    def __setup_detector_parameters(self, detector_parameters):
        pass

    def add_noise_to_signal(self):
        pass


class Detector(APD, PMT, PP):
    def __init__(self, detector_type='apd', parameters=None, data_path=None):
        assert isinstance(detector_type, str), 'Expected data type for <detector_type> is str.'
        self.detector_type = detector_type
        if self.detector_type is 'apd':
            APD.__init__(self, self.__get_detector_parameters(parameters, data_path))
        elif self.detector_type is 'pmt':
            PMT.__init__(self, self.__get_detector_parameters(parameters, data_path))
        elif self.detector_type is 'pp':
            PP.__init__(self, self.__get_detector_parameters(parameters, data_path))
        else:
            raise ValueError('The requested detector type:' + self.detector_type + ' is not yet supported')

    def __get_detector_parameters(self, parameters, data_path):
        if isinstance(parameters, etree._ElementTree):
            self.data_path = 'From external workflow.'
        elif data_path is None:
            self.data_path = 'detector/'+self.detector_type+'_default.xml'
            parameters = ut.GetData(data_path_name=self.data_path).data
        elif isinstance(data_path, str):
            self.data_path = data_path
            parameters = ut.GetData(data_path_name=self.data_path).data
        else:
            raise ValueError('Expected data type for <data_path> is str or None, '
                             'in which case default values will be loaded')
        assert isinstance(parameters, etree._ElementTree), 'Expected data type for parameters is etree._ElementTree.'
        return parameters
