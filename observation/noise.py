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
    def __init__(self, detector_parameters):
        Noise.__init__(self)
        self.__setup_detector_parameters(detector_parameters)

    def __setup_detector_parameters(self, detector_parameters):
        assert isinstance(detector_parameters, etree._ElementTree), 'Expected data type for <detector_parameters> ' \
                                                                    'is etree._ElementTree.'
        assert detector_parameters.getroot().find('head').find('type').text is 'apd', \
            'The detector type to be set is APD. Please check input data.'
        self.detector_temperature = float(detector_parameters.getroot().find('body').find('temperature').text)
        self.detector_gain = float(detector_parameters.getroot().find('body').find('gain').text)
        self.quantum_efficiency = float(detector_parameters.getroot().find('body').find('quantum_efficiency').text)
        self.noise_index = float(detector_parameters.getroot().find('body').find('noise_index').text)
        self.bandwidth = float(detector_parameters.getroot().find('body').find('bandwidth').text)
        self.dark_voltage = float(detector_parameters.getroot().find('body').find('dark_voltage').text)
        self.load_resistance = float(detector_parameters.getroot().find('body').find('load_resistance').text)
        self.load_capacity = float(detector_parameters.getroot().find('body').find('load_capacity').text)
        self.voltage_noise = float(detector_parameters.getroot().find('body').find('voltage_noise').text)
        self.internal_capacity = float(detector_parameters.getroot().find('body').find('internal_capacity').text)

    def add_noise_to_signal(self):
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
