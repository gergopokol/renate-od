import numpy
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


class Generator:
    def __init__(self, signal=0):
        self.signal = signal

    def _generate_poisson(self):
        if isinstance(self.signal, pandas.Series):
            noised_signal = pandas.Series(numpy.random.poisson(self.signal))
            return noised_signal

        elif isinstance(self.signal, pandas.DataFrame):
            noised_signal = pandas.DataFrame(numpy.random.poisson(self.signal))
            return noised_signal

        else:
            raise TypeError('The expected data type for <_generate_poisson> is <pandas.DataFrame or pandas.Series>')

    def _generate_gaussian(self, deviation):
        if isinstance(self.signal, pandas.Series):
            noised_signal = pandas.Series(numpy.random.normal(loc=self.signal, scale=deviation))
            return noised_signal

        elif isinstance(self.signal, pandas.DataFrame):
            noised_signal = pandas.DataFrame(numpy.random.normal(self.signal, scale=deviation))
            return noised_signal

        else:
            raise TypeError('The expected data type for <_generate_gaussian> is <pandas.DataFrame or pandas.Series>')


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


class Noise(Generator):
    def __init__(self, parameters, constants):
        self.parameters = parameters
        self.constants = constants

    def _photon_noise_generator(self, signal):
        g = Generator(signal)
        signal = g._generate_poisson()

    def _shot_noise_generator(self, signal):
        mean = signal * self.constants.charge_electron * self.parameters.gain * self.parameters.quantum_efficiency * self.parameters.load_resistance
        deviation = numpy.sqrt(2 * self.constants.charge_electron * signal * self.constants.charge_electron * self.parameters.quantum_efficiency * self.parameters.gain * self.parameters.gain * self. parameters.gain ^self.parameters.noise_index *self.parameters.bandwidth) * self.parameters.load_resistance
        g = Generator(mean)
        signal = g._generate_gaussian(deviation)

    def _johnson_noise_generator(self, signal):
        deviation = numpy.sqrt(4 * self.constants.boltzmanns_constant * self.parameters.temperature * self.parameters.bandwidth * self.parameters.load_resistance)
        g = Generator(signal)
        signal = g._generate_gaussian(deviation)

    def _voltage_noise_generator(self, signal):
        deviation = self.parameters.voltage_noise * numpy.sqrt(1 / (2 * numpy.pi * self.parameters.load_resistance * (self.parameters.load_capacity + self.parameters.internal_capacities))) + self.parameters.voltage_noise * (1 + self.parameters.internal_capacities / self.parameters.load_capacity) * numpy.sqrt(1.57 * 1 / (2 * numpy.pi * self.parameters.load_resistance* self.parameters.load_capacity))
        g = Generator(signal)
        signal = g._generate_gaussian(deviation)

    def _dark_noise_generator(self, signal):
        deviation = numpy.sqrt(2 * self.constants.charge_electron * self.parameters.dark_current * self.parameters.bandwidth) * self.parameters.load_resistance
        g = Generator(signal)
        signal = g._generate_gaussian(deviation)

