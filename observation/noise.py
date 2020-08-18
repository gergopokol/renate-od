import numpy
import scipy
import utility.getdata as ut
import pandas
import utility.constants as const


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
    def __init__(self, Data=0, Deviation=0):
        self.Data = Data
        self.Deviation = Deviation

    def GeneratePoisson(self):
        if isinstance(self.Data, pandas.Series):
            Temporary = numpy.random.poisson(self.Data)
            NoisedSignal = pandas.Series(Temporary)
            return NoisedSignal

        elif isinstance(self.Data, pandas.DataFrame):
            Temporary = numpy.random.poisson(self.Data)
            NoisedSignal = pandas.DataFrame(Temporary)
            return NoisedSignal

        else:
            raise TypeError('The expected data type for <GeneratePoisson> is <pandas.DataFrame or pandas.Series>')

    def GenerateGaussian(self):
        if isinstance(self.Data, pandas.Series):
            Temporary = numpy.random.normal(loc=self.Data, scale=self.Deviation)
            NoisedSignal = pandas.Series(Temporary)
            return NoisedSignal

        elif isinstance(self.Data, pandas.DataFrame):
            Temporary = numpy.random.normal(self.Data, scale=self.Deviation)
            NoisedSignal = pandas.DataFrame(Temporary)
            return NoisedSignal

        else:
            raise TypeError('The expected data type for <GenerateGaussian> is <pandas.DataFrame or pandas.Series>')

    pass


class Parameters:
    def __init__(self):
        self._read_parameter_data

    def _read_parameter_data(self):
        parameters = numpy.loadtxt(fname="parameters/parameters.txt")
        self.Gain = parameters[0]
        self.QuantumEfficiency = parameters[1]
        self.NoiseIndex = parameters[2]
        self.Bandwidth = parameters[3]
        self.LoadResistance = parameters[4]
        self.LoadCapacity = parameters[5]
        self.DarkCurrent = parameters[6]
        self.Temperature = parameters[7]
        self.VoltageNoise = parameters[8]
        self.InternalCapacities = parameters[9]


class Noise(Generator):
    def __init__(self, Parameters, Constants):
        self.Parameters = Parameters
        self.Constants = Constants

    def PhotonNoiseGenerator(self, Signal):
        g = Generator(Signal)
        Signal = g.GeneratePoisson()

    def ShotNoiseGenerator(self, Signal):
        Mean = Signal * self.Constants.charge_electron * self.Parameters.Gain * self.Parameters.QuantumEfficiency * self.Parameters.LoadResistance
        Deviation = numpy.sqrt(2 * self.Constants.charge_electron * Signal * self.Constants.charge_electron * self.Parameters.QuantumEfficiency * self.Parameters.Gain * self.Parameters.Gain * self. Parameters.Gain^self.Parameters.NoiseIndex *self.Parameters.Bandwidth) * self.Parameters.LoadResistance
        g = Generator(Mean, Deviation)
        Signal = g.GenerateGaussian()

    def JohnsonNoiseGenerator(self, Signal):
        Deviation = numpy.sqrt(4 * self.Constants.Boltzmanns_constant * self.Parameters.Temperature * self.Parameters.Bandwidth * self.Parameters.LoadResistance)
        g = Generator(Signal, Deviation)
        Signal = g.GenerateGaussian()

    def VoltageNoiseGenerator(self, Signal):
        Deviation = self.Parameters.VoltageNoise * numpy.sqrt(1 / (2 * numpy.pi * self.Parameters.LoadResistance * (self.Parameters.LoadCapacity + self.Parameters.InternalCapacities))) + self.Parameters.VoltageNoise * (1 + self.Parameters.InternalCapacities / self.Parameters.LoadCapacity) * numpy.sqrt(1.57 * 1 / (2 * numpy.pi * self.Parameters.LoadResistance* self.Parameters.LoadCapacity))
        g = Generator(Signal, Deviation)
        Signal = g.GenerateGaussian()

    def DarkNoiseGenerator(self, Signal):
        Deviation = numpy.sqrt(2 * self.Constants.charge_electron * self.Parameters.DarkCurrent * self.Parameters.Bandwidth) * self.Parameters.LoadResistance
        g = Generator(Signal, Deviation)
        Signal = g.GenerateGaussian()

    pass
