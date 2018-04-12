import matplotlib.pyplot
import pandas
import h5py


class Profiles:
    def __init__(self, access_path, key='profiles'):
        self.access_path = access_path
        self.key = key

    def plot_populations(self):
        profiles = pandas.read_hdf(self.access_path, key=self.key)
        for level in range(9):
            label = 'level ' + str(level)
            matplotlib.pyplot.plot(profiles['beamlet_grid'], profiles[label], label=label)
            matplotlib.pyplot.yscale('log', nonposy='clip')
            matplotlib.pyplot.ylim((1e-5, 1))
        matplotlib.pyplot.legend(loc='best', ncol=1)
        matplotlib.pyplot.xlabel('x')
        matplotlib.pyplot.grid()
        matplotlib.pyplot.show()

    def plot_all_profiles(self):
        profiles = pandas.read_hdf(self.access_path, key=self.key)
        matplotlib.pyplot.figure()
        grid = matplotlib.pyplot.GridSpec(3,1)
        ax1 = matplotlib.pyplot.subplot(grid[0,0])
        ax1.plot(profiles['beamlet_grid'], profiles['beamlet_density'], label='Density', color='b')
        ax1.set_ylabel('Density [1/m3]', color='b')
        ax2 = ax1.twinx()
        ax2.plot(profiles['beamlet_grid'], profiles['beamlet_electron_temp'], color='r',
                               label='Electron_temperature')
        ax2.plot(profiles['beamlet_grid'], profiles['beamlet_ion_temp'], label='Ion_temperature', color='m')
        ax2.set_ylabel('Temperature [eV]', color='r')
        ax1.legend(loc='upper left')
        ax2.legend(loc='lower right')
        matplotlib.pyplot.title('Plasma profiles')
        ax1.grid()
        matplotlib.pyplot.tight_layout()
        matplotlib.pyplot.subplot(grid[1:,0])
        for level in range(9):
            label = 'level ' + str(level)
            matplotlib.pyplot.plot(profiles['beamlet_grid'], profiles[label], label=label)
        matplotlib.pyplot.yscale('log', nonposy='clip')
        matplotlib.pyplot.ylim((1e-5, 1))
        matplotlib.pyplot.legend(loc='best', ncol=1)
        matplotlib.pyplot.xlabel('Distance [m]')
        matplotlib.pyplot.ylabel('Relative population [-]')
        matplotlib.pyplot.title('Beamlet profiles')
        matplotlib.pyplot.grid()
        matplotlib.pyplot.tight_layout()
        matplotlib.pyplot.show()

    def benchmark_to_renate(self):
        profiles = pandas.read_hdf(self.access_path, key=self.key)
        benchmark_profiles = h5py.File(self.access_path, 'r')
        for i in range(9):
            matplotlib.pyplot.plot(profiles['beamlet_grid'],
                                   benchmark_profiles['benchmark']['renate_data_values'].value[i, :],
                                   label='RENATE - level ' + str(i))
        for level in range(9):
            label = 'level ' + str(level)
            matplotlib.pyplot.plot(profiles['beamlet_grid'], profiles[label], '--', label='ROD - ' + label)
        matplotlib.pyplot.yscale('log', nonposy='clip')
        matplotlib.pyplot.ylim((1e-5, 1))
        matplotlib.pyplot.legend(loc='right', ncol=1)
        matplotlib.pyplot.xlabel('Distance [m]')
        matplotlib.pyplot.ylabel('Relative population [-]')
        matplotlib.pyplot.title('Beamlet profiles')
        matplotlib.pyplot.grid()
        matplotlib.pyplot.tight_layout()
        matplotlib.pyplot.show()
