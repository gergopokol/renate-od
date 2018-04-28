import matplotlib.pyplot
import pandas
import h5py


class Profiles:
    def __init__(self, access_path, key='profiles'):
        self.access_path = access_path
        self.key = key
        self.profiles = pandas.read_hdf(self.access_path, key=self.key)

    def plot_populations(self):
        axis = matplotlib.pyplot.subplot()
        axis = self.setup_population_axis(axis)
        matplotlib.pyplot.show()

    def plot_all_profiles(self):
        fig1 = matplotlib.pyplot.figure()
        grid = matplotlib.pyplot.GridSpec(3,1)
        ax1 = matplotlib.pyplot.subplot(grid[0,0])
        ax1 = self.setup_density_axis(ax1)
        ax2 = ax1.twinx()
        ax2 = self.setup_temperature_axis(ax2)
        ax1.set_title('Plasma profiles')
        ax3 = matplotlib.pyplot.subplot(grid[1:,0])
        ax3 = self.setup_population_axis(ax3)
        fig1.tight_layout()
        matplotlib.pyplot.show()

    def benchmark_to_renate(self):
        fig1 = matplotlib.pyplot.figure()
        ax1 = matplotlib.pyplot.subplot()
        ax1=self.setup_population_axis(ax1)
        ax1.legend(loc='best', ncol=2)
        ax1.set_title('Beamlet profiles - benchmark of RENATE-OD to RENATE')
        ax1.grid()
        fig1.tight_layout()
        matplotlib.pyplot.show()

    def get_number_of_levels(self, profiles):
        levels=profiles.filter(like='level', axis=1)
        number_of_levels = len(levels.keys())
        if number_of_levels ==0:
            number_of_levels = 9
        return number_of_levels

    def setup_density_axis(self, axis):
        axis.plot(self.profiles['beamlet_grid'], self.profiles['beamlet_density'], label='Density', color='b')
        axis.set_ylabel('Density [1/m3]')
        axis.yaxis.label.set_color('b')
        axis.legend(loc='upper left')
        axis.grid()
        return axis

    def setup_temperature_axis(self, axis):
        axis.plot(self.profiles['beamlet_grid'], self.profiles['beamlet_electron_temp'], color='r',
                               label='Electron_temperature')
        axis.plot(self.profiles['beamlet_grid'], self.profiles['beamlet_ion_temp'], '--', label='Ion_temperature', color='m')
        axis.set_ylabel('Temperature [eV]')
        axis.yaxis.label.set_color('r')
        axis.legend(loc='lower right')
        axis.grid()
        return axis

    def setup_population_axis(self, axis):
        number_of_levels = self.get_number_of_levels(self.profiles)
        for level in range(number_of_levels):
            label = 'level ' + str(level)
            axis.plot(self.profiles['beamlet_grid'], self.profiles[label], label=label)
        axis.set_yscale('log', nonposy='clip')
        axis.set_ylim([1e-5, 1])
        axis.set_xlabel('Distance [m]')
        axis.set_ylabel('Relative population [-]')
        axis.legend(loc='best', ncol=1)
        axis.set_title('Beamlet profiles')
        axis.grid()
        return axis

    def setup_benchmark_axis(self):
        benchmark_profiles = h5py.File(self.access_path, 'r')
        for i in range(9):
            matplotlib.pyplot.plot(self.profiles['beamlet_grid'], benchmark_profiles['benchmark']['renate_data_values'].value[i, :],
                      label='RENATE - level ' + str(i))
