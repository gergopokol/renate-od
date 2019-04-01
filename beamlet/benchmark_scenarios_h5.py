import numpy
import pandas


class Scenarios:
    def __init__(self):
        self.grid = numpy.linspace(0,1,101)

    def make_homo(self, component, q, z, a, density, temperature):
        elements = pandas.DataFrame([[-1, q], [0, z], [0, a]], index=['q', 'Z', 'A'], columns=['electron', 'ion1']).transpose()

        new_profiles = pandas.DataFrame([self.grid, [density]*len(self.grid), [temperature]*len(self.grid),
                                         [density]*len(self.grid), [temperature]*len(self.grid)],
                                        pandas.MultiIndex.from_arrays([['beamlet_grid', 'electron', 'electron', 'ion1', 'ion1'],
                                                                       ['', 'density', 'temperature', 'density', 'temperature'],
                                                                       ['m', 'm-3', 'eV', 'm-3', 'eV']],names=['type','property', 'unit'])).transpose()

        elements.to_hdf('data/output/'+component+'_'+str(density)+'m-3_'+str(temperature)+'eV.h5', key='components')
        new_profiles.to_hdf('data/output/'+component+'_'+str(density)+'m-3_'+str(temperature)+'eV.h5', key='profiles')


