import os
import h5py
import sys
import matplotlib.pyplot
import pandas
from utility import getdata
from crm_solver import beamlet
from lxml import etree

hdf5_id = h5py.File('../data/beamlet/test_profiles.h5', 'r')
print(hdf5_id['benchmark']['renate_data_values'].value)

matplotlib.pyplot.plot(hdf5_id['profiles']['block0_values'].value[:,2],hdf5_id['benchmark']['renate_data_values'].value[0,:],'--')
matplotlib.pyplot.plot(hdf5_id['profiles']['block0_values'].value[:,2],hdf5_id['benchmark']['renate_data_values'].value[1,:],'--')
matplotlib.pyplot.plot(hdf5_id['profiles']['block0_values'].value[:,2],hdf5_id['benchmark']['renate_data_values'].value[2,:],'--')
matplotlib.pyplot.plot(hdf5_id['profiles']['block0_values'].value[:,2],hdf5_id['benchmark']['renate_data_values'].value[3,:],'--')
matplotlib.pyplot.plot(hdf5_id['profiles']['block0_values'].value[:,2],hdf5_id['benchmark']['renate_data_values'].value[4,:],'--')
matplotlib.pyplot.plot(hdf5_id['profiles']['block0_values'].value[:,2],hdf5_id['benchmark']['renate_data_values'].value[5,:],'--')
matplotlib.pyplot.plot(hdf5_id['profiles']['block0_values'].value[:,2],hdf5_id['benchmark']['renate_data_values'].value[6,:],'--')
matplotlib.pyplot.plot(hdf5_id['profiles']['block0_values'].value[:,2],hdf5_id['benchmark']['renate_data_values'].value[7,:],'--')
matplotlib.pyplot.plot(hdf5_id['profiles']['block0_values'].value[:,2],hdf5_id['benchmark']['renate_data_values'].value[8,:],'--')
matplotlib.pyplot.yscale('log')

os.chdir('../')

profiles=pandas.DataFrame(hdf5_id['profiles']['block0_values'].value,
                          columns=['beamlet_density','beamlet_electron_temp','beamlet_grid','beamlet_ion_temp'] )

ROD_beamlet=beamlet.Beamlet(profiles=profiles, data_path='../data/beamlet/test.xml')
ROD_beamlet.solve_numerically()
ROD_beamlet.plot_populations()
