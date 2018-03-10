import pandas
import h5py
import os

def convert_beamlet_profiles(data_path_name='data/beamlet/test_profiles.h5'):
    h5file = h5py.File(data_path_name,'r')
    pandas_profiles = pandas.DataFrame()
    columns = list(h5file.keys())
    for column in columns:
        pandas_profiles[column] = h5file[column].value
    h5file.close()
    os.rename(data_path_name,data_path_name + '.old')
    pandas_profiles.to_hdf(data_path_name)

