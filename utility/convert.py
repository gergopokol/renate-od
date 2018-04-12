import pandas
import h5py
import os
import utility


def convert_beamlet_profiles(data_path_name='data/beamlet/test_profiles.h5'):
    h5file = h5py.File(data_path_name,'r')
    pandas_profiles = pandas.DataFrame()
    columns = list(h5file.keys())
    for column in columns:
        pandas_profiles[column] = h5file[column].value
    h5file.close()
    os.rename(data_path_name,data_path_name + '.old')
    pandas_profiles.to_hdf(data_path_name)


def convert_from_cm2_to_m2(cross_section):
        return cross_section / 1.e4


def convert_from_cm_to_m(length):
        return length / 1.e2


def convert_from_10_9_to_1(density):
    return density * 1.e19


def convert_beamlet_profiles_to_si(data_path_name='beamlet/test_profiles.h5'):
    pandas_profiles = utility.getdata.GetData(data_path_name=data_path_name).data
    assert isinstance(pandas_profiles, pandas.DataFrame)
    full_data_path_name = 'data/' + data_path_name
    pandas_profiles['beamlet_density'] = convert_from_10_9_to_1(pandas_profiles['beamlet_density'])
    # pandas_profiles['beamlet_grid'] = convert_from_cm_to_m(pandas_profiles['beamlet_grid'])
    os.rename(full_data_path_name,full_data_path_name + '.non-si')
    pandas_profiles.to_hdf(full_data_path_name,'profiles')
    print('Beamlet.param converted to SI in file: ' + full_data_path_name)


def calculate_velocity_from_energy(energy, mass):
    velocity = (2 * float(energy) * 1.602176487e-16 / mass) ** 0.5
    return velocity
