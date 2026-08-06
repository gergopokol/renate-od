import utility.getdata


class Plasma(object):
    """
    Collects plasma composition and profile data to populate Beamlet objects with.
    Currently supports the 'grid' profile type: density/temperature profiles already
    laid out on a grid, stored the same way Beamlet reads/writes them (HDF5 'components'
    and 'profiles' keys, referenced by an XML param file's plasma_source element).
    """

    def __init__(self, source='local', data_path='device_data/test/test_plasma.xml'):
        self.source = source
        self.data_path = data_path
        self.components = None
        self.profiles = None

    def load_plasma_data(self):
        if self.source == 'local':
            self.__load_local_grid_profiles()
        else:
            raise ValueError('Plasma data source: ' + str(self.source) + ' is not supported. '
                             'Supported sources are: local.')

    def __load_local_grid_profiles(self):
        param = utility.getdata.GetData(data_path_name=self.data_path).data
        hdf5_path = param.getroot().find('body').find('plasma_source').text
        self.components = utility.getdata.GetData(data_path_name=hdf5_path, data_key=['components']).data
        self.profiles = utility.getdata.GetData(data_path_name=hdf5_path, data_key=['profiles']).data

    def load_plasma_components(self):
        if self.components is None:
            self.load_plasma_data()
        return self.components

    def load_plasma_profiles(self):
        if self.profiles is None:
            self.load_plasma_data()
        return self.profiles
