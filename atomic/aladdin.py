import pycollisiondb
from pycollisiondb.pycollisiondb import PyCollision

class FitFunction(object):
    def __init__(self, data):
        self.coefficiants = self.assign_from_fit(data, 'coeffs')
        self.elo = self.assign_from_fit(data, 'Elo')
        self.ehi = self.assign_from_fit(data, 'Ehi')
        self.name = self.assign_from_fit(data, 'func')
        self.precision = self.assign_from_fit(data, 'fit_unc_perc')

    @staticmethod
    def assign_from_fit(data, tag):
        try:
            return data[tag]
        except KeyError:
            return data

class AladdinCrossSection(object):
    def __init__(self, data):
        if not isinstance(data, pycollisiondb.pycoll_ds.PyCollDataSet):
            raise TypeError('The expected type from aladdin API is PyCollision')
        self.energy = data.x
        self.cross_section = data.y
        self.len = data.n
        self.reaction = self.assign_data(data, tag='reaction')
        self.type = self.assign_data(data, tag='process_types')
        self.uncertainty = float(self.assign_from_json(data, tag='unc_perc'))
        self.threshold = float(self.assign_from_json(data, tag='threshold'))
        self.function = self.assign_from_json(data, tag='data_from_fit')
        if self.function:
            self.fit = FitFunction(data.metadata['json_data']['fit'])

    @staticmethod
    def assign_from_json(data, tag):
        try:
            return data.metadata['json_data'][tag]
        except KeyError:
            return None

    @staticmethod
    def assign_data(data, tag):
        try:
            return data.metadata[tag]
        except KeyError:
            return None

class AladdinData(object):
    def __init__(self, url_source = 'aladdin'):
        if isinstance(url_source, str):
            if url_source == 'aladdin':
                self.api_url = 'https://db-amdis.org/aladdin2/api/'
            elif url_source == 'collisiondb':
                self.api_url = 'https://db-amdis.org/collisiondb/api/'
            else:
                raise ValueError('The requested data source is not supported: ' + url_source)
        else:
            raise TypeError('The required data type for url specification is <str>.')

    def get_cross_section(self, transition, energy_grid):
        pass
