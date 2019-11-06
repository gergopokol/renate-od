import unittest


class CrmTestCase(unittest.TestCase):

    @staticmethod
    def _areSeriesEqual(series1, series2, precision):
        booleans = abs(series1[1:] - series2[1:])/series1[1:] <= precision
        if False in booleans:
            return False
        else:
            return True

    def assertAlmostEqualRateEvolution(self, actual, reference, precision=1E-3, msg=''):
        if reference.profiles['level '+actual.atomic_db.inv_atomic_dict[0]][0] == 1:
            normalization_factor = actual.profiles['level '+actual.atomic_db.inv_atomic_dict[0]][0]
        else:
            normalization_factor = 1.
        for level_index in range(actual.atomic_db.atomic_levels):
            level = 'level ' + actual.atomic_db.inv_atomic_dict[level_index]
            self.assertTrue(self._areSeriesEqual(actual.profiles[level]/normalization_factor,
                                                 reference.profiles[level], precision))

    def assertNotAlmostEqualRateEvolution(self, actual, reference, precision=1E-3, msg=''):
        pass

    def assertAlmostEqualEmissionDensity(self, actual, reference, precision=1E-3, msg=''):
        pass

    def assertNotAlmostEqualEmissionDensity(self, actual, reference, precision=1E-3, msg=''):
        pass

    def assertAlmostEqualBeamAttenuation(self, actual, reference, precision=1E-3, msg=''):
        pass

    def assertNotAlmostEqualBeamAttenuation(self, actual, reference, precision=1E-3, msg=''):
        pass

    def assertAlmostEqualRelativePopulation(self, actual, reference, precision=1E-3, msg=''):
        pass

    def assertNotAlmostEqualRelativePopulation(self, actual, reference, precision=1E-3, msg=''):
        pass
