import unittest


class CrmTestCase(unittest.TestCase):
    def assertAlmostEqualRateEvolution(self, actual, reference, precision=1E-3, msg=''):
        if reference.profiles['level '+actual.atomic_db.inv_atomic_dict[0]][0] == 1:
            normalization_factor = actual.profiles['level '+actual.atomic_db.inv_atomic_dict[0]][0]
        else:
            normalization_factor = 1.
        for level_index in range(actual.atomic_db.atomic_levels):
            for index in range(len(actual.profiles)):
                if actual.profiles['level '+actual.atomic_db.inv_atomic_dict[level_index]][index] != 0:
                    self.assertTrue(abs(actual.profiles['level '+actual.atomic_db.inv_atomic_dict[level_index]][index] -
                                        reference.profiles['level ' + actual.atomic_db.inv_atomic_dict[level_index]][index])/
                                    reference.profiles['level ' + actual.atomic_db.inv_atomic_dict[level_index]][index] <= precision)

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
