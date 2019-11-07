import unittest
from unittest.util import safe_repr


class CrmTestCase(unittest.TestCase):

    def _areSeriesAlmostEqual(self, series1, series2, precision):
        statement = True
        status_message = 'Value Series 1, Value Series 2, Almost Equal'
        if len(series2) != len(series1):
            self.failureException('First series has %s elements whereas second series %s elements \n' %
                                  (safe_repr(len(series1)), safe_repr(len(series2))))
        for i in range(1, len(series1)):
            if not abs(series1[i] - series2[i])/series1[i] <= precision:
                statement = False
            status_message += '\n %s, %s, %s' % (safe_repr(series1[i]), safe_repr(series2[i]),
                                                 safe_repr(abs(series1[i] - series2[i])/series1[i] <= precision))
        if statement:
            return True, status_message
        else:
            return False, status_message

    @staticmethod
    def _getNormalization(actual, reference):
        if reference.profiles['level '+reference.atomic_db.inv_atomic_dict[0]][0] == 1:
            return actual.profiles['level '+actual.atomic_db.inv_atomic_dict[0]][0]
        else:
            return 1.

    def assertAlmostEqualRateEvolution(self, actual, reference, precision=1E-3, msg=''):
        for level_index in range(actual.atomic_db.atomic_levels):
            level = 'level ' + actual.atomic_db.inv_atomic_dict[level_index]
            statement, status = self._areSeriesAlmostEqual(actual.profiles[level]/self._getNormalization(actual, reference),
                                                           reference.profiles[level], precision)
            if not statement:
                standardMsg = 'Population evolution on %s are not within relative error of %s. Series 1 = actual, ' \
                              'Series 2 = reference \n' % (level, safe_repr(precision)) + status
                msg = self._formatMessage(msg, standardMsg)
                self.fail(msg)

    def assertNotAlmostEqualRateEvolution(self, actual, reference, precision=1E-3, msg=''):
        for level_index in range(actual.atomic_db.atomic_levels):
            level = 'level ' + actual.atomic_db.inv_atomic_dict[level_index]
            statement, status = self._areSeriesAlmostEqual(actual.profiles[level] / self._getNormalization(actual, reference),
                                                           reference.profiles[level], precision)
            if statement:
                standardMsg = 'Population evolution on %s are within relative error of %s. This is NOT expected ' \
                              'Series 1 = actual, Series 2 = reference \n' % (level, safe_repr(precision)) + status
                msg = self._formatMessage(msg, standardMsg)
                self.fail(msg)

    def assertAlmostEqualEmissionDensity(self, actual, reference, precision=1E-3, msg=''):
        default_level_vals = actual.atomic_db.set_default_atomic_levels()
        statement, status = self._areSeriesAlmostEqual(actual.profiles[default_level_vals[3]],
                                                       reference.profiles[default_level_vals[3]], precision)
        if not statement:
            standardMsg = 'Linear emission density values for transition: %s are not within relative error of %s.' \
                          'Series 1 = actual, Series 2 = reference \n' % (default_level_vals[3], safe_repr(precision)) \
                          + status
            msg = self._formatMessage(msg, standardMsg)
            self.fail(msg)

    def assertNotAlmostEqualEmissionDensity(self, actual, reference, precision=1E-3, msg=''):
        default_level_vals = actual.atomic_db.set_default_atomic_levels()
        statement, status = self._areSeriesAlmostEqual(actual.profiles[default_level_vals[3]],
                                                       reference.profiles[default_level_vals[3]], precision)
        if statement:
            standardMsg = 'Linear emission density values for transition: %s are within relative error of %s. ' \
                          'This is not expected. Series 1 = actual, Series 2 = reference \n' % (default_level_vals[3],
                                                                                                safe_repr(precision)) \
                                                                                                + status
            msg = self._formatMessage(msg, standardMsg)
            self.fail(msg)

    def assertAlmostEqualBeamAttenuation(self, actual, reference, precision=1E-3, msg=''):
        status, statement = self._areSeriesAlmostEqual(actual.profiles['linear_density_attenuation'],
                                                       reference.profiles['linear_density_attenuation'], precision)
        if not status:
            standardMsg = 'Beam attenuation is not within relative error of %s. ' \
                          'Series 1 = actual, Series 2 = reference \n' % (safe_repr(precision)) + status
            msg = self._formatMessage(msg, standardMsg)
            self.fail(msg)

    def assertNotAlmostEqualBeamAttenuation(self, actual, reference, precision=1E-3, msg=''):
        status, statement = self._areSeriesAlmostEqual(actual.profiles['linear_density_attenuation'],
                                                       reference.profiles['linear_density_attenuation'], precision)
        if status:
            standardMsg = 'Beam attenuation is within relative error of %s. This is not expected.' \
                          'Series 1 = actual, Series 2 = reference \n' % (safe_repr(precision)) + status
            msg = self._formatMessage(msg, standardMsg)
            self.fail(msg)

    def assertAlmostEqualRelativePopulation(self, actual, reference, precision=1E-3, msg=''):
        for level_index in range(actual.atomic_db.atomic_levels):
            level = 'rel.pop ' + actual.atomic_db.inv_atomic_dict[level_index]
            status, statement = self._areSeriesAlmostEqual(actual.profiles[level], reference.profiles[level], precision)
            if not status:
                standardMsg = 'Relative population evolution on level %s are not within relative error of %s. ' \
                              'Series 1 = actual, Series 2 = reference. \n' % (safe_repr(level), safe_repr(precision)) \
                              + statement
                msg = self._formatMessage(msg, standardMsg)
                self.fail(msg)

    def assertNotAlmostEqualRelativePopulation(self, actual, reference, precision=1E-3, msg=''):
        for level_index in range(actual.atomic_db.atomic_levels):
            level = 'rel.pop ' + actual.atomic_db.inv_atomic_dict[level_index]
            status, statement = self._areSeriesAlmostEqual(actual.profiles[level], reference.profiles[level], precision)
            if status:
                standardMsg = 'Relative population evolution on level %s are within relative error of %s. ' \
                              'This is not expected. Series 1 = actual, Series 2 = reference. \n' % \
                              (safe_repr(level), safe_repr(precision)) + statement
                msg = self._formatMessage(msg, standardMsg)
                self.fail(msg)

