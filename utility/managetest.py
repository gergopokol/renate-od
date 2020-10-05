from unittest import TestCase
from utility.manage import Version
from unittest.util import safe_repr


class VersionTest(TestCase):

    EQUAL_TEST = '3.4.5'
    NOT_EQUAL_TEST = '3.3.5'
    GREATER_THEN_TEST_TRUE = ['2.3.4', '2.10.3', '3.3.5', '3.4.4']
    GREATER_THEN_TEST_FALSE = ['3.4.5', '4.4.5', '3.5.4', '3.4.6']

    def setUp(self):
        self.version = Version('3.4.5')

    def tearDown(self):
        del self.version

    def test_version_equality_operator(self):
        reference = Version(self.EQUAL_TEST)
        self.assertTrue(self.version == reference, msg='Objects: %s and %s are expected to be equal' %
                                                       (safe_repr(self.version), safe_repr(reference)))

    def test_version_inequality_operator(self):
        reference = Version(self.NOT_EQUAL_TEST)
        self.assertTrue(self.version != reference, msg='Objects: %s and %s are expected not to be equal' %
                                                       (safe_repr(self.version), safe_repr(reference)))

    def test_version_greater_then_operator_true(self):
        for case in self.GREATER_THEN_TEST_TRUE:
            reference = Version(case)
            self.assertTrue(self.version > reference, msg='Object: %s is expected to be greater then %s' %
                                                          (safe_repr(self.version), safe_repr(reference)))

    def test_version_greater_then_operator_false(self):
        for case in self.GREATER_THEN_TEST_FALSE:
            reference = Version(case)
            self.assertFalse(self.version > reference, msg='Object: %s is expected not to be greater then %s' %
                                                           (safe_repr(self.version), safe_repr(reference)))
