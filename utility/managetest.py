from unittest import TestCase
from utility.manage import Version
from utility.manage import CodeInfo
from unittest.util import safe_repr


class VersionTest(TestCase):

    EQUAL_TEST = '3.4.5'
    NOT_EQUAL_TEST = '3.3.5'
    GREATER_THEN_TEST_TRUE = ['2.3.4', '2.10.3', '3.3.5', '3.4.4']
    GREATER_THEN_TEST_FALSE = ['3.4.5', '4.4.5', '3.5.4', '3.4.6']
    LESS_THEN_TEST_TRUE = ['4.4.5', '3.5.4', '3.4.6']
    LESS_THEN_TEST_FALSE = ['3.4.5', '2.10.3', '3.3.5', '3.4.4']
    MAJOR_RELEASE_TEST = [4, 0, 0]
    MINOR_RELEASE_TEST = [3, 5, 0]
    BUGFIX_RELEASE_TEST = [3, 4, 6]

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

    def test_version_less_then_operator_true(self):
        for case in self.LESS_THEN_TEST_TRUE:
            reference = Version(case)
            self.assertTrue(self.version < reference, msg='Object: %s is expected to be less then %s' %
                                                          (safe_repr(self.version), safe_repr(reference)))

    def test_version_less_then_operator_false(self):
        for case in self.LESS_THEN_TEST_FALSE:
            reference = Version(case)
            self.assertFalse(self.version < reference, msg='Object: %s is expected not to be less then %s' %
                                                           (safe_repr(self.version), safe_repr(reference)))
            
    def test_str_converion(self):
        self.assertIsInstance(str(self.version), str, msg='The Version class is expected to be str convertible.')

    def test_major_release(self):
        self.version.release_major_version()
        expected_reference = str(self.MAJOR_RELEASE_TEST[0]) + '.' + \
                             str(self.MAJOR_RELEASE_TEST[1]) + '.' + \
                             str(self.MAJOR_RELEASE_TEST[2])
        self.assertTrue(self.version.major_version == self.MAJOR_RELEASE_TEST[0] and
                        self.version.minor_version == self.MAJOR_RELEASE_TEST[1] and
                        self.version.bugfix_version == self.MAJOR_RELEASE_TEST[2],
                        msg='Major release of object %s is expected to be %s.' %
                            (safe_repr(self.version), safe_repr(expected_reference)))

    def test_minor_release(self):
        self.version.release_minor_version()
        expected_reference = str(self.MINOR_RELEASE_TEST[0]) + '.' + \
                             str(self.MINOR_RELEASE_TEST[1]) + '.' + \
                             str(self.MINOR_RELEASE_TEST[2])
        self.assertTrue(self.version.major_version == self.MINOR_RELEASE_TEST[0] and
                        self.version.minor_version == self.MINOR_RELEASE_TEST[1] and
                        self.version.bugfix_version == self.MINOR_RELEASE_TEST[2],
                        msg='Major release of object %s is expected to be %s.' %
                            (safe_repr(self.version), safe_repr(expected_reference)))

    def test_bugfix_release(self):
        self.version.release_bugfix_version()
        expected_reference = str(self.BUGFIX_RELEASE_TEST[0]) + '.' + \
                             str(self.BUGFIX_RELEASE_TEST[1]) + '.' + \
                             str(self.BUGFIX_RELEASE_TEST[2])
        self.assertTrue(self.version.major_version == self.BUGFIX_RELEASE_TEST[0] and
                        self.version.minor_version == self.BUGFIX_RELEASE_TEST[1] and
                        self.version.bugfix_version == self.BUGFIX_RELEASE_TEST[2],
                        msg='Major release of object %s is expected to be %s.' %
                            (safe_repr(self.version), safe_repr(expected_reference)))


class CodeInfoTest(TestCase):

    def setUp(self):
        self.info = CodeInfo()

    def tearDown(self):
        del self.info
