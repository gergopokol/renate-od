from unittest import TestCase
from utility.manage import Version
from unittest.util import safe_repr


class VersionTest(TestCase):

    EQUAL_TEST = '3.4.5'
    NOTEQUALTEST = '3.3.5'

    def setUp(self):
        self.version = Version('3.4.5')

    def tearDown(self):
        del self.version

    def test_version_equality_operator(self):
        reference = Version(self.EQUAL_TEST)
        self.assertTrue(self.version == reference, msg='Objects: %s and %s are expected to be equal' %
                                                       (safe_repr(self.version), safe_repr(reference)))

    def test_version_inequality_operator(self):
        reference = Version(self.NOTEQUALTEST)
        self.assertTrue(self.version != reference, msg='Objects: %s and %s are expected not to be equal' %
                                                       (safe_repr(self.version), safe_repr(reference)))

