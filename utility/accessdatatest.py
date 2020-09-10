import unittest
from utility.accessdata import AccessData


class AccessDataTest(unittest.TestCase):

    TEST_PATH = 'test_dataset/access_tests/'
    PUBLIC_TEST = 'public_test'
    PRIVATE_TEST = 'private_test'

    def setUp(self):
        self.access = AccessData(None)

    def tearDown(self):
        del self.access
