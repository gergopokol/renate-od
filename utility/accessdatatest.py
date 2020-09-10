import unittest
from utility.accessdata import AccessData


class AccessDataTest(unittest.TestCase):

    TEST_PATH = 'test_dataset/access_tests/'
    PUBLIC_TEST = 'public_test'
    PRIVATE_TEST = 'private_test'
    PRIVATE_SERVES_ADDRESS = 'data@deep.reak.bme.hu:~/private_html/renate-od'
    PUBLIC_SERVER_ADDRESS = 'http://deep.reak.bme.hu/~data/renate-od'

    def setUp(self):
        self.access = AccessData(None)

    def tearDown(self):
        del self.access

    def test_private_data_path(self):
        self.assertEqual(self.access.server_private_address, self.PRIVATE_SERVES_ADDRESS,
                         msg='Server private data address does not match.')

    def test_public_server_address(self):
        self.assertEqual(self.access.server_public_address, self.PUBLIC_SERVER_ADDRESS,
                         msg='Server public data address does not match.')


