import os
import unittest
from utility.accessdata import AccessData


class AccessDataTest(unittest.TestCase):

    TEST_PATH = 'test_dataset/access_tests/'
    PUBLIC_TEST = 'public_test'
    PRIVATE_TEST = 'private_test'
    PRIVATE_SERVES_ADDRESS = 'data@deep.reak.bme.hu:~/private_html/renate-od'
    PUBLIC_SERVER_ADDRESS = 'http://deep.reak.bme.hu/~data/renate-od'
    DUMMY_FOLDER = 'dummy'
    CONTACT_INFO = 'pokol@reak.bme.hu'

    def setUp(self):
        self.access = AccessData(None)

    def tearDown(self):
        del self.access

    def test_private_data_path(self):
        self.assertEqual(self.access.server_private_address, self.PRIVATE_SERVES_ADDRESS,
                         msg='Server private data address does not match expected server private address.')

    def test_public_server_address(self):
        self.assertEqual(self.access.server_public_address, self.PUBLIC_SERVER_ADDRESS,
                         msg='Server public data address does not match expected server public address.')

    def test_dummy_folder_name(self):
        self.assertEqual(self.access.dummy_directory, self.DUMMY_FOLDER,
                         msg='Dummy folder name does not match expected dummy folder name.')

    def test_contact_information(self):
        self.assertEqual(self.access.contact_address, self.CONTACT_INFO,
                         msg='Contact information does not match expected contact information.')

    def test_user_local_directory(self):
        self.assertEqual(self.access.user_local_data_directory, os.path.join(os.path.dirname(__file__), '..', 'data'),
                         msg='User local data path does not match expected user local data path.')

    def test_user_local_common_directory(self):
        self.assertEqual(self.access.common_local_data_directory, os.path.join(os.path.dirname(__file__), '..',
                         'common_data'), msg='User local common data path does not match expected user local '
                                             'common data path.')
