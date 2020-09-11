import os
import unittest
from utility.accessdata import AccessData
from utility.accessdata import DEFAULT_SETUP


class AccessDataTest(unittest.TestCase):

    TEST_PATH = 'test_dataset/access_tests/'
    PUBLIC_TEST = 'public_test.txt'
    PRIVATE_TEST = 'private_test.txt'
    PRIVATE_SERVES_ADDRESS = 'data@deep.reak.bme.hu:~/private_html/renate-od'
    PUBLIC_SERVER_ADDRESS = 'http://deep.reak.bme.hu/~data/renate-od'
    DUMMY_FOLDER = 'dummy'
    CONTACT_INFO = 'pokol@reak.bme.hu'
    PRIVATE_KEY = 'data/deep-data.ppk'

    def setUp(self):
        self.access = AccessData(None)

    def tearDown(self):
        del self.access

    def test_private_server_address(self):
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

    def test_private_key(self):
        self.assertEqual(self.access.private_key, self.PRIVATE_KEY, msg='Default private key does not match expected '
                                                                        'default private key.')

    def test_server_path_setup(self):
        path = self.TEST_PATH + '/' + self.PRIVATE_TEST
        self.access.server_path_setup(server_path=path)
        self.assertEqual(self.access.server_private_path, self.access.server_private_address + '/' + path,
                         msg='Server private path does not match expected server private path.')
        self.assertEqual(self.access.server_public_path, self.access.server_public_address + '/' + path,
                         msg='Server public path does not match expected server public path.')

    def test_local_path_setup(self):
        path = self.TEST_PATH + '/' + self.PRIVATE_TEST
        self.access.local_path_setup(local_path=path)
        self.assertEqual(self.access.common_local_data_path,
                         os.path.join(self.access.common_local_data_directory, path),
                         msg='Actual common local path does not match expected common local path.')
        self.assertEqual(self.access.user_local_data_path, os.path.join(self.access.user_local_data_directory, path),
                         msg='Actual user local path does not match expected user local path.')
        self.assertEqual(self.access.user_local_dummy_path, os.path.join(self.access.user_local_data_directory,
                         self.access.dummy_directory, path), msg='Actual user local dummy path does not match expected '
                                                                 'user local dummy path.')

    def test_public_server_data_check(self):
        self.access.server_path_setup(self.TEST_PATH + self.PUBLIC_TEST)
        self.assertTrue(self.access.check_public_server_data_path(), msg='The datafile: ' + self.PUBLIC_TEST +
                        ' is supposed to be located on the public server at: ' + self.TEST_PATH)
        self.access.server_path_setup(self.TEST_PATH + self.PRIVATE_TEST)
        self.assertFalse(self.access.check_public_server_data_path(), msg='The datafile: ' + self.PRIVATE_TEST +
                         ' is not supposed to be located on the public server at: ' + self.TEST_PATH)

    def test_private_server_data_check(self):
        self.access.local_path_setup(self.access.private_key)
        if self.access.check_user_local_data_path():
            pass

    def test_common_local_data_check(self):
        self.access.common_local_data_directory = os.path.dirname(__file__)
        self.access.local_path_setup(local_path=DEFAULT_SETUP)
        self.assertTrue(self.access.check_common_local_data_path(), msg='DEFAULT SETUP file is expected to exist '
                                                                        'in the location specified.')
        self.assertEqual(self.access.access_path, os.path.join(os.path.dirname(__file__), DEFAULT_SETUP),
                         msg='The actual data path to be accessed does not match to the expected access data path.')
        self.access.local_path_setup(self.PRIVATE_KEY)
        self.assertFalse(self.access.check_common_local_data_path(),
                         msg='PRIVATE_TEST file is expected not to be located in the specified data access path.')

    def test_user_local_data_check(self):
        self.access.user_local_data_directory = os.path.dirname(__file__)
        self.access.local_path_setup(local_path=DEFAULT_SETUP)
        self.assertTrue(self.access.check_user_local_data_path(), msg='DEFAULT SETUP file is expected to exist '
                                                                      'in the location specified.')
        self.assertEqual(self.access.access_path, os.path.join(os.path.dirname(__file__), DEFAULT_SETUP),
                         msg='The actual data path to be accessed does not match to the expected access data path.')
        self.access.local_path_setup(self.PRIVATE_KEY)
        self.assertFalse(self.access.check_user_local_data_path(),
                         msg='PRIVATE_TEST file is expected not to be located in the specified data access path.')

    def test_user_local_dummy_data_check(self):
        self.access.user_local_data_directory = os.path.dirname(__file__)
        self.access.dummy_directory = ''
        self.access.local_path_setup(local_path=DEFAULT_SETUP)
        self.assertTrue(self.access.check_user_local_dummy_path(), msg='DEFAULT SETUP file is expected to exist '
                                                                       'in the location specified.')
        self.access.local_path_setup(self.PRIVATE_KEY)
        self.assertFalse(self.access.check_user_local_dummy_path(),
                         msg='PRIVATE_TEST file is expected not to be located in the specified data access path.')
