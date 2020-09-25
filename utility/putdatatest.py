import unittest
from utility.accessdata import AccessData
from utility.putdata import PutData
from utility.getdata import GetData


class PutDataTest(unittest.TestCase):

    PRIVATE_DOWNLOAD_TEST = 'test_dataset/access_tests/private_test.xml'
    PUBLIC_DOWNLOAD_TEST = 'test_dataset/access_tests/public_test.xml'
    UPLOAD_TEST = 'test_dataset/private_test.xml'

    def setUp(self):
        self.put = PutData()

    def tearDown(self):
        del self.put

    def test_1_parentage(self):
        self.assertIsInstance(self.put, AccessData, msg='The PutData object is expected to be the '
                                                        'child of the AccessData object.')

    def test_2_upload_to_private(self):
        if self.put.private_key is not None:
            download = GetData(data_path_name=self.PRIVATE_DOWNLOAD_TEST)
            self.put.to_server(self.PRIVATE_DOWNLOAD_TEST, server_path=self.UPLOAD_TEST, server_type='private')
            self.assertTrue(self.put.check_private_server_data_path(self.put.server_private_path), msg='Private upload '
                            'test file expected to be located at: ' + self.put.server_private_path)

    def test_3_delete_from_private(self):
        if self.put.private_key is not None:
            self.put.delete_from_server(self.UPLOAD_TEST, server_type='private')
            self.assertFalse(self.put.check_private_server_data_path(self.put.server_private_path), msg='Private server'
                             ' delete routine expected to remove test file from private server location: ' +
                             self.UPLOAD_TEST)

    def test_4_upload_to_public(self):
        if self.put.private_key is not None:
            download = GetData(data_path_name=self.PRIVATE_DOWNLOAD_TEST)
            self.put.to_server(self.PRIVATE_DOWNLOAD_TEST, server_path=self.UPLOAD_TEST, server_type='public')
            self.assertTrue(self.put.check_private_server_data_path(self.put.server_to_public_path),
                            msg='Public upload test file expected to be located at: ' +
                                self.put.server_to_public_path)

    def test_5_delete_from_public(self):
        if self.put.private_key is not None:
            self.put.delete_from_server(self.UPLOAD_TEST, server_type='public')
            self.assertFalse(self.put.check_private_server_data_path(self.put.server_to_public_path),
                             msg='Public server delete routine expected to remove test file from public server '
                                 'location: ' + self.UPLOAD_TEST)

    def test_6_move_private_to_public(self):
        if self.put.private_key is not None:
            self.put.move_data(self.PRIVATE_DOWNLOAD_TEST, self.PRIVATE_DOWNLOAD_TEST,
                               data_migration='private-to-public')
            self.assertFalse(self.put.check_private_server_data_path(path=self.put._set_private_server_path
                                                                     (self.PRIVATE_DOWNLOAD_TEST)),
                             msg='Test file is expected to be moved from the private data storage: ' +
                                 self.put._set_private_server_path(self.PRIVATE_DOWNLOAD_TEST))
            self.assertTrue(self.put.check_private_server_data_path(path=self.put._set_public_server_access_path
                                                                    (self.PRIVATE_DOWNLOAD_TEST)),
                            msg='Test file is expected to be moved to the public data storage: ' +
                                self.put._set_public_server_access_path(self.PRIVATE_DOWNLOAD_TEST))

    def test_7_move_public_to_private(self):
        if self.put.private_key is not None:
            self.put.move_data(self.PRIVATE_DOWNLOAD_TEST, self.PRIVATE_DOWNLOAD_TEST,
                               data_migration='public-to-private')
            self.assertFalse(self.put.check_private_server_data_path(path=self.put._set_public_server_access_path
                                                                     (self.PRIVATE_DOWNLOAD_TEST)),
                             msg='Test file is expected to be moved from the public data storage: ' +
                                 self.put._set_public_server_access_path(self.PRIVATE_DOWNLOAD_TEST))
            self.assertTrue(self.put.check_private_server_data_path(path=self.put._set_private_server_path
                                                                    (self.PRIVATE_DOWNLOAD_TEST)),
                            msg='Test file is expected to be moved to the private data storage: ' +
                                self.put._set_private_server_path(self.PRIVATE_DOWNLOAD_TEST))
