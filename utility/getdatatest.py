import os
import unittest
from shutil import rmtree
from lxml.etree import _ElementTree
from utility.getdata import GetData
from utility.accessdata import AccessData


class GetDataTest(unittest.TestCase):

    PRIVATE_DOWNLOAD_TEST = 'test_dataset/access_tests/private_test.xml'
    PUBLIC_DOWNLOAD_TEST = 'test_dataset/access_tests/public_test.xml'
    LOAD_TXT_TEST = 'test_dataset/access_tests/public_test.txt'
    HDF5_TEST = 'test_dataset/access_tests/public_test.h5'

    def setUp(self):
        self.access = AccessData(None)

    def tearDown(self):
        public_folder = os.path.join(os.getcwd(), 'data', 'dummy', 'test_dataset')
        private_folder = os.path.join(os.getcwd(), 'data', 'test_dataset')
        if os.path.isdir(public_folder):
            rmtree(public_folder)
        if os.path.isdir(private_folder):
            rmtree(private_folder)
        del self.access
        del self.data

    def test_parentage(self):
        self.data = GetData(self.PUBLIC_DOWNLOAD_TEST)
        self.assertIsInstance(self.data, AccessData, msg='The GetData object is expected to a child of AccessData.')

    def test_private_data_download(self):
        if self.access.private_key is not None:
            self.data = GetData(self.PRIVATE_DOWNLOAD_TEST)
            self.assertTrue(os.path.exists(os.path.join(os.getcwd(), 'data', self.PRIVATE_DOWNLOAD_TEST)),
                            msg='The private download test case expected private test data to be downloaded in user '
                                'private data folder.')

    def test_public_data_download(self):
        self.data = GetData(self.PUBLIC_DOWNLOAD_TEST)
        self.assertTrue(os.path.exists(os.path.join(os.getcwd(), 'data', 'dummy', self.PUBLIC_DOWNLOAD_TEST)),
                        msg='The public download test case expected public test data to be downloaded in user '
                            'dummy, public data folder.')

    def test_XML_data_loader(self):
        self.data = GetData(self.PUBLIC_DOWNLOAD_TEST).data
        self.assertIsInstance(self.data, _ElementTree, msg='The data type for loading .xml type of data is '
                                                           'expected to be of <_ElementTree>')

    def test_str_loader_from_txt(self):
        self.data = GetData(self.LOAD_TXT_TEST).data
        self.assertIsInstance(self.data, str, msg='The data type for loading .txt type of data is '
                                                  'expected to be of <str>')
