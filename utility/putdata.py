from utility.accessdata import AccessData
from scp import SCPException


class PutData(AccessData):
    def __init__(self):
        AccessData.__init__(self, data_path_name=None)

    def to_server(self, local_path, server_path=None, server_type='public'):
        if server_path is None:
            if isinstance(local_path, str):
                self.data_path_name = local_path
                self._path_setup()
            else:
                raise TypeError('<local_path> is expected to be of type str. '
                                'The provided data type is: '+str(type(local_path)))
        elif isinstance(server_path, str):
            self._server_path_setup(server_path=server_path)
            self._local_path_setup(local_path=local_path)
        else:
            raise TypeError('<server_path> is expected to be of type str.'
                            'The provided data type is: '+str(type(server_path)))
        if self.check_user_local_data_path():
            if not isinstance(server_type, str):
                raise TypeError('The requested server type is a str information. Provided data is: '
                                + str(type(server_type)))
            if server_type == 'public':
                if self.check_public_server_data_path():
                    print('The file: ' + self.server_public_path + ' is already on the public server.')
                else:
                    self.connect(protocol='scp')
                    try:
                        self.scp.put(self.user_local_data_path, self.server_public_path, recursive=True)
                        print('Successfully placed: ' + self.user_local_data_path + ' to public server location: '
                              + self.server_public_path)
                    except SCPException:
                        print('Could not put file: ' + self.user_local_data_path + ' to public server location: '
                              + self.server_public_path)
                    finally:
                        self.disconnect()
            elif server_type == 'private':
                if self.check_private_server_data_path():
                    print('The file: ' + self.server_private_path + ' is already on the public server.')
                else:
                    self.connect(protocol='scp')
                    try:
                        self.scp.put(self.user_local_data_path, self.server_private_path, recursive=True)
                        print('Successfully placed: ' + self.user_local_data_path + ' to private server location: '
                              + self.server_private_path)
                    except SCPException:
                        print('Could not put file: ' + self.user_local_data_path + ' to private server location: '
                              + self.server_private_path)
                    finally:
                        self.disconnect()
            else:
                raise ValueError('The requested server type <'+server_type+'> '
                                                                           'where data is to be ported does not exist!')
        else:
            raise FileNotFoundError('There is no local data at: ' + self.user_local_data_path)

    def delete_from_server(self, data_path, server_type='public'):
        if isinstance(data_path, str):
            self.data_path_name = data_path
            self.path_setup()
        else:
            raise TypeError('<data_path> is expected to be of type str. '
                            'The provided data type is: '+str(type(data_path)))

        if not isinstance(server_type, str):
            raise TypeError('The requested server type is a str information. Provided data is: '+str(type(server_type)))

        if server_type == 'public':
            return
        elif server_type == 'private':
            return
        else:
            raise ValueError('The requested server type <'+server_type+'> from which data is to be deleted '
                                                                       'does not exist!')

    def move_file(self, from_path, to_path, server_type='public'):
        pass
