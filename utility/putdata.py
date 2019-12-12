from utility.accessdata import AccessData


class PutData(AccessData):
    def __init__(self):
        AccessData.__init__(self, data_path_name=None)

    def to_server(self, local_path, server_path=None, server_type='public'):
        if server_path is None:
            if isinstance(local_path, str):
                self.data_path_name = local_path
                self.path_setup()
            else:
                raise TypeError('<local_path> is expected to be of type str. '
                                'The provided data type is: '+str(type(local_path)))
        elif isinstance(server_path, str):
            self.server_path_setup(server_path)
            self.local_path_setup(local_path)
        else:
            raise TypeError('<server_path> is expected to be of type str.'
                            'The provided data type is: '+str(type(server_path)))

        if not isinstance(server_type, str):
            raise TypeError('The requested server type is a str information. Provided data is: '+str(type(server_type)))

        if server_type == 'public':
            return
        elif server_type == 'private':
            return
        else:
            raise ValueError('The requested server type <'+server_type+'> where data is to be ported does not exist!')

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
