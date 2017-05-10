import h5py


def get_data_from_hdf5(name, source):
    try:
        hdf5_id = h5py.File(name, 'r')
    except IOError:
        print("File not found!", name)
        quit()
    else:
        data = hdf5_id[source].value
        hdf5_id.close()
        return data
