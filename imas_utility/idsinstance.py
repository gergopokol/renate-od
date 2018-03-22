import pyual
import os


class ImasObject:
    def __init__(self, shot, run, user=None, machine=None):
        assert isinstance(shot, int)
        self.shot = shot

        assert isinstance(run, int)
        self.run = run

        if user is None:
            self.user = os.getenv('USER')
            self.local_data_base = False
        else:
            assert isinstance(user, str)
            self.user = user
            self.local_data_base = True

        if machine is None:
            self.machine = 'iter'
        else:
            assert isinstance(machine, str)
            self.machine = machine

        self.build_imas_pointer()

    def build_imas_pointer(self):
        try:
            if self.local_data_base is True:
                self.imas_pointer = pyual.Client(self.shot, self.run, user=self.user, machine=self.machine)
            else:
                self.imas_pointer = pyual.Client(self.shot, self.run)
        except:
            print, 'Shot ' + str(self.shot) + ', run ' + str(self.run) + ' could not be opened'
            print, 'IDS load ----> Aborted. Check for the existence of IDS in one of the following locations:'
            print, '1. /work/imas/shared/'+self.machine+'db/3/0/....'
            print, '2. /home/ITER/user/public/imasdb/....'
            exit()
