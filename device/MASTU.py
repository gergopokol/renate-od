import numpy as np
from scipy.interpolate import RegularGridInterpolator
from utility.geometrical_objects import Point
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py


# 2Dprofile[r,z]


class PSFforMASTU:

    def __init__(self, efit_path, los_list, psf_plane):
        self.efit_data = self.read_MASTU_EFIT(efit_path)
        self.psf_plane = psf_plane
        self.los_list = los_list

    def read_MASTU_EFIT(self, path):
        f = h5py.File(path)
        data_dict = {}
        data_dict['time'] = f['epm']['time'][...]
        print('time'+': '+str(data_dict['time'].shape))
        for key in list(f['epm']['output']['profiles2D'].keys()):
            data_dict[key] = f['epm']['output']['profiles2D'][key][...]
            print(key+': '+str(data_dict[key].shape))
        f.close()
        return data_dict

    def make_interpolators_at_time(self, time):
        time_ind = np.where(self.efit_data['time'] <= time)[-1][-1]
        self.Br_interpolator = RegularGridInterpolator((self.efit_data['r'], self.efit_data['z']),
                                                       self.efit_data['Br'][time_ind])
        self.Bz_interpolator = RegularGridInterpolator((self.efit_data['r'], self.efit_data['z']),
                                                       self.efit_data['Bz'][time_ind])
        self.Bphi_interpolator = RegularGridInterpolator((self.efit_data['r'], self.efit_data['z']),
                                                         self.efit_data['Bphi'][time_ind])

    def prepare_psf(self, time, beam, beam_interpolator, propagation='ccw', stepsize=1e-3):
        if propagation == 'ccw':
            self.prop_sign = 1
        elif propagation == 'cw':
            self.prop_sign = -1
        else:
            raise ValueError(
                'Propagation is either \'ccw\' (default) or \'cw\'')
        self.beam_interpolator = beam_interpolator
        self.stepsize = stepsize
        self.make_interpolators_at_time(time)
        self.ps_curves = []
        self.ps_weights = []
        for los in self.los_list:
            curve, weights = self.process_los(los)
            self.ps_curves.append(curve)
            self.ps_weights.append(weights)

    def process_los(self, los):
        int_p, int_w = los.interpolate_points(self.beam_interpolator)
        propagated = []
        weights = []
        for i, p in enumerate(int_p):
            point = Point(p)
            if (point.phi-self.psf_plane.origin.phi)*self.prop_sign < 0:
                propagated.append(self.propagate_los_point(Point(p)))
                weights.append(int_w[i])
        propagated = np.array(
            propagated, dtype=[('r', np.float64), ('z', np.float64), ('phi', np.float64)])
        return propagated, np.array(weights)

    def propagate_los_point(self, point):
        phi_old = point.phi
        z = point.z
        r = point.r
        b = self.get_b_vector(r, z)
        b = b/np.linalg.norm(b)*self.stepsize
        d = self.prop_sign*np.sign(b[2])
        b = b*d
        phi_new = phi_old+b[2]
        while(np.abs(phi_old-self.psf_plane.origin.phi) > np.abs(phi_new-self.psf_plane.origin.phi)):
            z = z+b[1]
            r = r+b[0]
            phi_old = phi_new
            b = self.get_b_vector(r, z)
            b = b/np.linalg.norm(b)*self.stepsize*d
            phi_new = phi_old+b[2]
        return r, z, phi_old

    def get_b_vector(self, r, z):
        br = self.Br_interpolator((r, z))
        bz = self.Bz_interpolator((r, z))
        bphi = self.Bphi_interpolator((r, z))
        return np.array([br, bz, bphi])

    def plot_detectors(self, dimensions):
        if len(dimensions) != len(self.los_list):
            raise ValueError(
                'Number of detectors not equal to number of LOSs.')
        for i, los in enumerate(self.los_list):
            center = self.psf_plane.transform_to_plane(los.end.cartesians)
            width, height = dimensions[i]
            plt.gca().add_patch(
                Rectangle((center[0]-width/2, center[1]-height/2), width, height,
                          edgecolor='k',
                          fill=False))

    def plot_curves(self):
        if hasattr(self, 'ps_curves'):
            cmap = mpl.cm.viridis
            for i, curve in enumerate(self.ps_curves):
                maxw = np.max(self.ps_weights[i])
                for j, p in enumerate(curve):
                    point = Point(cylindrical=(p[0], p[2], p[1]))
                    x, y = self.psf_plane.transform_to_plane(
                        (point.cartesians))[:2]
                    plt.scatter(x, y, color=cmap(self.ps_weights[i][j]/maxw))
