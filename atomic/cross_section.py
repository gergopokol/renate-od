from atomic.tools import Transition
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc

"""
Li references: 
[1] Wutte et.al. ADNDT 65, 155 (1997) https://www.sciencedirect.com/science/article/abs/pii/S0092640X97907361
[2] Schweinzer et.al. ADNDT 72, 239 (1999)  https://www.sciencedirect.com/science/article/abs/pii/S0092640X9990815X
[3] Schweinzer et.al. JPB 27, 137 (1994) https://iopscience.iop.org/article/10.1088/0953-4075/27/1/017
"""

ATOMIC_DICT = {'Li': {'e': {'2s-eloss': {'param': [5.39, 0.085, -0.004, 0.757, -0.178], 'eq': '0'},  # [1] Table p165
                            'w2s-2p': {'param': [1.847, 4.1818, -27.335, 89.34, 0, 52.788, 1.1971], 'eq': '2'},
                            'w2s-3s': {'param': [3.372, 4.7015, -4.6972, -3.0504, 11.332, 0, 0.51449], 'eq': '2'},
                            'w2s-3p': {'param': [3.833, 1.4705, 1.5511, 0.94339, 0, 0.12721, 0.53112], 'eq': '2'},
                            'w2s-3d': {'param': [3.877, 8.2190, 18.868, -51.214, 55.412, 0, 1.4119], 'eq': '2'},
                            '2s-2p': {'param': [1.847, -29.466, 10.106, 128.32, -86.415, 58.696, 0.77758], 'eq': '2'},
                            '2s-3s': {'param': [3.372, 4.50, -11.393, 9.2699, 1.8822, 0, 0.32060], 'eq': '2'},
                            '2s-3p': {'param': [3.833, 2.2138, -4.3282, 4.4758, 0, 0, 0.25068], 'eq': '2'},
                            '2s-3d': {'param': [3.877, 8.20, -8.2458, -6.4716, 12.953, 0, 0.61774], 'eq': '2'},
                            '2s-4s': {'param': [4.34, 0.926, -1.1803, -1.2032, 2.0279, 0, 0.053105], 'eq': '2'},
                            '2s-4p': {'param': [4.52, 0.05455, 1.0983, -1, 0.1222, 0.15, 0.12917], 'eq': '2'},
                            '2s-4d': {'param': [4.539, 1.8676, -1.8193, 0.2679, 0, 0, 0.25478], 'eq': '2'},
                            '2s-4f': {'param': [4.54, 0.25, 3.3341, -6.5182, 4.9176, 0, 0.90017], 'eq': '2'},
                            '2p-eloss': {'param': [3.543, 0.075347, 0.1882, -0.28343, 1.5382, -1.855, 1.3411], 'eq': '1'},
                            'w2p-eloss': {'param': [3.541], 'eq': '3'},
                            'w2p-3s': {'param': [1.525, 1.0693e-2, -2.4093, 5.7497, 0, 8.0951, 0.55336], 'eq': '2'},
                            'w2p-3p': {'param': [1.986, 10.799, -7.9161, 5.6946, 0, 0, 0.60959], 'eq': '2'},
                            'w2p-3d': {'param': [2.030, 3.8735e-3, 37.646, -5.2622, 0, 36.321, 1.0935], 'eq': '2'},
                            '2p-3s': {'param': [1.525, -12.287, 7.9788, 21.593, -10.454, 10.136, 0.35903], 'eq': '2'},
                            '2p-3p': {'param': [1.986, 10.353, -22.590, 21.347, 0, 0, 0.33146], 'eq': '2'},
                            '2p-3d': {'param': [2.03, -20.091, -78.831, 306.52, -146.03, 38.767, 1.7130], 'eq': '2'},
                            '2p-4s': {'param': [2.493, -0.45472, 6.2003, -23.437, 25.556, 0.70915, 1.1250], 'eq': '2'},
                            '2p-4p': {'param': [2.673, 2.4654, -3.9177, 4.7847, 0, 0, 0.78931], 'eq': '2'},
                            '2p-4d': {'param': [2.692, 4.8531, -40.566, 83.492, -46.007, 4.5560, 0.57257], 'eq': '2'},
                            '2p-4f': {'param': [2.693, 2.1425, 24.042, -77.016, 62.941, 0.24507, 0.79098], 'eq': '2'},
                            '3s-eloss': {'param': [2.018, 5.55e-15, 0.34812, -0.29262, 0.34264, 0.99613, -0.82395], 'eq': '1'},
                            'w3s-eloss': {'param': [2.018], 'eq': '3'},
                            'w3s-3p': {'param': [0.461, 8.4471e-2, 225.8, 0, 0, 322.51, 1.0632], 'eq': '2'},
                            'w3s-3d': {'param': [0.505, 91.515, -48.338, 127.34, 0, 0, 1.0738], 'eq': '2'},
                            '3s-3p': {'param': [0.461, -678.58, 805.29, 0, 0, 439.5, 0.78903], 'eq': '2'},
                            '3s-3d': {'param': [0.505, 96.957, -309.15, 1089.2, 0, 0, 2.4374], 'eq': '2'},
                            '3s-4s': {'param': [0.968, 18.409, -72.121, 34.575, 109.59, 0.94483, 1.2021], 'eq': '2'},
                            '3s-4p': {'param': [1.148, 8.4345, -10.284, 0, 25.887, 1.9035, 0.62176], 'eq': '2'},
                            '3s-4d': {'param': [1.167, 10.233, 407.74, -2173.8, 3286.7, 0, 3.7253], 'eq': '2'},
                            '3s-4f': {'param': [1.168, 28.327, 100.47, -421.76, 419.5, 0, 1.00], 'eq': '2'},
                            '3p-eloss': {'param': [1.557, 0.11218, 0.24966, -1.0074, 1.8549, -1.1973, 0.48164], 'eq': '1'},
                            'w3p-eloss': {'param': [1.557], 'eq': '3'},
                            'w3p-3d': {'param': [0.044, 3.06044e-2, 25.686, 156.57, 0, 260.22, 1.034], 'eq': '2'},
                            '3p-3d': {'param': [0.044, 186.96, -1032.5, 2307.1, 0, 230.05, 2.68], 'eq': '2'},
                            '3p-4s': {'param': [0.507, -117.31, 196.06, -78.658, 0, 64.033, 0.062851], 'eq': '2'},
                            '3p-4p': {'param': [0.687, 49.0, -218.58, 392.42, -201.84, 0, 0.61809], 'eq': '2'},
                            '3p-4d': {'param': [0.706, -202.18, 335.8, -132.6, 0, 113.64, 0.28384], 'eq': '2'},
                            '3p-4f': {'param': [0.707, 115.93, -99.313, -319.37, 758.96, 0, 2.3647], 'eq': '2'},
                            '3d-eloss': {'param': [1.513, 0.02339, 0.37229, 0.011639, -3.1118, 7.7385, -4.5292], 'eq': '1'},
                            'w3d-eloss': {'param': [1.513], 'eq': '3'},
                            '3d-4s': {'param': [0.463, 5.80, -13.713, 54.729, -44.469, 0, 0.69859], 'eq': '2'},
                            '3d-4p': {'param': [0.643, 2.2291, -10.245, 42.434, -28.528, 2.2158, 0.21465], 'eq': '2'},
                            '3d-4d': {'param': [0.662, 43.0, -142.81, 226.64, -110.26, 0, 0.6948], 'eq': '2'},
                            '3d-4f': {'param': [0.663, -221.53, -371.82, 3313.2, -1906.1, 221.29, 3.7250], 'eq': '2'},
                            '4s-eloss': {'param': [1.05], 'eq': '3'},
                            '4s-4p': {'param': [0.18, 0.072489, 816.84, 0, 0, 992.70, 1.1539], 'eq': '2'},
                            '4s-4d': {'param': [0.199, 384.65, -489.77, 382.85, 0, 0, 0.51756], 'eq': '2'},
                            '4s-4f': {'param': [0.2, 492.35, -626.91, 490.05, 0, 0, 0.51756], 'eq': '2'},
                            '4p-eloss': {'param': [0.87], 'eq': '3'},
                            '4p-4d': {'param': [0.019, 0.11050, 126.57, 382.62, 0, 778.86, 0.92249], 'eq': '2'},
                            '4p-4f': {'param': [0.02, 538.51, -685.678, 535.99, 0, 0, 0.51756], 'eq': '2'},
                            '4d-eloss': {'param': [0.851], 'eq': '3'},
                            '4d-4f': {'param': [0.001, 0.1547, 177.198, 535.668, 0, 1090.4, 0.92249], 'eq': '2'},
                            '4f-eloss': {'param': [0.85], 'eq': '3'}},
                      'p': {'2s-eloss': {},
                            '2s-2p': {},
                            '2s-3s': {},
                            '2s-3p': {},
                            '2s-3d': {},
                            '2s-4s': {},
                            '2s-4p': {},
                            '2s-4d': {},
                            '2s-4f': {},
                            '2p-eloss': {},
                            '2p-3s': {},
                            '2p-3p': {},
                            '2p-3d': {},
                            '2p-4s': {},
                            '2p-4p': {},
                            '2p-4d': {},
                            '2p-4f': {},
                            '3s-eloss': {},
                            '3s-3p': {},
                            '3s-3d': {},
                            '3s-4s': {},
                            '3s-4p': {},
                            '3s-4d': {},
                            '3s-4f': {},
                            '3p-eloss': {},
                            '3p-3d': {},
                            '3p-4s': {},
                            '3p-4p': {},
                            '3p-4d': {},
                            '3p-4f': {},
                            '3d-eloss': {},
                            '3d-4s': {},
                            '3d-4p': {},
                            '3d-4d': {},
                            '3d-4f': {},
                            '4s-eloss': {},
                            '4s-4p': {},
                            '4s-4d': {},
                            '4s-4f': {},
                            '4p-eloss': {},
                            '4p-4d': {},
                            '4p-4f': {},
                            '4d-eloss': {},
                            '4d-4f': {},
                            '4f-eloss': {}},
                      'Z': {'2s-eloss': {},
                            '2s-2p': {},
                            '2s-3s': {},
                            '2s-3p': {},
                            '2s-3d': {},
                            '2s-4s': {},
                            '2s-4p': {},
                            '2s-4d': {},
                            '2s-4f': {},
                            '2p-eloss': {},
                            '2p-3s': {},
                            '2p-3p': {},
                            '2p-3d': {},
                            '2p-4s': {},
                            '2p-4p': {},
                            '2p-4d': {},
                            '2p-4f': {},
                            '3s-v': {},
                            '3s-eloss': {},
                            '3s-3d': {},
                            '3s-4s': {},
                            '3s-4p': {},
                            '3s-4d': {},
                            '3s-4f': {},
                            '3p-eloss': {},
                            '3p-3d': {},
                            '3p-4s': {},
                            '3p-4p': {},
                            '3p-4d': {},
                            '3p-4f': {},
                            '3d-eloss': {},
                            '3d-4s': {},
                            '3d-4p': {},
                            '3d-4d': {},
                            '3d-4f': {},
                            '4s-eloss': {},
                            '4s-4p': {},
                            '4s-4d': {},
                            '4s-4f': {},
                            '4p-eloss': {},
                            '4p-4d': {},
                            '4p-4f': {},
                            '4d-eloss': {},
                            '4d-4f': {},
                            '4f-eloss': {}}},
               'H': {'e': {},
                     'p': {},
                     'Z': {}},
               'Na': {'e': {},
                      'p': {},
                      'Z': {}}
               }

CROSS_FUNC = {'0': lambda x, par: 1e-13*(par[1]*np.log(x/par[0]) + par[2]*(1-(par[0]/x)) +
                                         par[3]*(1-(par[0]/x))**2 + par[4]*(1-(par[0]/x))**3)/(par[0]*x),
              # [1] eq. on p165, electron - impact ionization for 2s
              '1': lambda x, par: 1e-13*(par[1]*np.log(x/par[0]) + par[2]*(1-(par[0]/x)) +
                                         par[3]*(1-(par[0]/x))**2 + par[4]*(1-(par[0]/x))**3 +
                                         par[5]*(1-(par[0]/x))**4 + par[6]*(1-(par[0]/x))**5)/(par[0]*x),
              # [2] eq.(2) electron impact ionisation for n <= 3
              '2': lambda x, par: 5.984*1e-16/x * ((x - par[0])/x)**par[6] * (par[1] + par[2]/(x/par[0]) +
                                         par[3]/(x/par[0])**2 + par[4]/(x/par[0])**3 + par[5]*np.log(x/par[0])),
              # [1,2] eq.(1) electron impact excitation
              '3': lambda x, par: 1e-14*(4*1*(np.log(x/par[0]))*(1-0.7*np.exp(-2.4*((x/par[0])-1)))/(x*par[0]) +
                                         2*4.2*(np.log(x/58))*(1-0.6*np.exp(-0.6*((x/58)-1)))/(x*58))
              # [1] eq on p166 electron impact ionization for n,l > 2s
              }


class CrossSection(object):
    def __init__(self, transition=Transition, impact_energy=float, extrapolate=False):
        self.transition = transition
        self.impact_energy = impact_energy
        self.generate_function()
		
    def generate_function(self):
        projectile=str(self.transition.projectile)
        target=str(self.transition.target)
        cross_dict=ATOMIC_DICT[projectile][target][str(self.transition)]
        cross=CROSS_FUNC[cross_dict['eq']](self.impact_energy,cross_dict['param'])
        self.function=cross
        return cross
		
    def show(self):
        plt.plot(self.impact_energy,self.function)
        plt.xscale('log')
        plt.yscale('log')
        plt.show()
    

class RateCoeff():
    def __init__(self,transition,crossection):
        self.transition=transition
        self.crossection=crossection
        
    def generate_rate(self,temperature,beamenergy):
        self.temperature=temperature
        self.beamenergy=beamenergy
        
        m_t=self.transition.target.mass
        w=np.sqrt(2*self.temperature*sc.eV/m_t)
        
        m_b=self.transition.projectile.mass
        v_b=np.sqrt(2*self.beamenergy*sc.eV/m_b)
        
        E_range=self.crossection.impact_energy
        v=np.sqrt(2*E_range*sc.eV/m_t)
        self.velocity=v
        kernel=v**2*self.crossection.function*(np.exp(-((v-v_b)/w)**2)-np.exp(-((v+v_b)/w)**2))/(np.sqrt(np.pi)*w*v_b**2)
        self.kernel=kernel
        
        self.rate=np.trapz(self.kernel,self.velocity)
        return self.rate
