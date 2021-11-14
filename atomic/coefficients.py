import numpy as np
from renate_od.atomic import tools
from renate_od.atomic import cross_section
import scipy.constants as sc

"""
Li references:
[1] Wutte et.al. ADNDT 65, 155 (1997) https://www.sciencedirect.com/science/article/abs/pii/S0092640X97907361
[2] Schweinzer et.al. ADNDT 72, 239 (1999)  https://www.sciencedirect.com/science/article/abs/pii/S0092640X9990815X
[3] Schweinzer et.al. JPB 27, 137 (1994) https://iopscience.iop.org/article/10.1088/0953-4075/27/1/017

H references:
[4] Janev et.al. IAEA-APID-4 (1993) https://inis.iaea.org/search/search.aspx?orig_q=RN:25024275
"""

Li_Schweinzer = {'e': {'2s-eloss': {'param': [5.39, 0.085, -0.004, 0.757, -0.178], 'eq': '0'},  # [1] Table p165
                       '2s-2p': {'param': [1.847, -29.466, 10.106, 128.32, -86.415, 58.696, 0.77758], 'eq': '2'},
                       '2s-3s': {'param': [3.372, 4.50, -11.393, 9.2699, 1.8822, 0, 0.32060], 'eq': '2'},
                       '2s-3p': {'param': [3.833, 2.2138, -4.3282, 4.4758, 0, 0, 0.25068], 'eq': '2'},
                       '2s-3d': {'param': [3.877, 8.20, -8.2458, -6.4716, 12.953, 0, 0.61774], 'eq': '2'},
                       '2s-4s': {'param': [4.34, 0.926, -1.1803, -1.2032, 2.0279, 0, 0.053105], 'eq': '2'},
                       '2s-4p': {'param': [4.52, 0.05455, 1.0983, -1, 0.1222, 0.15, 0.12917], 'eq': '2'},
                       '2s-4d': {'param': [4.539, 1.8676, -1.8193, 0.2679, 0, 0, 0.25478], 'eq': '2'},
                       '2s-4f': {'param': [4.54, 0.25, 3.3341, -6.5182, 4.9176, 0, 0.90017], 'eq': '2'},
                       '2p-eloss': {'param': [3.543, 0.075347, 0.1882, -0.28343, 1.5382, -1.855, 1.3411], 'eq': '1'},
                       '2p-3s': {'param': [1.525, -12.287, 7.9788, 21.593, -10.454, 10.136, 0.35903], 'eq': '2'},
                       '2p-3p': {'param': [1.986, 10.353, -22.590, 21.347, 0, 0, 0.33146], 'eq': '2'},
                       '2p-3d': {'param': [2.03, -20.091, -78.831, 306.52, -146.03, 38.767, 1.7130], 'eq': '2'},
                       '2p-4s': {'param': [2.493, -0.45472, 6.2003, -23.437, 25.556, 0.70915, 1.1250], 'eq': '2'},
                       '2p-4p': {'param': [2.673, 2.4654, -3.9177, 4.7847, 0, 0, 0.78931], 'eq': '2'},
                       '2p-4d': {'param': [2.692, 4.8531, -40.566, 83.492, -46.007, 4.5560, 0.57257], 'eq': '2'},
                       '2p-4f': {'param': [2.693, 2.1425, 24.042, -77.016, 62.941, 0.24507, 0.79098], 'eq': '2'},
                       '3s-eloss': {'param': [2.018, 5.55e-15, 0.34812, -0.29262, 0.34264, 0.99613, -0.82395],
                                    'eq': '1'},
                       '3s-3p': {'param': [0.461, -678.58, 805.29, 0, 0, 439.5, 0.78903], 'eq': '2'},
                       '3s-3d': {'param': [0.505, 96.957, -309.15, 1089.2, 0, 0, 2.4374], 'eq': '2'},
                       '3s-4s': {'param': [0.968, 18.409, -72.121, 34.575, 109.59, 0.94483, 1.2021], 'eq': '2'},
                       '3s-4p': {'param': [1.148, 8.4345, -10.284, 0, 25.887, 1.9035, 0.62176], 'eq': '2'},
                       '3s-4d': {'param': [1.167, 10.233, 407.74, -2173.8, 3286.7, 0, 3.7253], 'eq': '2'},
                       '3s-4f': {'param': [1.168, 28.327, 100.47, -421.76, 419.5, 0, 1.00], 'eq': '2'},
                       '3p-eloss': {'param': [1.557, 0.11218, 0.24966, -1.0074, 1.8549, -1.1973, 0.48164], 'eq': '1'},
                       '3p-3d': {'param': [0.044, 186.96, -1032.5, 2307.1, 0, 230.05, 2.68], 'eq': '2'},
                       '3p-4s': {'param': [0.507, -117.31, 196.06, -78.658, 0, 64.033, 0.062851], 'eq': '2'},
                       '3p-4p': {'param': [0.687, 49.0, -218.58, 392.42, -201.84, 0, 0.61809], 'eq': '2'},
                       '3p-4d': {'param': [0.706, -202.18, 335.8, -132.6, 0, 113.64, 0.28384], 'eq': '2'},
                       '3p-4f': {'param': [0.707, 115.93, -99.313, -319.37, 758.96, 0, 2.3647], 'eq': '2'},
                       '3d-eloss': {'param': [1.513, 0.02339, 0.37229, 0.011639, -3.1118, 7.7385, -4.5292], 'eq': '1'},
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
                 '1H1+': {'2s-eloss': {},
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
                          '4f-eloss': {}}}

Li_Wutte = {'e': {'2s-2p': {'param': [1.847, 4.1818, -27.335, 89.34, 0, 52.788, 1.1971], 'eq': '2'},
                  '2s-3s': {'param': [3.372, 4.7015, -4.6972, -3.0504, 11.332, 0, 0.51449], 'eq': '2'},
                  '2s-3p': {'param': [3.833, 1.4705, 1.5511, 0.94339, 0, 0.12721, 0.53112], 'eq': '2'},
                  '2s-3d': {'param': [3.877, 8.2190, 18.868, -51.214, 55.412, 0, 1.4119], 'eq': '2'},
                  '2p-eloss': {'param': [3.541], 'eq': '3'},
                  '2p-3s': {'param': [1.525, 1.0693e-2, -2.4093, 5.7497, 0, 8.0951, 0.55336], 'eq': '2'},
                  '2p-3p': {'param': [1.986, 10.799, -7.9161, 5.6946, 0, 0, 0.60959], 'eq': '2'},
                  '2p-3d': {'param': [2.030, 3.8735e-3, 37.646, -5.2622, 0, 36.321, 1.0935], 'eq': '2'},
                  '3s-eloss': {'param': [2.018], 'eq': '3'},
                  '3s-3p': {'param': [0.461, 8.4471e-2, 225.8, 0, 0, 322.51, 1.0632], 'eq': '2'},
                  '3s-3d': {'param': [0.505, 91.515, -48.338, 127.34, 0, 0, 1.0738], 'eq': '2'},
                  '3p-eloss': {'param': [1.557], 'eq': '3'},
                  '3p-3d': {'param': [0.044, 3.06044e-2, 25.686, 156.57, 0, 260.22, 1.034], 'eq': '2'},
                  '3d-eloss': {'param': [1.513], 'eq': '3'}},
            '1H1+': {'2s-eloss': {},
                     '2s-2p': {},
                     '2s-3s': {},
                     '2s-3p': {},
                     '2s-3d': {},
                     '2p-eloss': {},
                     '2p-3s': {},
                     '2p-3p': {},
                     '2p-3d': {},
                     '3s-eloss': {},
                     '3s-3p': {},
                     '3s-3d': {},
                     '3p-eloss': {},
                     '3p-3d': {},
                     '3d-eloss': {}}}


def get_Johnson_osc_coeff(n):  # [g0, g1, g2]
    if n == 1:
        return [1.1330, -0.4059, 0.0714]
    if n == 2:
        return [1.0785, -0.2319, 0.02947]
    if n >= 3:
        return [0.9935+0.2328/n-0.1296/n**2, -1/n*(0.6282-0.5598/n+0.5299/n**2), -1/n**2*(0.3887-1.181/n+1.470/n**2)]


def get_Johnson_osc_str(n, m):
    x = 1-(n/m)**2
    g0, g1, g2 = get_Johnson_osc_coeff(n)
    g = g0+g1/x+g2/x**2
    return 32/(3*3**0.5*np.pi)*n/m**3/x**3*g


def get_Janev_params(cross):
    transition = cross.transition
    e = cross.impact_energy
    if transition.name == 'ex':
        if str(transition.target) == 'e':
            if transition.from_level == '1':  # NOT GOOD!!!!!!!!
                n = int(transition.to_level)
                y = 1-(1/n)**2
                f = get_Johnson_osc_str(1, n)
                b = 1/n*(4.0-18.63/n+36.24/n**2-28.09/n**3)
                return {'param': [1, n, y, 0.45, 2*n**2*f/y, 2/3*n**2*(5+b), 13.6*(1-1/n**2)], 'eq': '12'}
            if int(transition.from_level) > 1:
                n = int(transition.from_level)
                m = int(transition.to_level)
                y = 1-(n/m)**2
                f = get_Johnson_osc_str(n, m)
                b = 1/n*(4.0-18.63/n+36.24/n**2-28.09/n**3)
                return {'param': [n, m, y, 1.94*n**(-1.57), 2*n**2*f/y, 4*n**4/(m**3*y**2)*(1+4/(3*y)+b/y**2),
                                  13.6*(1/n**2-1/m**2)], 'eq': '12'}
        if str(transition.target) == '1H1+':
            if transition.from_level == '1':
                n = int(transition.to_level)
                return {'param': [0.63771, 37.174, 0.39265, 3.2949e-4, 0.25757, -2.2950,
                                  0.050796, -5.5986, n], 'eq': '17'}
            if transition.from_level == '2' and int(transition.to_level) < 11:
                n = transition.to_level
                ratio = {'6': 0.4610, '7': 0.2475, '8': 0.1465, '9': 0.0920, '10': 0.0605}
                return {'param': [18.264, 18.973, 2.9056, 0.013701, 0.31711, -1.4775, ratio[n]], 'eq': '19'}
            if transition.from_level == '3' and int(transition.to_level) < 11:
                n = transition.to_level
                ratio = {'7': 0.4670, '8': 0.2545, '9': 0.1540, '10': 0.10}
                return {'param': [63.494, 11.507, 4.3417, 0.077953, 0.53461, -1.2881, ratio[n]], 'eq': '19'}
            if int(transition.from_level) > 3:
                n = int(transition.from_level)
                m = int(transition.to_level)
                eps = e/25e3
                s = m-n
                D = np.exp(-1/(n*m*eps**2))
                zp = 2/(eps*n**2*((2-n**2/m**2)**0.5+1))
                zm = 2/(eps*n**2*((2-n**2/m**2)**0.5-1))
                y = 1/(1-D*np.log(18*s)/(4*s))
                C = lambda z, z1: z**2*np.log(1+2*z/3)/(2*z1+3*z/2)
                return {'param': [n, eps, 8/(3*s)*(m/(s*n))**3*(0.184-0.04/s**(2/3))*(1-0.2*s/(n*m))**(1+2*s),
                                  D, np.log((1+0.53*eps**2*n*(m-2/m))/(1+0.4*eps)),
                                  (1-0.3*s*D/(n*m))**(1+2*s), 0.5*(eps*n**2/(m-1/m))**3,
                                  C(zm, y)-C(zp, y)], 'eq': '110'}
    if transition.name == 'eloss':
        if str(transition.target) == 'e':
            if int(transition.from_level) > 3:  # NOT GOOD!!!!!!!
                n = int(transition.from_level)
                y = 1-(1/n)**2
                f = get_Johnson_osc_str(1, n)
                b = 1/n*(4.0-18.63/n+36.24/n**2-28.09/n**3)
                return {'param': [n, y, 1.94*n**(-1.57), 2*n**2*f/y, 2/3*n**2*(5+b), 13.6/n**2], 'eq': '14'}
        if str(transition.target) == '1H1+':
            trans_ion=tools.Transition(projectile=transition.projectile,
                                       target=transition.target,
                                       from_level=transition.from_level,
                                       to_level='ion', trans='ion')
            cross_ion=cross_section.CrossSection(transition=trans_ion,
                                                 impact_energy=e,atomic_dict=cross.atomic_dict)
            trans_cx=tools.Transition(projectile=transition.projectile,
                                       target=transition.target,
                                       from_level=transition.from_level,
                                       to_level='cx', trans='cx')
            cross_cx=cross_section.CrossSection(transition=trans_cx,
                                                 impact_energy=e,atomic_dict=cross.atomic_dict)
            return {'param': cross_ion.function+cross_cx.function, 'eq': '114'}
    if transition.name == 'ion':
        if str(transition.target) == '1H1+':
            if int(transition.from_level) > 3:  # NOT GOOD!!!!!!!!!!!
                n = int(transition.from_level)
                e_red = (3/n)**2*e/1e3
                return {'param': [336.26, 13.608, 4.9910e+3, 3.0560e-1, 6.4364e-2, -0.14924,
                                  3.1525, -1.6314, n, e_red], 'eq': '111'}
    if transition.name == 'cx':
        if str(transition.target) == '1H1+':
            n = int(transition.from_level)
            if n == 2:
                e_red = e*n**2/1e3
                return {'param': [0.92750, 6.5040e+3, 1.3405e-2, 20.699, n, e_red], 'eq': '113'}
            if n == 3:
                e_red = e*n**2/1e3
                return {'param': [0.37271, 2.7645e+6, 1.5720e-3, 1.4857e+3, n, e_red], 'eq': '113'}
            if n >= 4:
                e_red = e*n**2/1e3
                return {'param': [0.21336, 1.0e+10, 1.8184e-3, 1.3426e+6, n, e_red], 'eq': '113'}


def H_deex_modifier(rate):
    g1=int(rate.transition.to_level)**2
    g2=int(rate.transition.from_level)**2
    if str(rate.transition.target) == 'e':
        deltaE=13.605693122994*(1/g1-1/g2)
        return rate.rate*g1/g2*np.exp(deltaE/rate.temperature)
    if str(rate.transition.target) == '1H1+':
        return rate.rate*g1/g2

            

H_ALADDIN = {'e': {'1-2': {'param': [1.4182, -20.877, 49.735, -46.249, 17.442, 4.4979], 'eq': '10'},
                   '1-3': {'param': [0.42956, -0.58288, 1.0693, 0.0, 0.75448, 0.38277, 12.09], 'eq': '11'},
                   '1-4': {'param': [0.24846, 0.19701, 0.0, 0.0, 0.243, 0.41844, 12.75], 'eq': '11'},
                   '1-5': {'param': [0.13092, 0.23581, 0.0, 0.0, 0.11508, 0.45929, 13.06], 'eq': '11'},
                   '2-3': {'param': [5.2373, 119.25, -595.39, 816.71, 38.906, 1.3196, 1.889], 'eq': '11'},
                   '1-eloss': {'param': [0.18450, -0.032226, -0.034539, 1.4003, -2.8115, 2.2986, 13.6], 'eq': '13'},
                   '2-eloss': {'param': [0.14784, 0.0080871, -0.062270, 1.9414, -2.1980, 0.95894, 3.4], 'eq': '13'},
                   '3-eloss': {'param': [0.058463, -0.051272, 0.85310, -0.57014, 0.76684, 0.0, 1.511], 'eq': '13'}
                   },
             '1H1+': {'1-2': {'param': [34.433, 44.507, 0.56870, 8.5476, 7.8501, -9.2217,
                                        1.8020e-2, 1.6931, 1.9422e-3, 2.9068], 'eq': '15'},
                      '1-3': {'param': [6.1950, 35.773, 0.54818, 5.5162e-3, 0.291114,
                                        -4.5264, 6.0311, -2.0679], 'eq': '16'},
                      '1-4': {'param': [2.0661, 34.975, 0.91213, 5.133e-4, 0.28953,
                                        -2.2849, 0.11528, -4.8970], 'eq': '16'},
                      '1-5': {'param': [1.2449, 32.291, 0.21176, 3.0826e-4, 0.31063,
                                        -2.4161, 0.024664, -6.3726], 'eq': '16'},
                      '1-6': {'param': [0.63771, 37.174, 0.39265, 3.2949e-4, 0.25757,
                                        -2.2950, 0.050796, -5.5986], 'eq': '16'},
                      '2-3': {'param': [394.51, 21.606, 0.62426, 0.013597, 0.16565, -0.8949], 'eq': '18'},
                      '2-4': {'param': [50.744, 19.416, 4.0262, 0.014398, 0.31584, -1.4799], 'eq': '18'},
                      '2-5': {'param': [18.264, 18.973, 2.9056, 0.013701, 0.31711, -1.4775], 'eq': '18'},
                      '3-4': {'param': [1247.5, 11.319, 2.6235, 0.068781, 0.521176, -1.2722], 'eq': '18'},
                      '3-5': {'param': [190.59, 11.096, 2.9098, 0.073307, 0.54177, -1.2894], 'eq': '18'},
                      '3-6': {'param': [63.494, 11.507, 4.3417, 0.077953, 0.53461, -1.2881], 'eq': '18'},
                      '1-ion': {'param': [12.899, 61.897, 9.2731e+3, 4.9749e-4, 3.9890e-2,
                                          -1.590, 3.1834, -3.7154], 'eq': '16'},
                      '2-ion': {'param': [107.63, 29.860, 1.0176e+6, 6.9713e-3, 2.8448e-2,
                                          -1.80, 4.7852e-2, -0.20923], 'eq': '16'},
                      '3-ion': {'param': [336.26, 13.608, 4.9910e+3, 3.0560e-1, 6.4364e-2,
                                          -0.14924, 3.1525, -1.6314], 'eq': '16'},
                      '1-cx': {'param': [3.2345, 235.88, 0.038371, 3.8068e-6, 1.1832e-10, 2.3713], 'eq': '112'},
                      },
             'generalized': get_Janev_params,
             'de-ex': H_deex_modifier}

ATOMIC_SOURCES = {'Li_Schweinzer': Li_Schweinzer,
                  'Li_Wutte': Li_Wutte,
                  'H_ALADDIN': H_ALADDIN}
