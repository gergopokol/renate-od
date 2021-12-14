from atomic.tools import Transition
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc

"""
Li references:
[1] Wutte et.al. ADNDT 65, 155 (1997) https://www.sciencedirect.com/science/article/abs/pii/S0092640X97907361
[2] Schweinzer et.al. ADNDT 72, 239 (1999)  https://www.sciencedirect.com/science/article/abs/pii/S0092640X9990815X
[3] Schweinzer et.al. JPB 27, 137 (1994) https://iopscience.iop.org/article/10.1088/0953-4075/27/1/017

H references:
[4] Janev et.al. IAEA-APID-4 (1993) https://inis.iaea.org/search/search.aspx?orig_q=RN:25024275
"""

CROSS_FUNC = {'0': lambda x, par: 1e-13*(par[1]*np.log(x/par[0]) + par[2]*(1-(par[0]/x)) +
                                         par[3]*(1-(par[0]/x))**2 + par[4]*(1-(par[0]/x))**3)/(par[0]*x),
              # [1] eq. on p165, electron - impact ionization for 2s

              '1': lambda x, par: 1e-13*(par[1]*np.log(x/par[0]) + par[2]*(1-(par[0]/x)) +
                                         par[3]*(1-(par[0]/x))**2 + par[4]*(1-(par[0]/x))**3 +
                                         par[5]*(1-(par[0]/x))**4 + par[6]*(1-(par[0]/x))**5)/(par[0]*x),
              # [2] eq.(2) electron impact ionisation for n <= 3

              '2': lambda x, par: 5.984*1e-16/x * ((x - par[0])/x)**par[6] * (par[1] + par[2]/(x/par[0]) +
                                                                              par[3]/(x/par[0])**2 + par[4] /
                                                                              (x/par[0])**3 + par[5]*np.log(x/par[0])),
              # [1,2] eq.(1) electron impact excitation

              '3': lambda x, par: 1e-14*(4*1*(np.log(x/par[0]))*(1-0.7*np.exp(-2.4*((x/par[0])-1)))/(x*par[0]) +
                                         2*4.2*(np.log(x/58))*(1-0.6*np.exp(-0.6*((x/58)-1)))/(x*58)),
              # [1] eq on p166 electron impact ionization for n,l > 2s

              '10': lambda e, par: 5.984e-16/e*(par[0]+par[1]/(e/10.2)+par[2]/(e/10.2)**2 +
                                                par[3]/(e/10.2)**3+par[4]/(e/10.2)**4+par[5]*np.log(e/10.2)),
              # [4] e[eV]>12.23 eV

              '11': lambda e, par: 5.984e-16/e*((e-par[6])/e)**par[5]*(par[0]+par[1]/(e/par[6])+par[2]/(e/par[6])**2 +
                                                                       par[3]/(e/par[6])**3+par[4]*np.log(e/par[6])),
              # [4] e[eV]>par[6]

              '12': lambda e, par: 1.76e-16*par[0]**2/(par[2]*e/par[6])*(1-np.exp(-1*par[3]*par[2]*e/par[6])) *
                                (par[4]*(np.log(e/par[6])+1/(2*e/par[6])) +
                                 (par[5]-par[4] * np.log(2*par[0]**2/par[2]))*(1-1/(e/par[6]))),
              # [4] e[eV]>par[6]

              '13': lambda e, par: 1e-13/(e*par[6])*(par[0]*np.log(e/par[6])+par[1]*(1-par[6]/e) +
                                                     par[2]*(1-par[6]/e)**2+par[3]*(1-par[6]/e)**3 +
                                                     par[4]*(1-par[6]/e)**4+par[5]*(1-par[6]/e)**5),
              # [4] e[eV]>13.6 eV

              '14J': lambda e, par: 1.76e-16/(e/par[5])*(1-np.exp(-par[2]*e/par[5])) *
              (par[3]*np.log(e/par[5])+(par[4]-par[3]*np.log(2*par[0]**2)) * (1-1/(e/par[5]))**2),
              # [4] e[eV]>par[5]

              '14': lambda e, par: 1.76e-16*par[0]**2/(e/par[5])*(1-np.exp(-par[2]*e/par[5])) *
              (par[3]*np.log(e/par[5])+(par[4]-par[3]*np.log(2*par[0]**2)) * (1-1/(e/par[5]))**2),
              # [4] e[eV]>par[5]

              '15': lambda e, par: 1e-16*par[0]*(np.exp(-par[1]/(e/1e3))*np.log(1+par[2]*e/1e3)/(e/1e3)+par[3] *
                                                 np.exp(-par[4]*e/1e3)/(e/1e3)**par[5]+par[6]*np.exp(-par[7]/(e/1e3)) /
                                                 (1+par[8]*(e/1e3)**par[9])),
              # [4] e[eV]>0.6 keV

              '16': lambda e, par: 1e-16*par[0]*(np.exp(-par[1]/(e/1e3))*np.log(1+par[2]*e/1e3)/(e/1e3) +
                                                 par[3]*np.exp(-par[4]*e/1e3)/((e/1e3)**par[5]+par[6]*(e/1e3)**par[7])),
              # [4] e[eV]>0.5 keV

              '17': lambda e, par: (6/par[8])**3*1e-16*par[0]*(np.exp(-par[1]/(e/1e3))*np.log(1+par[2]*e/1e3)/(e/1e3) +
                                                               par[3]*np.exp(-par[4]*e/1e3)/((e/1e3)**par[5]+par[6]*(e/1e3)**par[7])),
              # [4] e[eV]>0.5 keV

              '18': lambda e, par: 1e-16*par[0]*(np.exp(-par[1]/(e/1e3))*np.log(1+par[2]*e/1e3)/(e/1e3) +
                                                 par[3]*np.exp(-par[4]*e/1e3)/(e/1e3)**par[5]),
              # [4] e[eV]>0.5 keV
              '19': lambda e, par: par[6]*1e-16*par[0]*(np.exp(-par[1]/(e/1e3))*np.log(1+par[2]*e/1e3)/(e/1e3) +
                                                        par[3]*np.exp(-par[4]*e/1e3)/(e/1e3)**par[5]),
              # [4] e[eV]>0.5 keV

              '110': lambda e, par: 8.8e-17*par[0]**4/par[1]*(par[2]*par[3]*par[4]+par[5]*par[6]*par[7]),
              # [4] e[eV]>0.1 keV

              '111': lambda e, par: (par[8]/3)**4*1e-16*par[0]*(np.exp(-par[1]/(par[9])) *
                                                                np.log(1+par[2]*par[9])/(par[9]) +
                                                                par[3]*np.exp(-par[4]*par[9]) /
                                                                ((par[9])**par[5]+par[6]*(par[9])**par[7])),
              # [4] e[eV]>0.1 keV

              '112': lambda e, par: 1e-16*par[0]*np.log(par[1]/(e/1e3)+par[5])/(1+par[2]*e/1e3+par[3]*(e/1e3)**3.5 +
                                                                                par[4]*(e/1e3)**5.4),
              # [4] e[eV]>1 eV

              '113': lambda e, par: par[4]**4*1e-16*par[0]*np.log(par[1]/par[5]+par[3]) /
                                    (1+par[2]*par[5]+3.0842e-6*par[5]**3.5+1.1832e-10*par[5]**5.4),
              # [4] e[eV]>1 eV

              '114': lambda e, par: par,  # de-excitation

              '115': lambda e, par: par[8]*1e-16*par[6]*par[0]*(np.exp(-par[1]/par[7])*np.log(1+par[2]*par[7])/par[7] +
                                                                par[3]*np.exp(-par[4]*par[7])/par[7]**par[5]),
              # [4] e[eV]>1 keV

              '116': lambda e, par: (5/par[9])**3*par[8]*1e-16*par[6]*par[0]*(np.exp(-par[1]/par[7])*np.log(1+par[2]*par[7])/par[7] +
                                                                              par[3]*np.exp(-par[4]*par[7])/par[7]**par[5]),

              '117': lambda e, par: par[9]*par[8]*1e-16*par[6]*par[0]*(np.exp(-par[1]/par[7])*np.log(1+par[2]*par[7])/par[7] +
                                                                       par[3]*np.exp(-par[4]*par[7])/par[7]**par[5]),

              '118': lambda e, par: par[9]*par[8]*8.86e-17*par[0]**4/par[1]*(par[2]*par[3]*par[4]+par[5]*par[6]*par[7]),

              '119': lambda e, par: np.exp(-par[1]*par[7]/par[6])*par[0]**4*3.52e-16*par[7]**2/par[6]*(par[2]*(np.log(par[6]/(par[5]**2-par[6])) -
                                                                                                               par[6]/par[5]**2)+par[3]-par[4]/par[6]),
              '120': lambda e, par: par[7]*1e-16*par[0]*np.log(par[1]/par[6]+par[2])/(1+par[3]*par[6]+par[4]*par[6]**3.5+par[7]*par[6]**5.4),
              '121': lambda e, par: par[2]**4*par[3]*7.04e-16*par[0]/(par[4]**3.5*(1+par[1]*par[4]**2))*(1-np.exp(-2*par[4]**3.5*(1+par[1]*par[4]**2)
                                                                                                                  / 3*par[0])),
              '122': lambda e, par: par[8]**4*par[7]*1e-16*par[0]*np.log(par[1]/par[6]+par[2])/(1+par[3]*par[6]+par[4]*par[6]**3.5+par[7]*par[6]**5.4),
              '123': lambda e, par: 1e-16*par[0]*(np.exp(-par[1]/par[10])*np.log(1+par[2]*par[10])/(par[10])+par[3] *
                                                 np.exp(-par[4]*par[10])/(par[10])**par[5]+par[6]*np.exp(-par[7]/(par[10])) /
                                                  (1+par[8]*(par[10])**par[9])),
              # same as 15
              '124': lambda e, par: 1e-16*par[0]*(np.exp(-par[1]/(par[8]))*np.log(1+par[2]*par[8])/(par[8]) +
                                                  par[3]*np.exp(-par[4]*par[8])/((par[8])**par[5]+par[6]*(par[8])**par[7])),
              # same as 16
              '125': lambda e, par: (6/par[8])**3*1e-16*par[0]*(np.exp(-par[1]/(par[9]))*np.log(1+par[2]*par[9])/(par[9]) +
                                                                par[3]*np.exp(-par[4]*par[9])/((par[9])**par[5]+par[6]*(par[9])**par[7])),
              # same as 17
              '126': lambda e, par: 1e-16*par[0]*(np.exp(-par[1]/(par[6]))*np.log(1+par[2]*par[6])/(par[6]) +
                                                  par[3]*np.exp(-par[4]*par[6])/(par[6])**par[5]),
              # same as 18
              '127': lambda e, par: par[7]*1e-16*par[0]*(np.exp(-par[1]/(par[6]))*np.log(1+par[2]*par[6])/(par[6]) +
                                                         par[3]*np.exp(-par[4]*par[6])/(par[6])**par[5]),
              # same as 19
              '128': lambda e, par: 1.76e-16*par[0]**4/par[1]*(par[2]*par[3]*par[4]+par[5]*par[6]*par[7]),
              # same as 110
              '129': lambda e, par: 1e-16*par[0]*(np.exp(-par[1]/par[9])/(1+par[2]*par[9]**par[3]+par[4]*par[9]**3.5 +
                                                                          par[5]*par[9]**5.4)+par[6]*np.exp(-par[7]*par[9])/par[9]**par[8]),
              '130': lambda e, par: 7.04e-16*par[4]**4*par[0]*(1-np.exp(-4/(3*par[0])*(1+par[5]**par[1]+par[2]*par[5]**3.5 +
                                                                                       par[3]*par[5]**5.4)))/(1+par[5]**par[1]+par[2]*par[5]**3.5 +
                                                                                                                par[3]*par[5]**5.4)
              }


class CrossSection(object):
    def __init__(self, transition=Transition, impact_energy=float, atomic_dict=None, extrapolate=False):
        self.transition = transition
        self.impact_energy = impact_energy
        self.atomic_dict = atomic_dict
        self.__generate_function()

    def __get_generating_params(self):
        target = str(self.transition.target)
        if str(self.transition) in self.atomic_dict[target].keys():
            param_dict = self.atomic_dict[target][str(self.transition)]
        else:
            param_dict = self.atomic_dict['generalized'](self)
        return param_dict

    def __generate_function(self):
        trans_og_switch = False
        if self.transition.name == 'de-ex':
            # print('The crossection will be given for the excitation counterpart of \
            #       the de-excitation transition given. The appropriate scaling is done \
            #           when calculating the rate coefficients.')
            trans_og = self.transition
            trans_og_switch = True
            trans_ex = Transition(projectile=self.transition.projectile,
                                  target=self.transition.target,
                                  from_level=self.transition.to_level,
                                  to_level=self.transition.from_level,
                                  trans='ex')
            self.transition = trans_ex
        param_dict = self.__get_generating_params()
        cross = CROSS_FUNC[param_dict['eq']](self.impact_energy, param_dict['param'])
        self.function = cross
        if trans_og_switch:
            self.transition = trans_og
        return cross

    def show(self):
        fig, ax = plt.subplots()
        ax.plot(self.impact_energy, self.function)
        ax.set_xscale('log')
        ax.set_yscale('log')
        return fig, ax


class RateCoeff:
    def __init__(self, transition, crossection):
        self.transition = transition
        self.crossection = crossection

    def generate_rate(self, temperature, beamenergy):
        self.temperature = temperature
        self.beamenergy = beamenergy

        m_t = self.transition.target.mass
        w = np.sqrt(2*self.temperature*sc.eV/m_t)

        m_b = self.transition.projectile.mass
        v_b = np.sqrt(2*self.beamenergy*sc.eV/m_b)

        E_range = self.crossection.impact_energy
        v = np.sqrt(2*E_range*sc.eV/m_t)
        self.velocity = v
        kernel = v**2*self.crossection.function*(np.exp(-((v-v_b)/w)**2) -
                                                 np.exp(-((v+v_b)/w)**2))/(np.sqrt(np.pi)*w*v_b**2)
        self.kernel = kernel

        self.rate = np.trapz(self.kernel, self.velocity)
        if self.transition.name == 'de-ex':
            self.rate = self.crossection.atomic_dict['de-ex'](self)

        return self.rate
