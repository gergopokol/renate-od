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
                                         par[3]/(x/par[0])**2 + par[4]/(x/par[0])**3 + par[5]*np.log(x/par[0])),
              # [1,2] eq.(1) electron impact excitation
              
              '3': lambda x, par: 1e-14*(4*1*(np.log(x/par[0]))*(1-0.7*np.exp(-2.4*((x/par[0])-1)))/(x*par[0]) +
                                         2*4.2*(np.log(x/58))*(1-0.6*np.exp(-0.6*((x/58)-1)))/(x*58)),
              # [1] eq on p166 electron impact ionization for n,l > 2s
              
              '10': lambda e, par: 5.984e-16/e*(par[0]+par[1]/(e/10.2)+par[2]/(e/10.2)**2+
                                                par[3]/(e/10.2)**3+par[4]/(e/10.2)**4+par[5]*np.log(e/10.2)),
              # [4] e[eV]>12.23 eV
              
              '11': lambda e, par: 5.984e-16/e*((e-par[6])/e)**par[5]*(par[0]+par[1]/(e/par[6])+par[2]/(e/par[6])**2+
                                                                     par[3]/(e/par[6])**3+par[4]*np.log(e/par[6])),
              # [4] e[eV]>par[6]
              
              }


class CrossSection(object):
    def __init__(self, transition=Transition, impact_energy=float, atomic_dict=None, extrapolate=False):
        self.transition = transition
        self.impact_energy = impact_energy
        self.ATOMIC_DICT=atomic_dict
        self.__generate_function()
		
    def __generate_function(self):
        target=str(self.transition.target)
        cross_dict=self.ATOMIC_DICT[target][str(self.transition)]
        cross=CROSS_FUNC[cross_dict['eq']](self.impact_energy,cross_dict['param'])
        self.function=cross
        return cross
		
    def show(self):
        fig,ax=plt.subplots()
        ax.plot(self.impact_energy,self.function)
        ax.set_xscale('log')
        ax.set_yscale('log')
        return fig,ax
    

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
