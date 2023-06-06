from .utils import PhysicalQuantities
import numpy as np
import data2ensembles.mathFuncs as mfuncs
import matplotlib.pyplot as plt

def correlation_anisotropic_emf(params, args, sv = False):
    '''
    This is the correlation function associated with the spectral density 
    J_anisotropic_emf()
    '''

    time, ex,ey,ez, = args
    dx = params['dx']#.value
    dy = params['dy']#.value
    dz = params['dz']#.value

    isotropy_check = 0
    if dx-dy < 1e-5:
        if dx-dz < 1e-5:
            isotropy_check = 1

    if isotropy_check == 0:
        taus, amps = mfuncs.calculate_anisotropic_d_amps(dx,dy,dz,ex,ey,ez)
        ds = np.array([1/t for t in taus])
    else:
        ds = np.array([(dx+dy+dz)/3])
        amps = np.array([1])

    c0 = np.zeros(len(time))
    # could probably do this just with numpy arrays for speed
    for ai, di in zip(amps, ds):
        # print('time ', time)
        # print(-1*di*time)
        c0 = c0 + ai * np.e**(-1*di*time)
    
    S_long = params['Sf']*params['Ss']

    c_internal = S_long
    c_f_index = -1*(1e6*time)/(1e6*params['tau_f'])
    c_s_index = -1*(1e6*time)/(1e6*params['tau_s'])
    
    c_internal = c_internal +  (1-params['Sf'])*np.e**(c_f_index)
    c_internal = c_internal +  (params['Sf']-S_long)*np.e**(c_s_index)

    # this is here to account for the super fast decays we see within the
    # first step of MD trajectories
    
    if sv == True:
        c_internal = c_internal * params['Sv']
    
    #print(params['Sf'].value, params['Ss'].value, params['tau_f'].value, params['tau_s'].value, )
    # print('C0', c0)
    # print('cinternal', c_internal)
    c_total =  c0 * c_internal

    # plt.plot(time, c_internal, label='internal')
    # plt.plot(time, c0, label='c0')
    # plt.legend()
    # plt.show()
    return c0, c_internal, c_total


    
