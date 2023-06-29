from .utils import PhysicalQuantities
import numpy as np
import data2ensembles.mathFuncs as mfuncs
import matplotlib.pyplot as plt

'''
This module is  used to keep correlation functions. In general 
all these functions should have the signature: 

correlation_function_name(params, args, kwargs)

params needs to be an object that can be indexed like a dictionary 
and stores parameters that can be fitted by LMFIT during parameter determination. 

'''

def correlation_anisotropic_emf(params, args, sv=False):
    '''

    This is the correlation function for extended model free inside
    an anisotropic diffusing object. 

    References for this can be found at: 
    1) Deviations from the simple two-parameter model-free approach to the 
    interpretation of nitrogen-15 nuclear magnetic relaxation of proteins
    doi.org/10.1021/ja00168a070

    2) Ab Initio Prediction of NMR Spin Relaxation Parameters from Molecular 
    Dynamics Simulations
    doi.org/10.1021/acs.jctc.7b00750

    3) Rotational diffusion anisotropy of proteins from simultaneous analysis of 
    15N and 13Cα nuclear spin relaxation
    DOI: 10.1023/a:1018631009583

    This is the correlation function associated with the spectral density 
    J_anisotropic_emf()

    Parameters
    ----------
        params : dictionary, Parameters class from LMFIT
            This object stores the parameters for LMFIT during 
            least squares fitting. In principle it should contain any 
            object that can be indexed like a dictioary with the following 
            keys: ('dx', 'dz', dz', 'tau_f', 'tau_s', 'Sf', 'Ss'). These 
            correpond to the diffusion tensor and the extended model free timescales 
            and emplitudes

        args : list, tupple
            This list/tupple should contain (time, ex,ey,ez)
            where ex,ey and ez are the cosine angles to the principle axis of 
            the system

        sv : bool
            this is used to provide an additional order parameter for the 
            fast internal correlation function such that c_internal = 
            sv*c_internal. This can be helpful when fitting molecular dynamics 
            simulations if you have a large drop before the first point. 

    Returns
    -------
        c0 : ndarray 
            This is a numpy array with the computed correlation function for diffusion
        c_internal : ndarray
            This is a numpy array with the computed correlation function for internal motions
        c_total : ndarray
            This the total correlation function c0*c_internal
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

    c_total =  c0 * c_internal
    return c0, c_internal, c_total


    
