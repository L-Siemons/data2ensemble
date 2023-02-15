from .utils import *
import numpy as np
import copy


'''
Here there are some functions for relaxation rates. The definitions  for
r1_YX(), r2_YX() and noe_YX() are taken from the model free manual available at
https://comdnmr.nysbc.org/comd-nmr-dissem/comd-nmr-software/software/modelfree/modelfree_manual.pdf
'''

#define some classes that the functions will use 
PhysicalQuantities = PhysicalQuantities()

def anisotropic_interaction_approximation(params, spectral_density, omega,
    csa_atom_name, csa_params, csa_cosine_angles):
    
    '''
    This funtion makes the approximation for an anisotropic interaction in anisotropic 
    diffusion that was used in 

    Chemical Shift Anisotropy Tensors of Carbonyl, Nitrogen and Amide Proton 
    Nuclei in Proteins through Cross-Correlated Relaxation in NMR Spectroscopy
    Karine Loth, Philippe Pelupessy and Geoffrey Bodenhausen
    '''
    csa_params_xx, csa_params_yy, csa_params_xy = csa_params
    csa_cosine_angles_xx, csa_cosine_angles_yy, csa_cosine_angles_xy = csa_cosine_angles



    #these two are calculated using the full model for the sepctral density since they 
    jxx = spectral_density(csa_params_xx,[omega]+csa_cosine_angles_xx)
    jyy = spectral_density(csa_params_xx,[omega]+csa_cosine_angles_yy)

    #isotropic diffusion
    diso = (params['dx'] + params['dy'] + params['dz'])/3

    # sphere with no motions 
    params_sphere_solid =  copy.deepcopy(csa_params_xy)
    params_sphere_solid['S_long'] = 1
    params_sphere_solid['dx'] = diso
    params_sphere_solid['dy'] = diso
    params_sphere_solid['dz'] = diso

    # sphere with motions
    params_sphere_mobile =  copy.deepcopy(csa_params_xy)
    params_sphere_mobile['dx'] = diso
    params_sphere_mobile['dy'] = diso
    params_sphere_mobile['dz'] = diso

    # anisotropic without motions
    params_anisotropic_solid =  copy.deepcopy(csa_params_xy)
    params_anisotropic_solid['S_long'] = 1

    # the approximation for the jxy
    j_sphere_solid = spectral_density(params_sphere_solid,[omega]+csa_cosine_angles_xy)
    j_sphere_mobile = spectral_density(params_sphere_mobile,[omega]+csa_cosine_angles_xy)
    j_anisotropic_solid =  spectral_density(params_anisotropic_solid,[omega]+csa_cosine_angles_xy)

    jxy = j_sphere_solid + (j_anisotropic_solid - j_sphere_solid) + (j_sphere_mobile - j_sphere_solid)
    return jxx, jyy, jxy

def anisotropic_csa_in_anisotropic_diffusion(params, spectral_density, omega,
    csa_atom_name, csa_params, csa_cosine_angles, 
    PhysQ=PhysicalQuantities):

    jxx, jyy, jxy = anisotropic_interaction_approximation(params, spectral_density, omega,
    csa_atom_name, csa_params, csa_cosine_angles)

    sigmaxx, sigmayy, sigmazz = PhysQ.csa_anisotropic[csa_atom_name]

    sig_diffxx = sigmaxx - sigmazz
    sig_diffyy = sigmayy - sigmazz

    term1 = jxx*6*sig_diffxx**2
    term2 = jyy*6*sig_diffyy**2
    term3 = 12*sig_diffyy*sig_diffxx*jxy
    return term1 + term2 + term3

def r1_YX(params, spectral_density, fields,
    rxy, csa_atom_name, x, y='h', 
    cosine_angles = [], csa_cosine_angles=None, csa_params=None,
    PhysQ=PhysicalQuantities):

    '''
    This function calculates the R1 rate for atom X in a X-Y spin
    pair. It is assumed that the 'Y' = h, however this can be changed.

    params - dictionary like obejct containing the parameters for spectral_density
    fields - fields in tesla
    spectral_density - spectral density and the input for it must be (params, omega)
    rXH - distance between atoms X and y in meters
    csa_atom_name - is the name of the CSA atom
    cosine_angles - these are a list that can be used to pass additional paremeters to spectral_density
     '''

    dd_prefactor = (PhysQ.calc_dd(x,y,rxy)**2)/4
    #csa_prefactor = PhysQ.calc_iso_csa(fields, x, csa_atom_name)**2
    omega_x = PhysQ.calc_omega(x, fields)
    omega_y = PhysQ.calc_omega(y, fields)  

    term1 = dd_prefactor*(spectral_density(params, [omega_y-omega_x]+cosine_angles) + \
                       3.*spectral_density(params, [omega_x]+cosine_angles) + \
                       6.*spectral_density(params, [omega_y+omega_x]+cosine_angles))

    csa_j_term = anisotropic_csa_in_anisotropic_diffusion(params, spectral_density, omega_x,
    csa_atom_name, csa_params ,csa_cosine_angles, 
    PhysQ=PhysQ)

    csa_prefactor = PhysQ.calc_aniso_csa_prefactor(fields, x, csa_atom_name)

    #check the csa prefactor is right!
    term2 = csa_prefactor*csa_j_term

    # this line below is for the axially symetric CSA 
    # term2 = csa_prefactor*spectral_density(params, [omega_x]+csa_cosine_angles)
    #print(term1, term2, term1/term2)
    #print('R1', term2/(term1 + term2))
    #print(f'{x} {y} {dd_prefactor} {csa_prefactor} {dd_prefactor/csa_prefactor:.2f} {rxy} {csa_atom_name}')
    return term1 + term2


def r2_YX(params, spectral_density, fields,
    rxy, csa_atom_name, x,
    y='h', cosine_angles=[], csa_cosine_angles=None,csa_params=None,
    PhysQ=PhysicalQuantities):

    '''
    This function calculates the R2 rate for atom X in a X-Y spin
    
    params - dictionary like obejct containing the parameters for spectral_density
    fields - fields in tesla
    spectral_density - spectral density and the input for it must be (params, omega)
    rxy - distance between atoms X and y in meters
    csa_atom_name - is the name of the CSA atom
    cosine_angles - these are a list that can be used to pass additional paremeters to spectral_density
    '''

    #print(rxy)
    dd_prefactor = (PhysQ.calc_dd(x,y,rxy)**2)/8
    #print(fields, x, csa_atom_name, PhysQ.calc_iso_csa(fields, x, csa_atom_name))
    csa_prefactor = PhysQ.calc_aniso_csa_prefactor(fields, x, csa_atom_name)/6
    omega_x = PhysQ.calc_omega(x, fields)
    omega_y = PhysQ.calc_omega(y, fields)

    term1 = dd_prefactor*(4*spectral_density(params, [0.]+cosine_angles) + \
                                spectral_density(params, [omega_y-omega_x]+cosine_angles) + \
                              3*spectral_density(params, [omega_x]+cosine_angles) + \
                              6*spectral_density(params, [omega_y]+cosine_angles) + \
                              6*spectral_density(params, [omega_x+omega_y]+cosine_angles))

    # if the csa_cosine_angles are none we use the angles for the dipolar interaction
    if csa_cosine_angles == None:
        csa_cosine_angles = cosine_angles

    #anisotropic csa
    csa_j_term_0 = anisotropic_csa_in_anisotropic_diffusion(params, spectral_density, 0,
    csa_atom_name, csa_params,csa_cosine_angles,
    PhysQ=PhysQ)

    csa_j_term_omega_x = anisotropic_csa_in_anisotropic_diffusion(params, spectral_density, omega_x,
    csa_atom_name, csa_params,csa_cosine_angles,
    PhysQ=PhysQ)

    term2 = csa_prefactor*(4*csa_j_term_0 + 3* csa_j_term_omega_x)

    # term2 = csa_prefactor*(4*spectral_density(params,[ 0.]+csa_cosine_angles) \
    #                      + 3*spectral_density(params, [omega_x]+csa_cosine_angles))
 
    return term1 + term2



def noe_YX(params, spectral_density, fields,
    rxy, x, r1, y='h', cosine_angles=[], PhysQ=PhysicalQuantities):

    '''
    This function calculates the noe rate for atom X in a X-Y spin
    
    params - dictionary like obejct containing the parameters for spectral_density
    fields - fields in tesla
    spectral_density - spectral density and the input for it must be (params, omega)
    rxy - distance between atoms X and y in meters
    csa_atom_name - is the name of the CSA atom
    cosine_angles - these are a list that can be used to pass additional paremeters to spectral_density
    r1 - the r1 rate (eg r1_YX)

    '''


    dd_prefactor = (PhysQ.calc_dd(x,y,rxy)**2)/(4*r1)
    gammas = PhysQ.gamma[y]/PhysQ.gamma[x]
    omega_x = PhysQ.calc_omega(x, fields)
    omega_y = PhysQ.calc_omega(y, fields)

    term2 = 1 + dd_prefactor * gammas * \
               (6*spectral_density(params, [omega_x+omega_y]+cosine_angles) \
                - spectral_density(params, [omega_y-omega_x]+cosine_angles))

    return term2

def r1_reduced_noe_YX(params, spectral_density, fields,
    rxy, x, y='h', cosine_angles=[], PhysQ=PhysicalQuantities):

    '''
    This function calculates the (Noe-1)*R1. This is being used in some fitting routines
    
    params - dictionary like obejct containing the parameters for spectral_density
    fields - fields in tesla
    spectral_density - spectral density and the input for it must be (params, omega)
    rxy - distance between atoms X and y in meters
    csa_atom_name - is the name of the CSA atom
    cosine_angles - these are a list that can be used to pass additional paremeters to spectral_density
    r1 - the r1 rate (eg r1_YX)

    '''
    dd_prefactor = (PhysQ.calc_dd(x,y,rxy)**2)/4
    gammas = PhysQ.gamma[y]/PhysQ.gamma[x]
    omega_x = PhysQ.calc_omega(x, fields)
    omega_y = PhysQ.calc_omega(y, fields)

    term2 = dd_prefactor * gammas * \
           (6*spectral_density(params, [omega_x+omega_y]+cosine_angles) \
            - spectral_density(params, [omega_y-omega_x]+cosine_angles))

    return term2