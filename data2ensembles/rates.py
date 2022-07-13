from .utils import *
import numpy as np

'''
Here there are some functions for relaxation rates. The definitions  for
r1_YX(), r2_YX() and noe_YX() are taken from the model free manual available at
https://comdnmr.nysbc.org/comd-nmr-dissem/comd-nmr-software/software/modelfree/modelfree_manual.pdf
'''

#define some classes that the functions will use 
PhysicalQuantities = PhysicalQuantities()

def r1_YX(params, spectral_density, fields,
    rxy, csa_atom_name, x, y='h', 
    cosine_angles = [], PhysQ=PhysicalQuantities):

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
    csa_prefactor = PhysQ.calc_iso_csa(fields, x, csa_atom_name)**2
    omega_x = PhysQ.calc_omega(x, fields)
    omega_y = PhysQ.calc_omega(y, fields)  

    term1 = dd_prefactor*(spectral_density(params, [omega_y-omega_x]+cosine_angles) + \
                       3.*spectral_density(params, [omega_x]+cosine_angles) + \
                       6.*spectral_density(params, [omega_y+omega_x]+cosine_angles))

    term2 = csa_prefactor*spectral_density(params, [omega_x]+cosine_angles)

    return term1 + term2


def r2_YX(params, spectral_density, fields,
    rxy, csa_atom_name, x,
    y='h', cosine_angles=[], PhysQ=PhysicalQuantities):

    '''
    This function calculates the R2 rate for atom X in a X-Y spin
    
    params - dictionary like obejct containing the parameters for spectral_density
    fields - fields in tesla
    spectral_density - spectral density and the input for it must be (params, omega)
    rxy - distance between atoms X and y in meters
    csa_atom_name - is the name of the CSA atom
    cosine_angles - these are a list that can be used to pass additional paremeters to spectral_density
    '''

    dd_prefactor = (PhysQ.calc_dd(x,y,rxy)**2)/8
    csa_prefactor = (PhysQ.calc_iso_csa(fields, x, csa_atom_name)**2)/6
    omega_x = PhysQ.calc_omega(x, fields)
    omega_y = PhysQ.calc_omega(y, fields)

    term1 = dd_prefactor*(4*spectral_density(params, [0.]+cosine_angles) + \
                                spectral_density(params, [omega_y-omega_x]+cosine_angles) + \
                              3*spectral_density(params, [omega_x]+cosine_angles) + \
                              6*spectral_density(params, [omega_y]+cosine_angles) + \
                              6*spectral_density(params, [omega_x+omega_y]+cosine_angles))

    term2 = csa_prefactor*(4*spectral_density(params,[ 0.]+cosine_angles) \
                         + 3*spectral_density(params, [omega_x]+cosine_angles))

    return term1 + term2



def noe_YX(params, spectral_density, fields,
    rxy, x, r1, y='h', cosine_angles=[], PhysQ=PhysicalQuantities):

    '''
    This function calculates the R2 rate for atom X in a X-Y spin
    
    params - dictionary like obejct containing the parameters for spectral_density
    fields - fields in tesla
    spectral_density - spectral density and the input for it must be (params, omega)
    rxy - distance between atoms X and y in meters
    csa_atom_name - is the name of the CSA atom
    cosine_angles - these are a list that can be used to pass additional paremeters to spectral_density
    r1 - the r1 rate (eg r1_YX)

    '''

    dd_prefactor = (PhysQ.calc_dd(x,y,rxy)**2)/(4*r1)
    gammas = PhysQ.gamma[x]/PhysQ.gamma[y]
    omega_x = PhysQ.calc_omega(x, fields)
    omega_y = PhysQ.calc_omega(y, fields)

    term2 = 1 + dd_prefactor * gammas * \
               (6*spectral_density(params, [omega_x+omega_y]+cosine_angles) \
                - spectral_density(params, [omega_y-omega_x]+cosine_angles))

    return term2