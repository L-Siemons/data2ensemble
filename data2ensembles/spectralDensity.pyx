from .utils import PhysicalQuantities
import numpy as np
import data2ensembles.mathFuncs as mfuncs
import cython
cimport numpy as np

PhysQ = PhysicalQuantities()


def J_iso_tauc_only(params, omega):
    """
    Isotropic spectral density function with only tauC 
    """
    omega = omega[0]
    return (2./5.) * params['tc'] / ( 1+(omega*params['tc'])**2) 

def J_iso_tauc_S2(params, omega):
    """
    Isotropic spectral density function with only tauC 
    """
    omega = omega[0]
    te = (params['tc']*params['tf'])/(params['tc']+params['tf'])
    return (2./5.) * (params['S2']*params['tc'] / ( 1+(omega*params['tc'])**2) + (1-params['S2'])*te / ( 1+(omega*te)**2))


def J_anisotropic_emf(params, args):

    '''
    This function uses a comepletely isotropic mode for the spectral density function
    with the extended model free formalism. 

    To select an isotropic model and a axial symetric one you can set the parameters 
    accordingly. 

    Take from https://link.springer.com/content/pdf/10.1023/A:1018631009583.pdf

    '''

    cdef float ex
    cdef float ey
    cdef float ez

    cdef float diso
    cdef float L2 

    cdef float tau 
    cdef float amp 
    cdef list taus 
    cdef list amps

    cdef float dx 
    cdef float dy
    cdef float dz

    cdef float sf 
    cdef float ss
    cdef float s2

    cdef float tau_s 
    cdef float tau_f
    cdef int isotropy_check 

    cdef np.ndarray[np.float_t, ndim=1] total
    cdef np.ndarray[np.float_t, ndim=1] omega
    cdef np.ndarray[np.float_t, ndim=1] term1
    cdef np.ndarray[np.float_t, ndim=1] term2_bot
    cdef np.ndarray[np.float_t,  ndim=1] term2

    cdef float term2_top 
    cdef float term3_top 
    
    cdef np.ndarray[np.float_t, ndim=1] term3_bot
    cdef np.ndarray[np.float_t, ndim=1] term3

    # unpack varriables
    dx = params['dx']#.value
    dy = params['dy']#.value
    dz = params['dz']#.value

    sf = params['Sf']#.value
    ss = params['Ss']#.value
    s2 = sf*ss
    tau_s = params['tau_s']#.value
    tau_f = params['tau_f']#.value

    omega, ex,ey,ez, = args

    isotropy_check = 0
    if dx-dy < 1e-5:
        if dx-dz < 1e-5:
            isotropy_check = 1

    if isotropy_check == 0:
        taus, amps = mfuncs.calculate_anisotropic_d_amps(dx,dy,dz,ex,ey,ez)
    else:
        diso = (dx+dy+dz)/3
        taus = [1/(6*diso)]
        amps = [1]

    # here we use the form in the relax manual
    total = np.zeros(len(omega))

    for tau, amp in zip(taus, amps):

        term1 = s2/(1+(omega*tau)**2)

        term2_top = (1-sf)*(tau_f+tau)*tau_f
        term2_bot = (tau_f + tau)**2 + (omega*tau_f*tau)**2
        term2 = term2_top/term2_bot

        term3_top =  (sf-s2)*(tau_s+tau)*tau_s
        term3_bot = (tau_s + tau)**2 + (omega*tau_s*tau)**2
        term3 = term3_top/term3_bot

        total = total +  tau*amp*(term1 + term2 + term3)
    
    total = total * 0.4
    return total

def J_anisotropic_mf_old(params, args):

    '''
    This function uses a comepletely isotropic mode for the spectral density function
    with the extended model free formalism. 

    To select an isotropic model and a axial symetric one you can set the parameters 
    accordingly. 

    Take from https://link.springer.com/content/pdf/10.1023/A:1018631009583.pdf

    this was relabelled from J_anisotropic_mf to J_anisotropic_mf_old()
    this probably breaks some functionality in estimate DiffusionTesor.py
    '''

    def delta2k(da, diso, L2):
        top = da - diso
        bot = np.sqrt(diso**2 - L2) 
        return top/bot

    def amp_part2(delta,a,b,c):
        return delta*(3*(a**4)+6*(b**2)*(c**2)-1)

    # unpack some varriables
    omega, cosine_angles = args 
    ez,ey, ex, = cosine_angles

    # give shorter names for later 
    dx = params['dxx']
    dy = params['dyy']
    dz = params['dzz']

    #calculate some values 
    diso = (dx + dy + dz)/3
    # L2 = (dx*dy + dx*dz + dy*dz)/3
    L2 = (dx*dy + dx*dz+ dy*dz)/3.

    # this is to handle the isotropic case where we end up dividing by 0 ... 
    isotropy_check = False
    if np.allclose(dx, dy):
        if np.allclose(dx, dz):
            isotropy_check = True

    if isotropy_check == False:

        delta2x = delta2k(dx, diso, L2)
        delta2y = delta2k(dy, diso, L2)
        delta2z = delta2k(dz, diso, L2)
        #calculate some taus 
        tau1 = 1/(4*dx + dy + dz)
        tau2 = 1/(dx + 4*dy + dz)
        tau3 = 1/(dx + dy + 4*dz)

        tau_part1 = 6*np.sqrt(diso**2 - L2)
        tau4 = 1/(6*diso + tau_part1)
        tau5 = 1/(6*diso - tau_part1)

        taus = [tau1, tau2, tau3, tau4, tau5]

        amplitudes = []

        a1 = 3*(ey**2)*(ez**2)
        a2 = 3*(ex**2)*(ez**2)
        a3 = 3*(ex**2)*(ey**2)

        part1 = 0.25 * ( 3*(ex**4 + ey**4 + ez**4) -1)

        ## the order of ex, ey and ez needs to be checked!!!
        part2 = (1./12.)*(amp_part2(delta2x,ex,ey,ez) + \
                          amp_part2(delta2y,ey,ez,ex) + \
                          amp_part2(delta2z,ez,ex,ey))

        a4 = part1 - part2
        a5 = part1 + part2 
        amplitudes = [a1,a2,a3,a4,a5]
    else:
        tau = [1/(6*diso)]
        amplitudes = [1]

    total = 0.
    S2 = params['S2_f']*params['S2_s']
    for tau, amp in zip(taus, amplitudes):

        # tau_f_eff = params['tau_f']*tau/(params['tau_f']+tau)
        # tau_s_eff = params['tau_s']*tau/(params['tau_s']+tau)

        term1_top = amp*tau*S2
        term1_bot = 1+(omega*tau)**2
        term1_total = term1_top/term1_bot

        # term2_top = amp*tau_f_eff*(1-params['S2_f'])
        # term2_bot = 1+(omega*tau_f_eff)**2
        # term2_total = term2_top/term2_bot

        # term3_top = amp*tau_s_eff*(params['S2_f']-S2)
        # term3_bot =  1+(omega*tau_f_eff)**2
        # term3_total = term3_top/term3_bot

        total = total + term1_total #+ term2_total + term3_total

    return total


def J_axial_symetric(params, omega):

    """
    This function has the model free Jw for an axially symetric spheroid. 
    It can be used to fit D_perp, D_parrallel, S_slow, S_fast, and t_fast

    The functions are taken from: 
    https://comdnmr.nysbc.org/comd-nmr-dissem/comd-nmr-software/software/modelfree/modelfree_manual.pdf
    and 
    https://link.springer.com/content/pdf/10.1007/s10858-007-9214-2.pdf
    https://pubs.acs.org/doi/pdf/10.1021/ja00168a070
    """

    # unpack the omega to get the angle we want
    omega, angle = omega

    d_perp = params['dperpendicular'] # perpendicular part 
    d_para = params['dparallel']

    # correlartion times from diffusion
    tau1 = 1./(6*d_perp)
    tau2 = 1/(5*d_perp + d_para)
    tau3 = 1/(2*d_perp + 4 * d_para)

    # group them together
    taus = [tau1, tau2, tau3]

    # amplitudes
    a1 = 0.25*((3*mfuncs.cos_square(angle)-1)**2)

    cos_square_sin_square = mfuncs.cos_square(angle)*mfuncs.sin_square(angle)
    a2 = 3*cos_square_sin_square
    a3 = (3./4.)*(mfuncs.sin_square(angle)**2)

    #join the amplitudes together; 
    amps = [a1, a2, a3]

    total = 0.
    S2 = params['S2_f']*params['S2_s']
    for tau, amp in zip(taus, amps):

        # tau_f_eff = params['tau_f']*tau/(params['tau_f']+tau)
        # tau_s_eff = params['tau_s']*tau/(params['tau_s']+tau)

        term1_top = amp*tau#*S2
        term1_bot = 1+(omega*tau)**2
        term1_total = term1_top/term1_bot

        # term2_top = amp*tau_f_eff*(1-params['S2_f'])
        # term2_bot = 1+(omega*tau_f_eff)**2
        # term2_total = term2_top/term2_bot

        # term3_top = amp*tau_s_eff*(params['S2_f']-S2)
        # term3_bot =  1+(omega*tau_f_eff)**2
        # term3_total = term3_top/term3_bot

        total = total + term1_total #+ term2_total + term3_total

    return (2./5.)*total


def spectral_density_anisotropic_exponencial_sum(params, args):
    '''
    The aim of this function is to take internal correlation funstion from MD 
    (see self.spectral_density_fuction and spectral_density_fuction) and to apply and anisotropic diffusion 
    so that one could then calculate NMR relaxation rates.

    maybe I should move this to the spectral density module. 

    here we use a set of exponencial decays doing from time_1 to time_<N> with the amplitudes amp_1 to amp_<N>

    '''

    # unpack varriables
    dx = params['dx']
    dy = params['dy']
    dz = params['dz']

    omega, ex,ey,ez, = args 
    taus, amps = mfuncs.calculate_anisotropic_d_amps(dx,dy,dz,ex,ey,ez)
    curve_count = sum([1 if "amp" in i else 0 for i in params.keys()])

    try:
        total = np.zeros(len(omega))
    except TypeError:
        # this arises in the case where omega has a lenth of 1. 
        total = 0

    for taui, ampi in zip(taus, amps):

        taui = taui
        #diffusion part
        term1_top = ampi * taui * params['S_long']
        term1_bottom = 1+(omega*taui)**2
        term1_total = term1_top/term1_bottom
        total = total + term1_total

        # this check means we dont do the sum for internal motions when it is not there 
        # it also means we can just switch off the internal motions by setting S_long = 1
        # and not worry about the params['amp_%i'%(i)] terms.
        if params['S_long'] != 1:
            #now we do the internal motions 
            for i in range(curve_count):
                i = i + 1

                #correction for fitting the correlation times in ps
                tau_internal = params['time_%i'%(i)]
                amp_internal = params['amp_%i'%(i)]

                tau_eff = taui*tau_internal/(tau_internal+taui)
                term_internal = amp_internal*ampi*tau_eff/(1+(omega*tau_eff)**2)
                
                total = total + term_internal

    return (2./5.)*total

# def J_exponencial_sum(params, omega):
#     """J_exponencial_sum calculates the spectral density using a
#     functional form that is a sum of decays similar to that in

#     Fitting Side-Chain NMR Relaxation Data Using Molecular Simulations
#     Felix Kümmerer, Simone Orioli, David Harding-Larsen, Falk Hoffmann,
#     Yulian Gavrilov, Kaare Teilum, and Kresten Lindorff-Larsen*
#     https://doi.org/10.1021/acs.jctc.0c01338


#     and it also uses a ellipsoid diffusion tensor as described in
#     Ab Initio Prediction of NMR Spin Relaxation Parameters from Molecular Dynamics Simulations
#     Po-chia Chen*†Orcid, Maggy Hologne‡, Olivier Walker‡, and Janosch Hennig†
#     https://pubs.acs.org/doi/pdf/10.1021/acs.jctc.7b00750

#     To use this class params needs to be a dictionary with the following entries:
#     'S_long' - long time limit order parameter
#     'amp_<n>' - amplitudes for each exponential decay
#     'time_<n>' - time in ps for each exponential decay
#     'd_par' - parallel component of the diffusion tensor
#     'd_per' - perpendicular component of the diffusion tensor
#     'theta' - this should be the average angle between the XY vector and d_par, in radians

#     the d and a definitions are taken from this paper
#     https://link.springer.com/content/pdf/10.1023/A:1018631009583.pdf
#     and https://link.springer.com/article/10.1023/A:1011283809984?noAccess=true
#     """

#     # get how many amps we have
#     curve_count = 0
#     for i in params.keys():
#         if 'amp' in i:
#             curve_count = curve_count+1

#     # spectral density stuff
#     d1 = 6*params['d_per']
#     d2 = 5*params['d_per'] + params['d_par']
#     d3 = 2*params['d_per'] + 4*params['d_par']
#     d_list = [d1,d2,d3]

#     a1 = (((3*np.cos(params['theta'])**2)-1)**2)/4
#     a2 = 3*(np.sin(params['theta'])**2) * (np.cos(params['theta'])**2)
#     a3 = 0.75*(np.sin(params['theta'])**4)
#     a_list = [a1,a2,a3]

#     if type(omega) == np.ndarray:
#         total = np.zeros(len(omega))
#     else:
#         total = 0.

#     for ak, dk in zip(a_list,d_list):

#         #tumbling part!
#         tumb = params['S_long']*dk/(dk**2 + omega**2)

#         if type(omega) == np.ndarray:
#             internal_sum = np.zeros(len(omega))
#         else:
#             internal_sum = 0.

#         # these are the internal motions
#         for i in range(curve_count):
#             i = i + 1
#             ampi = 'amp_%i'%(i)
#             timei = 'time_%i'%(i)
#             internal_top = params[ampi]*(dk+params[timei])

#             internal_bot = (dk+params[timei])**2 + omega**2
#             internal = internal_top/internal_bot
#             internal_sum = internal_sum + internal

#         total = total + ak*(tumb + internal_sum)

#     return total

