
import numpy as np 


# some trig functions
def sin_square(a):
    b = np.sin(a)
    return b**2

def cos_square(a):
    b = np.cos(a)
    return b**2

def cosine_angles(vec, axis):

    p1,p2,p3 = axis

    mag = np.linalg.norm(vec)
    ex = np.dot(vec, p1)/mag
    ey = np.dot(vec, p2)/mag
    ez = np.dot(vec, p3)/mag

    return ex,ey,ez

def delta2k(da, diso, L2):
    top = da - diso
    bot = np.sqrt(diso**2 - L2) 
    return top/bot

def amp_part2(delta,a,b,c):
    return delta*(3*(a**4)+6*(b**2)*(c**2)-1)

def calculate_anisotropic_d_amps(dx, dy,dz, ex,ey,ez):

    '''
    When looking at anisotropic spectral density functions we 
    often need to determine effective correlation times and amplitudes 
    to replace the isotropic tc
    '''

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
        taus = [1/(6*diso)]
        amplitudes = [1]

    return taus, amplitudes
