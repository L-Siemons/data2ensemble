import numpy as np 
from scipy.spatial.transform import Rotation as R
from scipy import linalg as scilinalg
import math
import cython
cimport numpy as np

# some trig functions
def sin_square(a):
    b = np.sin(a)
    return b**2

def cos_square(a):
    b = np.cos(a)
    return b**2

def distance(a,b):
    '''
    calculates the distance between two points - this is here incase I want to speed up with 
    cython
    '''
    return math.dist(a,b)

def cosine_angles(vec, axis):

    p1,p2,p3 = axis
    mag = np.linalg.norm(vec)
    ex = np.dot(vec, p1)/mag
    ey = np.dot(vec, p2)/mag
    ez = np.dot(vec, p3)/mag

    #print('CHECK THE ORDER OF EX, EY,EZ')
    return [ex,ey,ez]

def delta2k(float da, float diso, float L2):
    cdef float top = da - diso
    cdef float bot = np.sqrt(diso**2 - L2) 
    return top/bot

def amp_part2(float delta, float a,float b, float c):
    return delta*(3*(a**4)+6*(b**2)*(c**2)-1)

def calculate_anisotropic_d_amps(dx, dy,dz, float ex,float ey, float ez):

    '''
    When looking at anisotropic spectral density functions we 
    often need to determine effective correlation times and amplitudes 
    to replace the isotropic tc
    '''

    #calculate some values 
    cdef float diso = (dx + dy + dz)/3
    # L2 = (dx*dy + dx*dz + dy*dz)/3
    cdef float L2 = (dx*dy + dx*dz+ dy*dz)/3.
    cdef int isotropy_check = 1
    
    # this is to handle the isotropic case where we end up dividing by 0 ... 
    if dx - dy < 1e-5:
        if  dx - dz < 1e-5:
            isotropy_check = 0

    cdef float delta2x 
    cdef float delta2y
    cdef float delta2z

    cdef float tau1
    cdef float tau2
    cdef float tau3
    cdef float tau4
    cdef float tau5
    cdef float tau_part1

    cdef float a1
    cdef float a2
    cdef float a3
    cdef float a4
    cdef float a5

    cdef float part1
    cdef float part2 

    cdef list amplitudes
    cdef list taus
    
    if isotropy_check == 1:

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

# def matrix_exp(np.ndarray[np.float_t, ndim=2] a):
def matrix_exp(a):

    # this should give the same result as scilinalg.expm() 
    # they might have different ordering due to the ordering of the eigen values
    # cdef np.ndarray[np.float_t, ndim=1] eig
    # cdef np.ndarray[np.float_t, ndim=2] eigvec
    # cdef np.ndarray[np.float_t, ndim=2] final_matrix

    eig, eigvec = np.linalg.eig(a)
    final_matrix = eigvec @ np.diag(np.exp(-eig)) @ np.linalg.inv(eigvec)
    return final_matrix

def construct_operator(matricies, times, product=True):

    # the order in which these operators are listed I *think* is correct once we reverse them 
    # also check that the axis rolling is not transposing the matrix. Probably should look at this 
    # with an example
    
    # to deal with weather we have alist of matricies or just one.
    # cdef float t
    # cdef np.ndarray[np.float_t, ndim=2] r
    # cdef np.ndarray[np.float_t, ndim=2] final_poperator
    # cdef np.ndarray[np.float_t, ndim=3] roll 
    # cdef list operators

    if matricies.shape[-1] != 1:
        operators = [matrix_exp(-1.*r*t) for r,t in zip(np.rollaxis(matricies, 2), times)]
    elif matricies.shape[-1] == 1: 
        roll = np.rollaxis(matricies, 2)
        operators = [matrix_exp(-1*roll[0]*t) for t in times]

    # do we want to return the product of the list
    if product == True:
        final_poperator = np.linalg.multi_dot(np.flip(operators))
        return final_poperator
    else:
        return operators

def calc_csa_axis(resid, atomname, csa_orientations_info, uni):
    '''
    This function takes the CSA orientations info taken from read_csa_orientation_info
    and determines the axis of the CSA tensor in the molecular frame

    Note that two definitions are used. These are refered to 'TYPE 1' and 'TYPE 2'
    TYPE 1 : Bryce, D. L.; Grishaev, A.; Bax, A. J. Am. Chem. Soc. 2005, 127, 7387-7396.
    TYPE 2 : Ying, J.; Grishaev, A.; Bryce, D. L.; Bax, A. J. Am. Chem. Soc. 2006, 128, 11443-11454.
    '''

    type_ = csa_orientations_info[0]
    if type_ == '1':

        # unpack the information
        csaAtom = csa_orientations_info[1]
        atom1, atom2, atom3 = csa_orientations_info[3:6]

        #in xyz order
        d11_cosine_angles = [float(a) for a in csa_orientations_info[6:9]]
        d22_cosine_angles = [float(a) for a in csa_orientations_info[9:12]]
        d33_cosine_angles = [float(a) for a in csa_orientations_info[12:15]]
    
    if type_ == '2':
        # unpack the information
        csaAtom = csa_orientations_info[1]
        atom1, atom2, atom3 = csa_orientations_info[3:6]
        angle = np.deg2rad(float(csa_orientations_info[6]))

    #these should be some general things to calculate
    res = uni.select_atoms(f'resid {resid}')
    pos1  = res.select_atoms(f"name {atom1}").positions[0]
    pos2  = res.select_atoms(f"name {atom2}").positions[0]
    pos3  = res.select_atoms(f"name {atom3}").positions[0]

    vec1 = (pos1 - pos2)/np.linalg.norm(pos1 - pos2)
    vec2 = (pos1 - pos3)/np.linalg.norm(pos1 - pos3)

    #print('>>>',atom1, atom3,pos1, pos2, pos3, pos1 - pos2, pos1 - pos3)
    #print(vec1, vec2)

    if type_ == '1':

        # this defines the local axis system 
        # bisect the angle atom1 atom2 atom3 
        x_local = -1*(vec2+vec1)/np.linalg.norm(vec2+vec1)

        # z is defined as perpendicular to the plane formed by these three atoms
        z_local = np.cross(vec1, vec2)
        z_local = z_local/np.linalg.norm(z_local)

        # y is the cross of x and y
        y_local = np.cross(x_local, z_local)*-1
        y_local = y_local/np.linalg.norm(y_local)
        local_system = [x_local, y_local, z_local]

        # now we need to transform this to the axis system of the CSA tensor
        d11 = np.linalg.solve(local_system, d11_cosine_angles)
        d22 = np.linalg.solve(local_system, d22_cosine_angles)
        d33 = np.linalg.solve(local_system, d33_cosine_angles)

        d11 = d11/np.linalg.norm(d11)
        d22 = d22/np.linalg.norm(d22)
        d33 = d33/np.linalg.norm(d33)

        d = (d11, d22,d33)
        d_selected_axis = d[int(csa_orientations_info[15])]

        return d_selected_axis, d

    if type_ == '2':

        #define the local geometry
        x_local = vec1*(-1)

        z_local = np.cross(vec1, vec2)*(-1)
        z_local = z_local/np.linalg.norm(z_local)

        y_local = np.cross(x_local, z_local)*-1
        y_local = y_local/np.linalg.norm(y_local)

        #now we can apply the rotation with a rotation vector. 
        r = R.from_rotvec(z_local*angle)
        
        d11 = r.apply(x_local)
        d22 = r.apply(y_local)
        # d33 is the same as z_local as the rotation does not affect it
        # I have defined it here to be explicit 
        d33 = z_local 

        d11 = d11/np.linalg.norm(d11)
        d22 = d22/np.linalg.norm(d22)
        d33 = d33/np.linalg.norm(d33)
        
        d = (d11, d22,d33)
        d_selected_axis = d[int(csa_orientations_info[7])]

        return d_selected_axis, d


