import numpy as np 
from scipy.spatial.transform import Rotation as R

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

    #print('CHECK THE ORDER OF EX, EY,EZ')
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


