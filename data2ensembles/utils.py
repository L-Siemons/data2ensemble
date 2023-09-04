import numpy as np
import pkg_resources
import scipy 
import scipy.constants 
from scipy.spatial.transform import Rotation as scipyRotation
import string
import numpy as np
import re
import pickle as pic

class PhysicalQuantities(object):
    """PhysicalQuantities contains properties of NMR atom types and
    physical constants"""
    def __init__(self):

        #set the gyromagnetic ratios 
        self.gamma = {}
        self.gamma['h'] = 267.522e6
        self.gamma['n'] = -27.126e6
        self.gamma['c'] = 67.2828e6
        self.gamma['p'] = 108.291e6

        # some physical quantities
        self.mu0 = scipy.constants.mu_0 # 4*np.pi*(10**-7) # N/A2
        self.h = scipy.constants.h #6.62607004e-34 #m2  = kg / s
        self.hbar = scipy.constants.hbar
        
        #read the CSA 
        self.csa_axially_symetric = {}
        file = pkg_resources.resource_filename('data2ensembles', 'dat/csa_dft_v2.dat')
        file = open(file, 'r')
        for i in file.readlines():
            if i[0] != '#':
                s = i.split('#')[0].split()
                if len(s) == 3:
                    key = (s[0], s[1])

                    if float(s[2]) > 1e-2: 
                        print('WARNING: The csa is rather large, did you miss a e-6?')
                        print(i)
                    self.csa_axially_symetric[key] = float(s[2])
        file.close()

        # read in the anisotropic CSA
        self.csa_anisotropic = {}
        file = pkg_resources.resource_filename('data2ensembles', 'dat/csa_anisotropic.txt')
        file = open(file, 'r')
        for i in file.readlines():
            if i[0] != '#':
                s = i.split('#')[0].split()
                if len(s) == 5:
                    key = (s[0], s[1])

                    csa_values = [float(s[a])*(2/3) for a in [2,3,4]]
                    if float(max(csa_values)) > 1e-2: 
                        print('WARNING: The csa is rather large, did you miss a e-6?')
                        print(csa_values)
                    self.csa_anisotropic[key] = csa_values
        file.close()

        #def read in bond lengths
        self.bondlengths = {}
        file = pkg_resources.resource_filename('data2ensembles', 'dat/dft_bond_lengths_v2.dat')
        file = open(file, 'r')
        for i in file.readlines():
            if i[0] != '#':
                s = i.split('#')[0].split()
                if len(s) == 3:
                    key = (s[0], s[1])
                    self.bondlengths[key] = float(s[2])
        file.close()

    def calc_omega(self, x, fields):
        '''
        Calculate the frequency in angular momentum
        Since the gyromagnetic ratio here is given in rad T-1 s-1 we dont need the 2pi unit
        '''
        return self.gamma[x] * fields

    def calc_dd(self,x,y,r):
        '''
        x - atom which must be in the dectionary self.gamma
        y - atom which must be in the dectionary self.gamma
        r - distance between x and y in meters

        Atom y is ussually the hydrogen and is the passive atom when calculating R_XY
        '''

        # scalling factor for the dipolar interaction 
        d_top = self.mu0 * self.hbar * self.gamma[x] * self.gamma[y]
        d_bot = 4*np.pi*(r**3)
        d = d_top/d_bot

        return d

    def calc_axially_symetric_csa(self, field, x, csa_atom_name):
        '''
        Calculate the isotropic CSA
        field - magnetic field in tesla
        x - atom
        csa_atom_name - atom name in the csa dictionary
        '''

        # the factor of 2pi is needed to keep it in omega!
        omega = self.calc_omega(x, field)
        csa_total = omega*self.csa_axially_symetric[csa_atom_name]/np.sqrt(3)
        return csa_total

    def calc_aniso_csa_prefactor(self, field, x, csa_atom_name, square=True):
        '''
        Calculate the isotropic CSA
        field - magnetic field in tesla
        x - atom
        csa_atom_name - atom name in the csa dictionary
        '''

        # the factor of 2pi is needed to keep it in omega!
        omega = self.calc_omega(x, field)
        if square == True:
            csa_total = (2/36)*omega**2
            return csa_total
        else:
            csa_total = (2/36)*omega
            return csa_total


def read_nmr_relaxation_rate(file):
        print(f'Reading in data from {file}')
        data = {}
        header_check = False

        #start reading the file
        f = open(file)
        for i in f.readlines():
            if i.split()[0][0] != '#':
                s = i.split()
                data[s[0]] = np.array([float(a) for a in s[1:]])
            else:
                data_headers = i
                header_check = True
        f.close()

        # good to label your files correctly!
        if header_check == False:
            raise NoHeader(f'The file {file} does not have a header!')

        return data, data_headers

def read_fitted_spectral_density(f):
    '''
    This function reads the spectral density that was written by
    AnalyseTrr.fit_all_correlation_functions()

    it produces a nested dictionary data[residue] = params
    '''

    data= {}
    f = open(f)
    for i in f.readlines():
        if i[0] != '#':
            s = i.split(":")
            if len(s) > 1 :
                resid = int(s[0])
                atoms = s[1]
                atoms_reversed = ','.join(list(reversed(atoms.split(','))))
                S_long = float(s[2])
                time = [float(j) for j in s[4].split(',')]
                amps = [float(j) for j in s[3].split(',')]

                params = {}
                params['S_long'] = S_long
                for indx , (timei,ampi) in enumerate(zip(time, amps)):

                    indx = indx+1
                    amp_tag = 'amp_%i' %(indx)
                    time_tag = 'time_%i' %(indx)

                    params[amp_tag] = ampi
                    params[time_tag]  = timei

                data[(resid, atoms)] = params
                # this is so we dont have to check the direction of 
                # the spectral density function each time, uses more memory but saves coding later
                data[(resid, atoms_reversed)] = params

    f.close()
    return data

def read_diffusion_tensor(file):
    f = open(file)
    line = f.readlines()[1]
    s = line.split()
    #given in dx,dy,dz
    values = [float(s[a]) for a in [1,4,7]]

    try:
        errors = [float(s[a]) for a in [2,5,7]]
    except ValueError:
        print('errors are not floats - might be NONE')
        errors = [s[a] for a in [2,5,7]]
    return values, errors

def read_csa_orientation_info():
    
    file = pkg_resources.resource_filename('data2ensembles', 'dat/csa_orientations.dat')
    f = open(file)
    data = {}
    for i in f.readlines():
        if i[0] != '#':
            s = i.split()

            #index by resnae
            if s[2] not in data:
                data[s[2]] = {}
            #index by atom type
            data[s[2]][s[1]] = s

    f.close()
    return data


def is_right_handed(M):
    """Is matrix right handed"""
    det = np.linalg.det(M)
    return np.allclose(det / np.abs(det), 1)


def make_right_handed(M):
    """
    excepts vectors in M are column vectors
    """
    if is_right_handed(M):
        return M
    # don't modify original
    M = M.copy()
    for i in range(3):
        M[:, i] *= -1
        if is_right_handed(M):
            return M
    raise RuntimeError("Couldn't find simple conversion to right-hand system")

def get_alignment_rotation(matrix, return_status):

    '''
    The rotation from one coordinate system to the reference system
    reference_system = np.array([[1,0,0],[0,1,0],[0,0,1]])
    '''

    #Now we check if U is right handed
    right_hand_status = is_right_handed(matrix)
    print(f'Matrix of eigenvectors is right handed: {right_hand_status}')
    if right_hand_status == False:
        raise('Some code needs to be written to handle this! \nSee the class PrepareReference() - Sorry, Lucas')

    reference_system = np.array([[1,0,0],[0,1,0],[0,0,1]])
    right_hand_status = is_right_handed(reference_system)

    # find the rotation between one frame and the other
    frame_rotation, rmsd = scipyRotation.align_vectors(reference_system, matrix)


    if return_status == 'rotvec':
        rot_vec = frame_rotation.as_rotvec(degrees=True)*-1
        return rot_vec

    else:
        raise('get_alignment_rotation() is returning None, this should not happen! Exiting')

def get_atom_info_from_rate_key(key):

        atom1_key = key.split('-')[0]
        atom2_key = key.split('-')[1]

        atom1_letters =  [i for i in atom1_key if i.isalpha()]
        atom2_letters =  [i for i in atom2_key if i.isalpha()]

        atom1_numbers = ''.join(i if i.isdigit() or i in string.punctuation else ' ' for i in atom1_key).split()
        atom2_numbers = ''.join(i if i.isdigit() or i in string.punctuation else ' ' for i in atom2_key).split()

        atom1_res_type = atom1_letters[0]
        atom2_res_type = atom2_letters[0]

        atom1_resid = atom1_numbers[0]
        atom2_resid = atom2_numbers[0]

        atom1_type = atom1_letters[1] + atom1_numbers[1]
        atom2_type = atom2_letters[1] + atom2_numbers[1]

        return atom1_letters, atom1_numbers, atom1_res_type, atom1_resid, atom1_type, atom2_letters, atom2_numbers, atom2_res_type, atom2_resid, atom2_type

def get_atom_info_from_tag(i):
    '''
    get the atom info from the tag
    '''
    split1 = i.split('-')[0]
    split2 = i.split('-')[1]

    atom2 = re.split('(\d+)', split1)
    atom2 = ''.join(atom2[2:])
    
    atom1 = re.split('(\d+)', split2)
    atom1 = ''.join(atom1[2:])
    
    resid = int(re.split('(\d+)', split2)[1])
    resname = re.split('(\d+)', split2)[0]

    res_info = (resid, resname, atom1, atom2)
    return res_info

def load_pickle(file):
    with open(file, 'rb') as handle:
        unserialized_data = pic.load(handle)
    return unserialized_data

def dict_with_full_keys_to_resid_keys(dictionary, atom_type):
    '''
    This takes a dictionary with keys like T9H1'-T9C1'
    and makes a dictionary with only the resid for A given atom type
    '''

    new = {}
    new_resinfo = {}

    for i in dictionary:
        if atom_type in i:
            res_info = get_atom_info_from_tag(i)
            resid, resname, atom1, atom2 = res_info
            new[resid] = dictionary[i]
            new_resinfo[resid] = get_atom_info_from_tag(i)

    return new, new_resinfo

def resinto_to_tag(resid, resname, atom1, atom2):
    return f"{resname}{resid}{atom2}-{resname}{resid}{atom1}"

def np2string_formatter(x):
    '''
    a function to pass np2string to format the flaots nicely
    '''
    
    formatted = "{:.3f}".format(x)  # Format the float with fixed decimal places
    if formatted == '0.000':
        formatted = ' 0.   '

    elif '-' not in formatted:
        formatted = ' ' + formatted
    return formatted
