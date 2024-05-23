
import data2ensembles.utils as utils
import data2ensembles.mathFuncs as mathFunc
import data2ensembles.rates
import data2ensembles as d2e
import data2ensembles.structureUtils as strucUtils
import data2ensembles.modelFree as mf
import data2ensembles.shuttling as shuttling
import data2ensembles.relaxation_matricies as relax_mat
import data2ensembles.spectralDensity as specDens
import data2ensembles.correlationFunctions as correlation_functions
#import data2ensembles.csa_spec_dens as csa_spec_dens


import os
import numpy as np
#import sys
import random as r
import copy
from tqdm import tqdm
import MDAnalysis as md
from MDAnalysis import transformations

import pickle as pic 
from lmfit import Minimizer, Parameters, report_fit
from tqdm import tqdm
import enlighten
import toml
import itertools

import scipy
import scipy.fft
import scipy.interpolate
import scipy.spatial
import scipy.spatial.transform
import glob
import matplotlib.pyplot as plt
import sys

PhysQ = utils.PhysicalQuantities()

class PrepareReference():
    '''
    Before analysis we might want to prepare the reference structure in some way
    '''
    def __init__(self, gro):

        self.gmx = 'gmx'
        self.gro = gro
        self.principal_axes_selection = 'all'
        self.diffusing_xtc = None

    def align_to_inertia_axis(self, outname='test.gro'):

        uni = md.Universe(self.gro)
        selection = uni.select_atoms(self.principal_axes_selection)
        moment_of_inertia = selection.moment_of_inertia()

        #these give x,y,z where the largest eigenvalue is along z
        eigen_values, eigen_vecs = np.linalg.eigh(moment_of_inertia)
        indices = np.argsort(eigen_values)
        #eigen_values = eigen_values[indices]
        U = eigen_vecs[:, indices]
        print(U)

        rot_vec = utils.get_alignment_rotation(U, 'rotvec')
        degrees_to_rotate = np.linalg.norm(rot_vec)
        print('degrees to rotate:', degrees_to_rotate)

        #now apply this rotation to the molecule
        ts = uni.trajectory.ts
        rotated = md.transformations.rotate.rotateby(degrees_to_rotate, direction=rot_vec, ag=selection)(ts)
        selection.write(outname)


class AnalyseTrr():
    """A class to calculate properties of the trajectory note this
    class requires gromacs and MDAnalysis"""
    def __init__(self, tpr, gro, path_prefix):

        self.gmx = 'gmx'
        self.tpr = tpr
        self.xtc = None
        self.unaligned_xtc = None
        self.gro = gro
        self.path_prefix = path_prefix
        self.relaxometry_experiment_details_file = None
        self.relaxometry_inensities = {}
        self.trr_time_step = 1e-12 # assume that the trajectory time step is 1 ps 


        # the number of curves to use when fitting the internal correlation functions
        self.curve_count = 3
        # This is a dummy correlation time used to help fit the internal correlation times
        # note that this should be seconds
        self.dummy_tauc = 5e-6

        # this dictionary contains the transform information from a local geometry to
        # the principle axis of the CSA tensor
        # this is created by  calc_cosine_angles
        self.csa_tensor_transform = None

        # to account for machine precision we might need to round some times 
        # this should be a few orders of magnitude smaller than the time step in your trr
        self.decimal_round = 15

        #how often should we write out pdfs? 
        self.plotfraction = 1

        # cutoff distances for atom pairs 
        self.atom_pair_distance_cutoff = 4e-10

    def write_diffusion_trace(self, params, file):

        folder = f"{self.path_prefix}_diffusion_rotacf_fit/"
        name = f"{folder}{file}"
        
        if os.path.isdir(folder) == False:
            os.mkdir(folder)

        print(f'writing diffusion tensor to {name}')
        f = open(name, 'w')
        f.write(f"#component value error\ndx: {params['dx'].value} {params['dx'].stderr} dy: {params['dy'].value} {params['dy'].stderr} dz: {params['dz'].value} {params['dz'].stderr}")
        f.close()

    def load_trr(self):
        '''
        load the trajectory
        '''

        if self.xtc == None:
            uni = md.Universe(self.gro)
        elif self.xtc != None and self.tpr != None:
            uni = md.Universe(self.gro, self.xtc)
        else:
            print('Something went wrong ... not sure what to load in')
        
        # make a dictionary they converts from the resid to the restype
        self.uni = uni
        self.resid2type = {}

        # this seems pretty ugly ...
        all_atom_res_types = self.uni.atoms.residues.resnames
        all_atom_res_ids = self.uni.atoms.residues.resids

        for i,j in zip(all_atom_res_ids, all_atom_res_types):
            if i not in self.resid2type:
                self.resid2type[i] = j

    def make_ndx_file(self, atom_info, index_name,supress=True,recalc=False):
        '''
        make the index file for the gmx rotacf command
        '''
        indx = open(index_name,'w')
        indx.write('[indx]\n')
        atom_info = self.make_atom_pairs_list(atom_info)
        include_counter = 0
        exclude_counter = 0
        already_existing = 0
        existing_files = glob.glob(f'{self.path_prefix}_rotacf/rotacf'+f'_*_*_*_*.xvg')
        self.gmx_atom_info = []
        print(f'Atom info length in making index: {len(atom_info)}')

        for res1, res2, atom_name1, atom_name2 in atom_info:
            #print(f'resid {res1} and name {atom_name1}\nresid {res2} and name {atom_name2}')
            xvg_out = f'{self.path_prefix}_rotacf/rotacf'+f'_{res1}_{atom_name1}_{res2}_{atom_name2}.xvg'
            
            if xvg_out not in existing_files or recalc==True:
                    
                a = self.uni.select_atoms(f'resid {res1} and name {atom_name1}')
                b = self.uni.select_atoms(f'resid {res2} and name {atom_name2}')

                key = (res1, atom_name1, atom_name2)

                if self.md_distances[key] <= self.atom_pair_distance_cutoff:

                    #python numbering!
                    a_indx = a[0].ix+1
                    b_indx = b[0].ix+1
                    include_counter = include_counter + 1
                    indx.write(f'{a_indx} {b_indx}\n')
                    self.gmx_atom_info.append([res1, res2, atom_name1, atom_name2])
                else: 
                    exclude_counter = exclude_counter + 1
           
            else:
                already_existing = already_existing + 1

        indx.close()
        print(f"Already existing ACFs: {already_existing}")
        print(f"Total ACFs in GMX index file: {include_counter}")
        print(f"contects excluded for being over the cutoff {exclude_counter}")
        print(f"cutoff distance {self.atom_pair_distance_cutoff}")

    def get_number_of_blocks(self,atom_names):
        '''
        get the number of blocks that have residue 1 and the atom names in them
        we subtract 1 because often the last block is not complete so we do not want to analyze it
        '''
        atom1 = atom_names[0]
        atom2 = atom_names[1]
        files = glob.glob(f"{self.path_prefix}_rotacf/rotacf_block_*_1_{atom1}_1_{atom2}.xvg")
        return len(files) - 1

    def read_gmx_xvg(self,f):
        '''
        read xvg file, assuming that is all float
        '''
        values = []
        #print('opening file ...')
        fi = open(f)
        #print(f'Reading; {f}')
        for i in fi.readlines():
            if i[0] not in ('#', '@'):
                s = i.split()
                values.append([float(j) for j in s])

        fi.close()
        return np.array(values)

    def read_csa_spectra_density_params(self, blocks=False):

        if blocks == False:
            csa_xx_spec_file = self.path_prefix + '_fit_params/internal_csa_xx_correlations_0.dat'
            csa_yy_spec_file = self.path_prefix + '_fit_params/internal_csa_yy_correlations_0.dat'
            csa_xy_spec_file = self.path_prefix + '_fit_params/internal_csa_xy_correlations_0.dat'
        else:
            print('LUCAS YOU NEED TO CODE THIS!')

        csa_xx_spectral_density_params = utils.read_fitted_spectral_density(csa_xx_spec_file)
        csa_yy_spectral_density_params = utils.read_fitted_spectral_density(csa_yy_spec_file)
        csa_xy_spectral_density_params = utils.read_fitted_spectral_density(csa_xy_spec_file)

        return csa_xx_spectral_density_params, csa_yy_spectral_density_params, csa_xy_spectral_density_params

    def add_diffution_spectral_density_params(self, params, diffusion):

        for indx, i in enumerate(['dx', 'dy', 'dz']):
            params[i] = diffusion[indx]

        return params

    def make_atom_pairs_list(self, atoms, distance_cutoff=True):
        '''
        Many functions want to iterate over residues and atom types. This function created this list
        This function returns a list where of atom pair infos

        [[res1 res2 atomname1 atomname2] ... N]

        some functions apply a distance cutoff and when we make the atoms pair list 
        we want to be able to take this into account
        '''

        # I could probably think of a better wat to do this
        atom_info = []
        for i in self.uni.residues:

            uni_res_atom_names = [k.name for k in  i.atoms]
            for a in atoms:

                check = a[0] in uni_res_atom_names and a[1] in uni_res_atom_names

                if distance_cutoff == True and check == True:
                    key = (i.resid, a[0], a[1])
                    dist = self.md_distances[key]

                    if dist > self.atom_pair_distance_cutoff:
                        check = False

                if check == True:
                    atom_info.append([i.resid, i.resid, a[0], a[1]])

        return atom_info

    def calc_inter_atom_distances(self, atom_names, skip=1000):

        print('calculating MD distances')
        atom_info = self.make_atom_pairs_list(atom_names, distance_cutoff=False)
        selections = {}
        distances = {}
        for res1, res2, atom_name1, atom_name2 in atom_info:

            key = (res1, atom_name1, atom_name2)
            sel1 = f'resid {res1} and name {atom_name1}'
            sel2 = f'resid {res2} and name {atom_name2}'
            sele = [self.uni.select_atoms(sel1)[0], self.uni.select_atoms(sel2)[0]]
            selections[key] = sele
            distances[key]= []

        for ts in tqdm(self.uni.trajectory[::skip]):

            for pair in selections:

                pos1 = selections[pair][0].position
                pos2 = selections[pair][1].position
                dist = mathFunc.distance(pos1, pos2)
                distances[pair].append(dist*1e-10)

        keys_list = list(distances.keys())
        for i in keys_list:
            distances[i] = np.mean(distances[i])#*1e-10
            i_rev = (i[0], i[2], i[1])
            distances[i_rev] = distances[i]
            #print(i , distances[i])

        self.md_distances = distances

    def calc_cosine_angles(self,
                        atom_names, 
                        calc_csa_angles=True, 
                        csa_ignore_list=[], 
                        skip=1000, 
                        calc_average_structure=True, 
                        write_out_angles=False, 
                        delete_rmsf_file=True, 
                        reference_gro_file = None,
                        dt = 1000, 
                        use_reference=False, 
                        print_axis=False):

        # try calculating the principle axis with gromacs
        # then average them and determine all the angles and 
        # average those

        # def average_principle_axis(a):

        #     a = np.average(a,axis=0) 
        #     a = a[1:4]
        #     return a



        rmsf_file = self.path_prefix+'_rmsf.xvg'

        average_pdb = self.path_prefix+'_average.pdb'
        # calculate average structure
        if calc_average_structure == True:
            gmx_command = f'{self.gmx} rmsf -f {self.xtc} -s {self.gro} -dt {dt} -ox {average_pdb} -o {rmsf_file} << EOF\n 0 \nEOF'
            os.system(gmx_command)

        if delete_rmsf_file:
            try:
                os.remove(rmsf_file)
            except FileNotFoundError:
                pass

        if use_reference == False:
            self.average_uni = md.Universe(average_pdb)
        else:
            self.average_uni = md.Universe(reference_gro_file)

        # here the larges axis is axis[2] this should be considered to be 'Dz' from the diffusion tensor
        # or in the case of a sphereoid D||
        axis = self.average_uni.select_atoms('all').principal_axes()

        #this file can be written out as a check if desired
        labels = ['large', 'middle', 'small']
        if write_out_angles == True:
            angles_out = open(self.path_prefix + '_dipolar_angles.dat', 'w')
            
            #write out the principle axis
            pas_out = open(self.path_prefix + '_principle_axis.dat', 'w')
            for i,j in zip(axis, labels):
                #this should go x,y,x
                pas_out.write(f'{j} {i[0]:0.2f} {i[1]:0.2f} {i[2]:0.2f}\n')
            pas_out.close()

        if print_axis == True:
            print('These are the principle axis of the system')
            for i,j in zip(axis, labels):
                #this should go x,y,x
                print(f'{j} {i[0]:0.2f} {i[1]:0.2f} {i[2]:0.2f}')

        self.cosine_angles = {}
        atom_info = self.make_atom_pairs_list(atom_names)

        # does the CSA have a different bond vector ? 
        if self.csa_tensor_transform == None:
            self.csa_tensor_transform = utils.read_csa_orientation_info()
            self.csa_cosine_angles = {}

        for res1, res2, atom_name1, atom_name2 in atom_info:

            atom1_sele = self.average_uni.select_atoms(f'resid {res1} and name {atom_name1}')[0]
            atom2_sele = self.average_uni.select_atoms(f'resid {res2} and name {atom_name2}')[0]
            atom1_resname = atom1_sele.resname[1]

            atom1_poss = atom1_sele.position
            atom2_poss = atom2_sele.position
            
            vec = atom2_poss - atom1_poss
            angs = mathFunc.cosine_angles(vec, axis)

            key = (res1, atom_name1, res2, atom_name2)
            key_rev = (res1, atom_name2, res2, atom_name1)
            self.cosine_angles[key] = list(angs)
            self.cosine_angles[key_rev] = list(angs)

            if write_out_angles == True:

                # explicit for my own understanding ...
                angx = np.rad2deg(np.arccos(angs[0]))
                angy = np.rad2deg(np.arccos(angs[1]))
                angz = np.rad2deg(np.arccos(angs[2]))
                angles_out.write(f'{res1}\t{atom_name1}\t{angx}\t{angy}\t{angz}\n')


            if (atom_name1, atom_name2) in csa_ignore_list:
                csa_atom_check = False
            else:
                csa_atom_check = True
            
            if csa_atom_check and calc_csa_angles:
                
                # here we calculate all the CSA tensor angles 
                # but I think we only need the cosine angles for d11 - Need to check
                d_selected, (d11, d22, d33) = mathFunc.calc_csa_axis(res1, atom_name1, 
                    self.csa_tensor_transform[atom1_resname][atom_name1], 
                    self.average_uni)

                # in this case we provide all of them here: 
                self.csa_cosine_angles[key] = [mathFunc.cosine_angles(a, axis) for a in (d11, d22, d33)]

        if write_out_angles == True:
            angles_out.close()

    def calc_rotacf(self, indx_file, atom_names, b=None, e=None, dt=None, xtc=None, distance_cutoff=False,
                    timestep=2e-12, calc_csa_tcf=False, csa_tcf_skip=100, dense_acf_cutoff=100e-9, sparse_acf_freqiuency=1e-9,
                    max_csa_ct_diff=30e-9, write_csa_pas=False, csa_tcf_evaluation_steps=100, csa_ignore_list=[]):
        '''
        calculate the rotational correlation function using gromacs 

        the time step is to account for the fact that in the gromacs .xvg files for rotacf it gives the time step 
        not the actual time.
        '''

        #adjuat the cutoff parameters to have the same timestep as the gmx trr
        # dense_acf_cutoff = dense_acf_cutoff / self.trr_time_step
        # sparse_acf_freqiuency = sparse_acf_freqiuency / self.trr_time_step

        if xtc == None:
            xtc = self.xtc

        try:
            os.mkdir(f'{self.path_prefix}_rotacf')
        except FileExistsError:
            pass

        #atom_info = self.make_atom_pairs_list(atom_names)
        out_name = f'{self.path_prefix}_rotacf/rotacf.xvg'
        print('out name:', out_name)
        #print(f'Atom info length {len(self.gmx_atom_info)}')
        command = f'{self.gmx} rotacf -s {self.tpr} -f {xtc} -o {out_name}.xvg -n {indx_file} -P 2 -d -noaver -xvg none'
        #print('>>>>' + command)
        if b != None:
            command = command + f' -b {b} '
        if e != None:
            command = command + f' -e {e} '
        if dt != None:
            command = command + f' -dt {dt}'
        # else:
        #     #so we can use it in the csa rotacf function if needed
        #     dt = 1e-12


        print(f'Running the command: \n {command}')
        os.system(command)

        print('reading %s'%(out_name))
        time = []
        data = []

        # here we read in the files line by line
        # here we write out the xvg files as we need them
        print('time step', timestep)
        atom_info_counter = 0
        with open(out_name+'.xvg') as txt_file:
            print('writing out the xvg files with ACFs')
            for i in tqdm(txt_file):
                s = i.split()

                if s[0] == '&':

                    acf_skip = int(sparse_acf_freqiuency/timestep)
                    acf_skip_counter = 0

                    #here we write out the data 
                    time = np.array(time)*timestep
                    res1, res2, atom_name1, atom_name2 = self.gmx_atom_info[atom_info_counter]
                    xvg_out = f'{self.path_prefix}_rotacf/rotacf'+f'_{res1}_{atom_name1}_{res2}_{atom_name2}.xvg'
                    
                    # could probably remove the for loop if I really wanted 
                    f = open(xvg_out, 'w')

                    for ti, coeffi in zip(time, data):
                        check = False

                        if ti < dense_acf_cutoff:
                            check = True

                        if acf_skip_counter == acf_skip:
                            check = True
                            acf_skip_counter = 0
                        else: 
                            acf_skip_counter = acf_skip_counter + 1

                        if check == True:
                            f.write(f'{ti} {coeffi}\n')                    
                    f.close()
                    data = []
                    time = []
                    atom_info_counter = atom_info_counter + 1

                else:
                    # in this block we colect the data
                    data.append(float(s[1]))
                    time.append(float(s[0]))

        os.remove(out_name+'.xvg')

        # after coding this I feel like it should be its own function ...
        if calc_csa_tcf == True:
            self.csa_cosine_angles_trr = {}

            selections = {}
            time_list = {}

            for indx , (res1, res2, atom_name1, atom_name2)  in enumerate(self.gmx_atom_info):
                if (atom_name1, atom_name2) not in csa_ignore_list:
                    selections[(res1, atom_name1)] = []
                    time_list[(res1, atom_name1)] = []

            # currently this takes too long. I could make this shorter by using only a fraction of the points
            # Maybe say every 50-100 points. This looks like it should decay slow enough to catch the main motions 
            # inspected using gnuplot FTW 

            # so here we calculate all the different axis directions 
            # looking at the paper it seems we only need: D11 and D22 to calculate the spectral densities Jxx, Jyy and Jxy
            # Chemical Shift Anisotropy Tensors of Carbonyl, Nitrogen, and Amide 
            # Proton Nuclei in Proteins through Cross-Correlated Relaxation in NMR Spectroscopy

            if b != None:
                csa_start = int(b)
                print(f'b {b}')
            else:
                csa_start = 0

            if e != None:
                csa_end = int(e)
                print(f'e {e}')
            else:
                csa_end = -1

            print('Collecting CSA Principle axis possitions:')
            for ts in tqdm(self.uni.trajectory[csa_start:csa_end:csa_tcf_skip]):
                time = self.uni.trajectory.time

                for indx , (res1, res2, atom_name1, atom_name2)  in enumerate(self.gmx_atom_info):

                    if (atom_name1, atom_name2) not in csa_ignore_list:
                        #print((atom_name1, atom_name2))
                        atom1_resname = self.resid2type[res1][1]

                        d_selected, (d11, d22, d33) = mathFunc.calc_csa_axis(res1, atom_name1, 
                                                            self.csa_tensor_transform[atom1_resname][atom_name1], 
                                                            self.uni)

                        time_list[(res1, atom_name1)].append(time)
                        selections[(res1, atom_name1)].append([d11, d22, d33])

            # def wite out the vectors
            if write_csa_pas == True: 
                for i in selections:
                    out_name = f'{self.path_prefix}_rotacf/csa_pas_axis_{i[0]}_{i[1]}.xvg'
                    f = open(out_name, 'w')
                    f.write('#time d11x d11y d11z d22x d22y d22z d33x d33y d33z\n')

                    for line in selections[i]:
                        f.write(f'{line[0]} ')
                        f.write(f'{line[0][0]} {line[0][1]} {line[0][2]} ')
                        f.write(f'{line[1][0]} {line[1][1]} {line[1][2]} ')
                        f.write(f'{line[2][0]} {line[2][1]} {line[2][2]}\n')

                    f.close()

            # calculate correlation functions for csa Axis xx and yy
            # this can almost certainly be done in a faster way

            # csa_ct_xx, csa_ct_yy, csa_ct_xy = csa_spec_dens.calc_csa_spectral_density(selections, time_list, self.path_prefix, max_csa_ct_diff)
            csa_ct_xx = {}
            csa_ct_yy = {}
            csa_ct_xy = {}


            print('Calculating C(t) for CSA principle axis:')
            print(f'CSA c(t) cutoff is {max_csa_ct_diff}')
            print(f'This is about {int(max_csa_ct_diff/(csa_tcf_skip*timestep))} steps')
            #print(selections)
            for i in tqdm(selections):

                current = np.array(selections[i])
                time_array = np.array(time_list[i])*timestep
                d11 = np.array(current[:,1].astype(float))
                d22 = np.array(current[:,2].astype(float))

                # this assumes that the smallest time point is time_array[0]
                # sanity check: are times in assending order, seems to be true!
                #print('Are times in assending order:', np.all(time_array[:-1] <= time_array[1:]))
                

                time_diffs = time_array - time_array[0]
                all_arrays, time_tot, cxx_tot, cyy_tot, cxy_tot  = [],[],[],[],[]
                
                time_array_len = len(time_array)

                for time_i, d11_i, d22_i in zip(time_array, d11, d22):

                    # make the t0 array
                    time_dim = np.abs(time_array-time_i)
                    mask = (time_dim < max_csa_ct_diff)
                    mask_len = sum(mask)

                    # this section allows us to evaluate the TCF less often
                    # hopefully speeding up the calculation!

                    # if csa_tcf_evaluation_steps != False:
                    #     if csa_tcf_evaluation_steps < mask_len:
                    #         false_count = mask_len - csa_tcf_evaluation_steps

                    #         # make the masks
                    #         sparse_csa_evaluation_true = [True] * csa_tcf_evaluation_steps
                    #         sparse_csa_evaluation_false = [False]*false_count
                    #         sparse_csa_evaluation = sparse_csa_evaluation_true + sparse_csa_evaluation_false

                    #         r.shuffle(sparse_csa_evaluation)
                    #         mask = mask.tolist()# and sparse_csa_evaluation

                    d11_constant = np.zeros((mask_len, 3)) + d11_i
                    d22_constant = np.zeros((mask_len, 3)) + d22_i

                    # this einsum should give a 1D np array where the 
                    # each entry is the dot product - I hope this works!

                    #print('LOK HERE ')
                    #print(d11[mask])
                    #print(d11[mask].shape, d11_constant.shape)
                    
                    cxx = np.einsum('ij,ij->i',d11_constant,d11[mask])
                    cyy = np.einsum('ij,ij->i',d22_constant,d22[mask])
                    cxy = np.einsum('ij,ij->i',d11_constant,d22[mask])

                    #cxx = np.array([np.dot(m,k) for m,k in zip(d11_constant,d11)])
                    #cyy = np.array([np.dot(m,k) for m,k in zip(d22_constant,d22)])
                    #cxy = np.array([np.dot(m,k) for m,k in zip(d11_constant,d22)])

                    # now apply the P2 part 
                    cxx = 1.5*cxx**2 - 0.5
                    cyy = 1.5*cyy**2 - 0.5
                    cxy = 1.5*cxy**2 - 0.5

                    time_tot.append(time_dim[mask])
                    cxx_tot.append(cxx)
                    cyy_tot.append(cyy)
                    cxy_tot.append(cxy)

                time_tot = np.concatenate(time_tot, axis=0)
                cxx_tot = np.concatenate(cxx_tot, axis=0)
                cyy_tot = np.concatenate(cyy_tot, axis=0)
                cxy_tot = np.concatenate(cxy_tot, axis=0)

                # so noe we want to sum according to the 
                # here we have a rounding step due to machine precision in the times 
                # this will remove the noise in the c(t) when we plot it later
                time_tot_round = np.around(time_tot, decimals=self.decimal_round)
                dts, idx, count = np.unique(time_tot_round, return_counts=True, return_inverse=True)
                cur_ct_xx = np.bincount(idx, cxx_tot)/count
                cur_ct_yy = np.bincount(idx, cyy_tot)/count
                cur_ct_xy = np.bincount(idx, cxy_tot)/count

                # this assumes that the smallest time point is time_array[0]
                # sanity check: are times in assending order, seems to be true!
                # print('Are times in assending order:', np.all(dts[:-1] <= dts[1:]))
                
                # save the correlation functions.
                csa_ct_xx[i] = [dts, cur_ct_xx]
                csa_ct_yy[i] = [dts, cur_ct_yy]
                csa_ct_xy[i] = [dts, cur_ct_xy]

            #write out! 
            all_strings = ''
            for i in csa_ct_xx:

                out_name = f'{self.path_prefix}_rotacf/csa_ct_{i[0]}_{i[1]}.xvg'
                f = open(out_name, 'w')

                f.write('#time xx yy xy\n')
                string = ''.join([ f'{q} {w} {e} {r}\n' for q,w,e,r in zip(csa_ct_xx[i][0], csa_ct_xx[i][1], csa_ct_yy[i][1], csa_ct_xy[i][1])])
                #print(string)
                f.write(string)
                f.close()

    def calc_rotacf_segments(self,indx_file, atom_selection_pairs, seg_blocks, xtc=None):
        '''
        calculate the rotational correlation function for all the blocks
        '''

        try:
            os.mkdir(f'{self.path_prefix}_rotacf')
        except FileExistsError:
            pass

        for indx, i in enumerate(seg_blocks):
            out_name_curr = self.path_prefix+'_rotacf/rotacf' + f'_block_{indx}'
            self.calc_rotacf(indx_file, out_name_curr, atom_selection_pairs, b=i[0], e=i[1], xtc=xtc)

    
    def extract_diffusion_rotacf(self, internal_dir, total_dir, plot_skip=int(1e3)):
        '''
        Here we divide the total correlation function by the internal correlation function 
        This should yeild a correlation function that describes only the diffusion and not 
        the internal motions.
        '''

        internal_motion_files = glob.glob(internal_dir + '/rotacf*xvg')
        total_motion_files = glob.glob(total_dir + '/rotacf*xvg')
        
        try:
            os.mkdir(f'{self.path_prefix}_diffusion_rotacf/')
        except FileExistsError:
            pass

        try:
            os.mkdir(f'{self.path_prefix}_diffusion_rotacf_pdfs/')
        except FileExistsError:
            pass

        for internal_f in tqdm(internal_motion_files):

            file = internal_f.split('/')[-1]
            total_f = total_dir + '/' + file

            #print(internal_f, total_f)

            if total_f in total_motion_files:

                internal_x, internal_y = self.read_gmx_xvg(internal_f).T
                total_x, total_y = self.read_gmx_xvg(total_f).T
                file_name = f'{self.path_prefix}_diffusion_rotacf/{file}'
                diffusion_y = total_y/internal_y

                f = open(file_name, 'w')
                for i,j in zip(internal_x, diffusion_y):
                    f.write(f'{i} {j}\n')
                f.close()

                # adjust these for plotting!
                internal_x = internal_x * 1e6
                total_x = total_x * 1e6

                #this way of writing figures is more memory efficient
                fig = plt.figure(num=1, clear=True)
                ax = fig.add_subplot()
                ax.plot(internal_x[::plot_skip], internal_y[::plot_skip], label='internal')
                ax.plot(total_x[::plot_skip], total_y[::plot_skip], label='total')
                ax.plot(total_x[::plot_skip], diffusion_y[::plot_skip], label='diffusion')
                ax.set_xlabel('time (ns)')
                ax.set_ylabel('correlation')
                ax.legend()
                pdf_name = f'{self.path_prefix}_diffusion_rotacf/' + file.split('.')[0]+'.pdf'
                ax.set_xscale('symlog')
                fig.savefig(pdf_name)
                fig.clf()
                plt.close(fig)



    def correlation_function(self, Params, x, theta=0):

        '''
        An internal correlation function. This is the same one that is used
        by Kresten in the absurder paper.
        '''

        total = 0
        for i in range(self.curve_count):
            i = i+1
            amp = 'amp_%i'%(i)
            time = 'time_%i'%(i)
            total = total + Params[amp]*(np.e**(-1*x/Params[time]))

        #this is for the vectors that are not aligned with eachother
        if theta != 0:
            total = (1.5*np.cos(theta)**2 - 0.5)*total

        total = total + Params['S_long']
        return total

    def spectral_density_fuction(self,params, omega, dummy_tauc=5e-6, theta=0):

        term2 = params['S_long']*dummy_tauc/(1+(omega*dummy_tauc)**2)
        total = 0

        for i in range(self.curve_count):
            i = i + 1

            #correction for fitting the correlation times in ps
            taui = params['time_%i'%(i)]
            ampi = params['amp_%i'%(i)]


            tau_eff = taui*dummy_tauc/(dummy_tauc+taui)
            term1 = ampi*tau_eff/(1+(omega*tau_eff)**2)
            total = total + term1

        #this is incase the vectors are not aligned
        if theta != 0:
            total = (1.5*np.cos(theta)**2 - 0.5)*total

        total = total + term2 

        return (2/5.)*total

    def spectral_density_fuction_numerical(self, x,dummy_tauc, y):
            #x = x
            dt = x[1]-x[0]
            tumbling = 0.2*(np.e**(-x/dummy_tauc))
            total = tumbling * y

            j_fft = scipy.fft.fft(total)*dt*2
            j_fft = scipy.fft.fftshift(j_fft)
            j_fft = j_fft.real #* 2/5 # seems like this factor is needed for normalising, not sure why ...
            j_fft_freq = scipy.fft.fftfreq(len(x), d=dt)
            j_fft_freq = scipy.fft.fftshift(j_fft_freq)
            j_fft_freq = np.pi*j_fft_freq*2
            return j_fft_freq, j_fft
    
    def correlation_function_anisotropic(self, x, diffusion_tensor_components, angles):
        '''
        Anisotropic correlation function.
        '''

        # unpack varriables
        dx,dy,dz = diffusion_tensor_components
        ex,ey,ez = angles

        taus, amps = mathFunc.calculate_anisotropic_d_amps(dx,dy,dz,ex,ey,ez)
        x = np.array(x)
        total = np.zeros(len(x))

        for ti, ampi in zip(taus, amps):

            di = 1/ti
            total = total + ampi * np.e**(-1*di * x)

        return total

    def spectral_density_anisotropic(self, params, args):
        '''
        this function is a bit of bloat at the moment. I could remove all the listings from here but I can do this 
        another time. 
        '''
        return specDens.spectral_density_anisotropic_exponencial_sum(params, args)

    def plot_correlation_function(self, params, x, y, block, res1, atom_name1, theta=0):
        '''
        This function plots the correlation functions
        '''

        x_model = np.linspace(0, max(x), 10000)
        y_model = self.correlation_function(params, x_model)

        # calculate numerical spectral density 
        j_fft_freq, j_fft = self.spectral_density_fuction_numerical(x,self.dummy_tauc, y)
        freq_max = np.max(abs(j_fft_freq))
        model_omega = j_fft_freq
        xx_j_model = self.spectral_density_fuction(params, model_omega, dummy_tauc=self.dummy_tauc, theta=theta)

        #print('plotting...')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3), ) 
        ax1.plot(x,y, label='raw acf')
        ax1.plot(x_model, y_model, label='fit')
        ax1.legend()

        ax2.scatter(j_fft_freq,j_fft, label='raw acf', c='C1')
        ax2.plot(model_omega, xx_j_model, label='fit', c='C2')
        ax2.legend()    

        plt.savefig(f'{self.path_prefix}_fits/rot_acf_{block}_{res1}_{atom_name1}.pdf')
        # plt.show()
        # os.exit()
        plt.clf()
        plt.close()

    def plot_csa_tcfs(self, values, time, cts, block, res1, atom_name1):

        values_xx, values_yy, values_xy = values
        ct_xx, ct_yy, ct_xy = cts

        # make the models
        x_model = np.linspace(0, max(time), 10000)
        xx_model = self.correlation_function(values_xx, x_model)
        yy_model = self.correlation_function(values_yy, x_model)
        xy_model = self.correlation_function(values_xy, x_model, theta=np.pi/2)

        #make the model spectral density, numerical points,
        xx_j_fft_freq, xx_j_fft = self.spectral_density_fuction_numerical(time,self.dummy_tauc, ct_xx)
        yy_j_fft_freq, yy_j_fft = self.spectral_density_fuction_numerical(time,self.dummy_tauc, ct_yy)
        xy_j_fft_freq, xy_j_fft = self.spectral_density_fuction_numerical(time,self.dummy_tauc, ct_xy)

        # model for the spectral density
        omega_model = np.linspace(min(xx_j_fft_freq), max(xx_j_fft_freq), 1000)
        xx_j_model = self.spectral_density_fuction(values_xx, omega_model, dummy_tauc=5e-6, theta=0)
        yy_j_model = self.spectral_density_fuction(values_yy, omega_model, dummy_tauc=5e-6, theta=0)
        xy_j_model = self.spectral_density_fuction(values_xy, omega_model, dummy_tauc=5e-6, theta=0)

        #plot
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(9, 3), ) 

        # correlation functions
        ax1.scatter(time, ct_xx, label='ct_xx', color='C1',s=2)
        ax1.plot(x_model, xx_model, color='C2')

        ax2.scatter(time, ct_yy, label='ct_yy', color='C1',s=2)
        ax2.plot(x_model, yy_model, color='C2')

        ax3.scatter(time, ct_xy, label='ct_xy', color='C1',s=2)
        ax3.plot(x_model, xy_model, color='C2')

        #spectral densities
        ax4.scatter(xx_j_fft_freq, xx_j_fft, label='j_xx', color='C1',s=2)
        ax4.plot(omega_model, xx_j_model, color='C2')
        ax4.set_yscale('symlog')

        ax5.scatter(yy_j_fft_freq, yy_j_fft, label='j_yy', color='C1',s=2)
        ax5.plot(omega_model, yy_j_model, color='C2')
        ax5.set_yscale('symlog')

        ax6.scatter(xy_j_fft_freq, xy_j_fft, label='j_xy', color='C1',s=2)
        ax6.plot(omega_model, xy_j_model, color='C2')
        ax6.set_yscale('symlog')
        
        ax1.legend()
        ax2.legend()
        ax3.legend()
        
        plt.savefig(f'{self.path_prefix}_fits/rot_acf_{block}_{res1}_{atom_name1}_csa_ct.pdf')
        plt.clf()
        plt.close()


    def fit_correlation_function(self,x,y, log_time_min=5e-13, log_time_max=10e-9, blocks=True, theta=0):
        '''
        This function fits an internal correlation function and also the corresponding spectral density
        function. To do this we assume that the there is an overall isotropic tumbling so we can use a
        a spectral desnity function we know. In effect this just acts as a numerical trick to help
        speed up the fitting.

        Note that the units of time here are those used in the gromacs rotacf. This is generally ps
        '''

        # here I have manually defined some variables
        # because It didnt seem to want to play well with the Lmfit classes

        curve_count = copy.deepcopy(self.curve_count)
        dummy_tauc = copy.deepcopy(self.dummy_tauc)
        correlation_function = self.correlation_function

        # def calc_jw(time, internal_corelations,):

        #     '''
        #     This function is used to calculate an numerical spectral density function
        #     here we assume an overall isotropic tumbling. In reality this is just a way to focus the fitting
        #     on a region of interest.

        #     Here the spectral density is calculated numerically
        #     '''

        #     dt = time[1]-time[0]
        #     tumbling = 0.4*(np.e**(-time/dummy_tauc))
        #     total = tumbling * internal_corelations

        #     j_fft = scipy.fft.fft(total)*dt*2
        #     j_fft = scipy.fft.fftshift(j_fft)
        #     j_fft_freq = scipy.fft.fftfreq(len(time), d=dt)
        #     j_fft_freq = scipy.fft.fftshift(j_fft_freq)
        #     j_fft_freq = np.pi*j_fft_freq*2
        #     return j_fft_freq, j_fft

        def func2min(Params, x,y, theta):

            '''
            The function we want to minimize. The fitting is carried out both in s and hz.
            '''

            correlation_diff =  y - correlation_function(Params, x, theta=theta)
            #j_fft_freq, j_fft = self.spectral_density_fuction_numerical(x,dummy_tauc, y)
            #spec_difference = j_fft - self.spectral_density_fuction(Params, j_fft_freq,  dummy_tauc = dummy_tauc, theta=theta)

            #all_diffs = np.concatenate([spec_difference, correlation_diff])
            return correlation_diff #all_diffs.flatten()


        # create a set of Parameters
        Params = Parameters()
        s_guess = np.mean(y[int(len(y)*0.75):])

        # allow for a negative S is theta is not 0
        # woohoo this worked! 
        if theta == 0:
            Params.add('S_long', value=s_guess, min=0, max=1)
        else:
            Params.add('S_long', value=s_guess, min=-1, max=1)

        log_times = np.geomspace(log_time_min, log_time_max, curve_count)
        amp_list = []

        #here we generate the parameters
        for i, taui in zip(range(self.curve_count-1), log_times):
            i = i+1

            taui = taui

            # create a set of Parameters
            Params.add('amp_%i'%(i), value=1/curve_count, min=0, max=1.)
            amp_list.append('amp_%i'%(i))
            Params.add('time_%i'%(i), value=taui, min=0)
            i = 1+1

        #these final parameters are used to ensure the sums are all correct
        expression = '1 - (' + ' + '.join(amp_list) + ' + S_long)'
        Params.add('amp_%i'%(curve_count), expr=expression, min=0, max=1.)
        Params.add('time_%i'%(curve_count), value=log_times[-1], min=0)

        # for i in Params:
        #     print(i, Params[i].value)
        # do fit, here with the default leastsq algorithm
        minner = Minimizer(func2min, Params, fcn_args=(x, y, theta))
        result = minner.minimize()
        report_fit(result)
        return result

    def make_correlation_fit_directories(self,):
        
        '''
        This function makes the directories used by functions that fit 
        the correlation functions
        '''
        
        try:
            os.mkdir(f'{self.path_prefix}_fits')
        except FileExistsError:
            pass

        try:
            os.mkdir(f'{self.path_prefix}_fits')
        except FileExistsError:
            pass

        try:
            os.mkdir(f'{self.path_prefix}_fit_params')
        except FileExistsError:
            pass

    def write_line_to_params_files(self, values, file):
        '''
        This function is used by fit_all_correlation_functions_dipolar and 
        fit_all_correlation_functions_csa
        to write out the fitted parameters
        '''

        amps = [str(values['amp_%i'%(i+1)]) for i in range(self.curve_count)]
        amps = ','.join(amps)
        times = [str(values['time_%i'%(i+1)]) for i in range(self.curve_count)]
        times = ','.join(times)
        slong = values['S_long']
        line = f'{res1}:{atom_name1},{atom_name2}:{slong}:{amps}:{times}\n'
        file.write(line)
        file.flush()


    def fit_all_correlation_functions_dipolar(self,
        atom_names, time_threshold=10e-9,
        log_time_min=5e-13, log_time_max=10e-9,blocks=True, calc_csa_tcf=False, 
        ignore_list=[], csa_ignore_list=[]):

        acf_fitting_kws = {
        'log_time_min' : log_time_min, 
        'log_time_max' : log_time_max
        }

        atom_info = self.make_atom_pairs_list(atom_names)
        atom1 = atom_names[0][0]
        atom2 = atom_names[0][1]

        if blocks==True:
            block_count = self.get_number_of_blocks(atom_names)
        else:
            block_count = 1

        for block in range(block_count):

            params_out = open(f'{self.path_prefix}_fit_params/internal_correlations_{block}.dat', 'w')
            if calc_csa_tcf == True:
                params_out_csa_xx = open(f'{self.path_prefix}_fit_params/internal_csa_xx_correlations_{block}.dat', 'w')
                params_out_csa_yy = open(f'{self.path_prefix}_fit_params/internal_csa_yy_correlations_{block}.dat', 'w')
                params_out_csa_xy = open(f'{self.path_prefix}_fit_params/internal_csa_xy_correlations_{block}.dat', 'w')

            params_out.write('# residue:atoms:S2:amps:times\n')
            for res1, res2, atom_name1, atom_name2 in atom_info:

                #file IO
                if [atom_name1, atom_name2] not in ignore_list:

                    if blocks == True:
                        name = f"{self.path_prefix}_rotacf/rotacf_block_{block}_{res1}_{atom_name1}_{res2}_{atom_name2}.xvg"
                    else:
                        name = f"{self.path_prefix}_rotacf/rotacf_{res1}_{atom_name1}_{res2}_{atom_name2}.xvg"

                    # should we plot the figure for this ?
                    plot_chance = r.random()
                    plot_status = plot_chance < self.plotfraction
                    print('>fitting rotacf for:', name)

                    #Sometimes we dont want to fit the whole Ci(t) as it will often plateau. This time is in ps
                    temp = self.read_gmx_xvg(name)
                    x = temp.T[0]
                    y = temp.T[1]

                    #print(min(x), max(x), time_threshold)
                    y = y[x<time_threshold]
                    x = x[x<time_threshold]

                    #print('starting fit ...')
                    result = self.fit_correlation_function(x,y, **acf_fitting_kws)
                    values = result.params.valuesdict()

                    #put the plotting in!
                    if plot_status == True:
                        self.plot_correlation_function(values, x, y, block, res1, atom_name1)

                    #residues_pbar.update()
                    write_line_to_params_files(values, params_out)

        params_out.close()

    def fit_all_correlation_functions_csa(self,
        atom_names, time_threshold=10e-9,
        log_time_min=5e-13, log_time_max=10e-9,blocks=True, calc_csa_tcf=False, 
        ignore_list=[], csa_ignore_list=[]):

        acf_fitting_kws = {
        'log_time_min' : log_time_min, 
        'log_time_max' : log_time_max
        }

        atom_info = self.make_atom_pairs_list(atom_names)
        atom1 = atom_names[0][0]
        atom2 = atom_names[0][1]

        if blocks==True:
            block_count = self.get_number_of_blocks(atom_names)
        else:
            block_count = 1

        for block in range(block_count):

            params_out = open(f'{self.path_prefix}_fit_params/internal_correlations_{block}.dat', 'w')
            if calc_csa_tcf == True:
                params_out_csa_xx = open(f'{self.path_prefix}_fit_params/internal_csa_xx_correlations_{block}.dat', 'w')
                params_out_csa_yy = open(f'{self.path_prefix}_fit_params/internal_csa_yy_correlations_{block}.dat', 'w')
                params_out_csa_xy = open(f'{self.path_prefix}_fit_params/internal_csa_xy_correlations_{block}.dat', 'w')

            params_out.write('# residue:atoms:S2:amps:times\n')
            for res1, res2, atom_name1, atom_name2 in atom_info:

                #file IO
                if [atom_name1, atom_name2] not in ignore_list:

                    if calc_csa_tcf == True:
                        if (atom_name1, atom_name2) not in csa_ignore_list:
                            csa_name = f'{self.path_prefix}_rotacf/csa_ct_{res1}_{atom_name1}.xvg'
                            #print(csa_name)
                            temp = self.read_gmx_xvg(csa_name)
                            time = temp.T[0]
                            ct_xx = temp.T[1][time < time_threshold]
                            ct_yy = temp.T[2][time < time_threshold]
                            ct_xy = temp.T[3][time < time_threshold]
                            time = time[time < time_threshold]

                            # fit the CSA c(t)s
                            result_xx = self.fit_correlation_function(time,ct_xx, **acf_fitting_kws)
                            result_yy = self.fit_correlation_function(time,ct_yy, **acf_fitting_kws)
                            result_xy = self.fit_correlation_function(time,ct_xy, theta=np.pi/2, **acf_fitting_kws)
                            
                            # make vaues dicts
                            values_xx = result_xx.params.valuesdict()
                            values_yy = result_yy.params.valuesdict()
                            values_xy = result_xy.params.valuesdict()

                            #write the lines to the respective files!
                            write_line_to_params_files(values_xx, params_out_csa_xx)
                            write_line_to_params_files(values_yy, params_out_csa_yy)
                            write_line_to_params_files(values_xy, params_out_csa_xy)

                            if plot_status:
                                cts = [ct_xx, ct_yy, ct_xy]
                                values = [values_xx, values_yy, values_xy]
                                self.plot_csa_tcfs(values, time, cts, block, res1, atom_name1)

            #close the parameter files!
            if calc_csa_tcf == True:
                params_out_csa_xx.close()
                params_out_csa_yy.close()
                params_out_csa_xy.close()


    def fit_all_correlation_functions(self,
        atom_names, time_threshold=10e-9,
        log_time_min=5e-13, log_time_max=10e-9,blocks=True, calc_csa_tcf=False, 
        ignore_list=[], csa_ignore_list=[]):

        kws = {'time_threshold': time_threshold,
        'log_time_min' : log_time_min, 
        'log_time_max': log_time_max,
        'blocks':blocks, 
        'calc_csa_tcf':calc_csa_tcf, 
        'ignore_list':ignore_list, 
        'csa_ignore_list':csa_ignore_list}


        # fit all the dipolar parts 
        self.fit_all_correlation_functions_dipolar(atom_names, **kws)

        # fit all the dipolar csa parts
        fit_all_correlation_functions_csa(atom_names,**kws)

    def fit_diffusion_tensor(self, atom_names, blocks=True, threshold=50e-9, timestep=2e-12):
        '''
        Here we fit the diffusion tensor to the rotational correlation functions with 
        the internal motions removed
        '''
        try:
            os.mkdir(f'{self.path_prefix}_diffusion_rotacf_fit')
        except FileExistsError:
            pass

        
        atom_info = self.make_atom_pairs_list(atom_names)
        correlation_functions = []
        times = []
        cosine_angles = []
        #axis = self.uni.select_atoms('all').principal_axes()

        print('loading data for fitting diffusion tensor')
        for res1, res2, atom_name1, atom_name2 in tqdm(atom_info):
            if blocks == True:
                name = f"{self.path_prefix}_diffusion_rotacf/rotacf_block_{block}_{res1}_{atom_name1}_{res2}_{atom_name2}.xvg"
            else:
                name = f"{self.path_prefix}_diffusion_rotacf/rotacf_{res1}_{atom_name1}_{res2}_{atom_name2}.xvg"

            temp = self.read_gmx_xvg(name)
            x = temp.T[0]
            y = temp.T[1]

            y = y[x<threshold]
            x = x[x<threshold]

            atom1_poss = self.uni.select_atoms(f'resid {res1} and name {atom_name1}')[0].position
            atom2_poss = self.uni.select_atoms(f'resid {res2} and name {atom_name2}')[0].position
            angs = self.cosine_angles[(res1, atom_name1, res2, atom_name2)]

            cosine_angles.append(angs)
            correlation_functions.append(y)
            times.append(x)

        #define the model 
        def model(params, time, cosine_angles):

            dx = params['dx']
            dy = params['dy']
            dz = params['dz']
            ds = (dx,dy,dz) 
            mod = self.correlation_function_anisotropic(time, ds, cosine_angles)
            return mod

        def residual(params, cosine_angles, correlation_functions, times):

            diffs = []

            # print(cosine_angles)

            for angles, corri, time in zip(cosine_angles, correlation_functions, times):
                current_model = model(params, time, angles)
                current_diffs = corri - current_model
                diffs.append(current_diffs)
            
            return np.concatenate(diffs)

        def cb_iteration(params, iter, resid, cosine_angles, correlation_functions, times):
            '''
            print out between iterations
            '''

            rmsd = np.mean(np.absolute(resid))
            print(f'it: {iter} {rmsd}')

        Params = Parameters()
        Params.add('dx', value=22289619, min=0, vary=True)
        Params.add('dy', value=21384570, min=0, vary=True)
        Params.add('dz', value=36879686, min=0, vary=True)

        minner = Minimizer(residual, Params, fcn_args=(cosine_angles, correlation_functions, times), iter_cb=cb_iteration)
        result = minner.minimize(method='powel')
        res_params = result.params
        report_fit(result)

        diso = (res_params['dx'].value + res_params['dy'].value + res_params['dz'].value)/3
        iso_tc = 1/(6*diso)
        print('isotropic tau c:' , iso_tc)

        for i, angles, corri   in zip(atom_info, cosine_angles, correlation_functions):
            res1, res2, atom_name1, atom_name2 = i 
            if blocks == True:
                name = f"{self.path_prefix}_diffusion_rotacf_fit/rotacf_block_{res1}_{atom_name1}_{res2}_{atom_name2}.pdf"
            else:
                name = f"{self.path_prefix}_diffusion_rotacf_fit/rotacf_{res1}_{atom_name1}_{res2}_{atom_name2}.pdf"

            current_model = model(res_params, x, angles)
            plt.plot(x,current_model, label='model')
            plt.plot(x,corri, label='rotacf')
            plt.legend()
            plt.savefig(name)
            plt.close()

        self.write_diffusion_trace(res_params, "diffusion_tensor.dat")
    
    
    def calculate_r1_r2_hetnoe(self, atom_names, diffusion_file, fields,x, y='h', blocks=False,dna=False, write_out=False, prefix='', ignore_atoms=[]):
        '''
        This function calculate r1 r2 and hetnoe at high field for a X-Y spin system
        '''

        try:
            os.mkdir(f'{self.path_prefix}_calculated_relaxation_rates')
        except FileExistsError:
            pass

        atom_info = self.make_atom_pairs_list(atom_names)

        data = {}
        params = {}
        diffusion_values, diffusion_errors = utils.read_diffusion_tensor(diffusion_file)
        print(f'Diffusion Tensor values: \ndx {diffusion_values[0]}\ndy {diffusion_values[1]}\ndz {diffusion_values[2]}')
        axis = self.uni.select_atoms('all').principal_axes()

        if blocks == False:
            spec_file = self.path_prefix + '_fit_params/internal_correlations_0.dat'
            # csa_xx_spec_file = self.path_prefix + '_fit_params/internal_csa_xx_correlations_0.dat'
            # csa_yy_spec_file = self.path_prefix + '_fit_params/internal_csa_yy_correlations_0.dat'
            # csa_xy_spec_file = self.path_prefix + '_fit_params/internal_csa_xy_correlations_0.dat'
        else:
            print('LUCAS YOU NEED TO CODE THIS!')

        spectral_density_params = utils.read_fitted_spectral_density(spec_file)
        csa_spectral_density_params = self.read_csa_spectra_density_params()
        csa_xx_spectral_density_params, csa_yy_spectral_density_params, csa_xy_spectral_density_params = csa_spectral_density_params

        if write_out == True:

            print(f'Writing to:')
            print(f'{self.path_prefix}_calculated_relaxation_rates/{prefix}r1.dat)')
            print(f'{self.path_prefix}_calculated_relaxation_rates/{prefix}r2.dat')
            print(f'{self.path_prefix}_calculated_relaxation_rates/{prefix}hetnoe.dat')

            print(f'{self.path_prefix}_calculated_relaxation_rates/{prefix}sigma_noe.dat')
            
            r1_out = open(f'{self.path_prefix}_calculated_relaxation_rates/{prefix}r1.dat', 'w')
            r1_out.write('#header bondName \n' )
            
            r2_out = open(f'{self.path_prefix}_calculated_relaxation_rates/{prefix}r2.dat', 'w')
            r2_out.write('#header bondName \n' )
            
            hetnoe_out = open(f'{self.path_prefix}_calculated_relaxation_rates/{prefix}hetnoe.dat', 'w')
            hetnoe_out.write('#header bondName \n' )

            sigma_noe_out = open(f'{self.path_prefix}_calculated_relaxation_rates/{prefix}sigma_noe.dat', 'w')
            sigma_noe_out.write('#header bondName \n' )


        # select only the atoms we want
        # probably should make a feature to merge this with the CSA ignore list
        
        reduced_atom_info = []
        # probably can do this with itertools 
        for res1, res2, atom_name1, atom_name2 in atom_info:

            # could abstract this check to a function because I do it a few times 
            if (atom_name1, atom_name2) not in ignore_atoms:
                reduced_atom_info.append([res1, res2, atom_name1, atom_name2])


        for res1, res2, atom_name1, atom_name2 in reduced_atom_info:

            #write out location
            if blocks == True:
                name = f"{self.path_prefix}_diffusion_rotacf/rotacf_block_{block}_{res1}_{atom_name1}_{res2}_{atom_name2}.xvg"
            else:
                name = f"{self.path_prefix}_diffusion_rotacf/rotacf_{res1}_{atom_name1}_{res2}_{atom_name2}.xvg"


            angs = self.cosine_angles[(res1, atom_name1, res2, atom_name2)]
            csa_angs = self.csa_cosine_angles[(res1, atom_name1, res2, atom_name2)]
            rxy = PhysQ.bondlengths[atom_name1, atom_name2]
            spectral_density_key = (res1, atom_name1+','+atom_name2)
            params = spectral_density_params[spectral_density_key]

            # make the csa params object
            csa_xx_params = csa_xx_spectral_density_params[spectral_density_key]
            csa_yy_params = csa_yy_spectral_density_params[spectral_density_key]
            csa_xy_params = csa_xy_spectral_density_params[spectral_density_key]
            csa_params = (csa_xx_params, csa_yy_params, csa_xy_params)

            #now we need to add the diffusion parameters
            for i in (params, csa_xx_params, csa_yy_params,csa_xy_params):
                i = self.add_diffution_spectral_density_params(i, diffusion_values)

            if dna == True:
                csa_atom_name = (atom_name1, self.resid2type[res1][1])
                resname = self.resid2type[res1][1]
            else:
                csa_atom_name = (atom_name1, self.resid2type[res1])
                resname = self.resid2type[res1]

            # print(res1)
            spectral_density = self.spectral_density_anisotropic
            r1 = d2e.rates.r1_YX(params, spectral_density, fields,rxy, csa_atom_name, x, y='h', 
                                cosine_angles = angs, csa_cosine_angles=csa_angs, csa_params=csa_params)

            r2 = d2e.rates.r2_YX(params, spectral_density, fields,rxy, csa_atom_name, x, y='h', 
                                cosine_angles = angs, csa_cosine_angles=csa_angs, csa_params=csa_params)
            
            hetnoe = d2e.rates.noe_YX(params, spectral_density, fields,
                               rxy, x, r1, y='h', cosine_angles=angs)

            sigma = d2e.rates.r1_reduced_noe_YX(params, spectral_density, fields, rxy, x, cosine_angles=angs)
            
            if write_out == True:
                key = resname+str(res2)+atom_name2+'-'+resname+str(res1)+atom_name1
                #print(res1, res2, atom_name1, atom_name2, write_out, key)
                r1_str = ' '.join([str(a) for a in r1])
                r2_str = ' '.join([str(a) for a in r2])
                hetnoe_str = ' '.join([str(a) for a in hetnoe])
                sigma_str = ' '.join([str(a) for a in sigma])

                r1_out.write(f"{key} {r1_str}\n")
                r2_out.write(f"{key} {r2_str}\n")
                hetnoe_out.write(f"{key} {hetnoe_str}\n")
                sigma_noe_out.write(f"{key} {sigma_str}\n")


        if write_out == True:
            r1_out.close()
            r2_out.close()
            hetnoe_out.close()
            sigma_noe_out.close()

    def calculate_relaxometry_intensities(self, atom_names, diffusion_file, 
        x, y='h', blocks=False,dna=False, write_out=False, prefix='', ignore_atoms=[]):
        '''
        This function calculate r1 r2 and hetnoe at high field for a X-Y spin system
        '''

        try:
            os.mkdir(f'{self.path_prefix}_calculated_relaxometry_intensities')
        except FileExistsError:
            pass

        atom_info = self.make_atom_pairs_list(atom_names)

        data = {}
        params = {}
        diffusion_values, diffusion_errors = utils.read_diffusion_tensor(diffusion_file)
        print(f'Diffusion Tensor values: \ndx {diffusion_values[0]}\ndy {diffusion_values[1]}\ndz {diffusion_values[2]}')
        axis = self.uni.select_atoms('all').principal_axes()

        if blocks == False:
            spec_file = self.path_prefix + '_fit_params/internal_correlations_0.dat'
        else:
            print('LUCAS YOU NEED TO CODE THIS!')

        spectral_density_params = utils.read_fitted_spectral_density(spec_file)

        csa_spectral_density_params = self.read_csa_spectra_density_params()
        csa_xx_spectral_density_params, csa_yy_spectral_density_params, csa_xy_spectral_density_params = csa_spectral_density_params

        #add diffusion values to the params
        for i in spectral_density_params:
            spectral_density_params[i] = self.add_diffution_spectral_density_params(spectral_density_params[i], diffusion_values)
        
        for i,j,k in zip(csa_xx_spectral_density_params, csa_yy_spectral_density_params, csa_xy_spectral_density_params):
            csa_xx_spectral_density_params[i] = self.add_diffution_spectral_density_params(csa_xx_spectral_density_params[i], diffusion_values)
            csa_yy_spectral_density_params[j] = self.add_diffution_spectral_density_params(csa_yy_spectral_density_params[j], diffusion_values)
            csa_xy_spectral_density_params[k] = self.add_diffution_spectral_density_params(csa_xy_spectral_density_params[k], diffusion_values)


        if write_out == True:
            print('make the writing out files!')
        
        reduced_atom_info = []

        # this could be prettier.
        for res1, res2, atom_name1, atom_name2 in atom_info:
            if (atom_name1, atom_name2) not in ignore_atoms:
                reduced_atom_info.append([res1, res2, atom_name1, atom_name2])

        # get times at each field
        
        if self.relaxometry_experiment_details_file != None:
            print(f"Loading: {self.relaxometry_experiment_details_file}")
            Shuttling = shuttling.ShuttleTrajectory(self.relaxometry_experiment_details_file)
            Shuttling.construct_all_trajectories()
        else:
            print('Please provide the file with information on the shuttled experiments')
            print('This is done by setting: AnalyseTrr.relaxometry_experiment_details_file')

        for res1, res2, atom_name1, atom_name2 in tqdm(reduced_atom_info):

            # define the parameters
            spectral_density_key = (res1, atom_name1+','+atom_name2)
            #print(spectral_density_key)

            # make the csa params object
            csa_xx_params = csa_xx_spectral_density_params[spectral_density_key]
            csa_yy_params = csa_yy_spectral_density_params[spectral_density_key]
            csa_xy_params = csa_xy_spectral_density_params[spectral_density_key]
            csa_params = (csa_xx_params, csa_yy_params, csa_xy_params)

            spectral_density = self.spectral_density_anisotropic

            atom_check = False
            if atom_name1 == "C1'":
                atom_check = True
                x_spin = "C1'"
                y_spin = "H1'"                
                active_protons = ["H1'", "H2'1","H2'2", "H3'"] 
                #active_protons = ["H1'", "H2'1", "H2'2","H3'", "H4'"]
                passive_protons = active_protons
                operator_size = 1+1 + len(active_protons)*2

            if atom_name1 == "C5":
                atom_check = True
                x_spin = "C5"
                y_spin = "H5"
                active_protons = ["H5", "H6"] 
                passive_protons = active_protons
                operator_size = 1+1 + len(active_protons)*2

            
            print(atom_name1)

            if atom_check == True:
                self.relaxometry_inensities[(res1, x_spin, y_spin)] = {}
                for i in Shuttling.sampled_fields:

                    #print(f'field  {i}')
                    traj = Shuttling.trajectory_time_at_fields[i]
                    forwards_field, forwards_time, backwards_field, backwards_time = traj 
                    
                    relaxation_matricies_forwards = relax_mat.relaxation_matrix(spectral_density_params, 
                    res1, 
                    spectral_density, 
                    forwards_field, 
                    self.md_distances,  
                    self.cosine_angles, 
                    self.csa_cosine_angles, 
                    csa_params,
                    self.resid2type[res1][1], 
                    operator_size,
                    x_spin,
                    y_spin,
                    active_protons,
                    passive_protons,
                    )

                    relaxation_matricies_backwards = relax_mat.relaxation_matrix(spectral_density_params, 
                    res1, 
                    spectral_density, 
                    backwards_field, 
                    self.md_distances,  
                    self.cosine_angles, 
                    self.csa_cosine_angles, 
                    csa_params,
                    self.resid2type[res1][1], 
                    operator_size,
                    x_spin,
                    y_spin,
                    active_protons,
                    passive_protons,)

                    relaxation_matricies_stablaization_delay = relax_mat.relaxation_matrix(spectral_density_params, 
                    res1, 
                    spectral_density, 
                    np.array([Shuttling.experiment_info[i]['high_field']]), 
                    self.md_distances,  
                    self.cosine_angles, 
                    self.csa_cosine_angles, 
                    csa_params, 
                    self.resid2type[res1][1],  
                    operator_size,
                    x_spin,
                    y_spin,
                    active_protons,
                    passive_protons )

                    forwrds_propergators = mathFunc.construct_operator(relaxation_matricies_forwards, forwards_time)
                    
                    backwards_operators = mathFunc.construct_operator(relaxation_matricies_backwards, backwards_time)

                    stablaization_operator = mathFunc.construct_operator(relaxation_matricies_stablaization_delay,
                        np.array([Shuttling.experiment_info[i]['stabalisation_delay']]), product=False)
                    
                    stablaization_operator = np.array(stablaization_operator)[0]


                    #this is the matrix that evolves with the varriable delay
                    low_field_relaxation_matrix = relax_mat.relaxation_matrix(spectral_density_params, 
                    res1, 
                    spectral_density, 
                    np.array([i]), 
                    self.md_distances,  
                    self.cosine_angles, 
                    self.csa_cosine_angles, 
                    csa_params, 
                    self.resid2type[res1][1],  
                    operator_size,
                    x_spin,
                    y_spin,
                    active_protons,
                    passive_protons)

                    # print("Field", i)
                    # print('Relaxation matrix:: ')
                    # print(sum(forwards_time))
                    # print(np.array_str(low_field_relaxation_matrix.T, precision=1, suppress_small=True))

                    delay_propergators = mathFunc.construct_operator(low_field_relaxation_matrix, 
                                            Shuttling.experiment_info[i]['delays'], product=False)
                    
                    #full_propergators =  delay_propergators # [np.linalg.multi_dot([forwrds_propergators, i, backwards_operators]) for i in delay_propergators]
                    full_propergators = [np.linalg.multi_dot([stablaization_operator, backwards_operators, i, forwrds_propergators]) for i in delay_propergators]

                    # print('Relaxation matrix', self.resid2type[res1][1], atom_name1)
                    # print(np.array_str(low_field_relaxation_matrix.T[0], precision=2, max_line_width=1000, suppress_small=True))

                    #w, v = np.linalg.eig(low_field_relaxation_matrix.T[0])
                    #print(np.array_str(w, precision=3, max_line_width=100, suppress_small=True))


                    # test
                    # full_propergators = mathFunc.construct_operator(low_field_relaxation_matrix, 
                    #                          Shuttling.experiment_info[i]['delays'], product=False)


                    #print('propergator')
                    #print(np.array_str(full_propergators[-1], precision=3, max_line_width=100, suppress_small=True))
                    #w, v = np.linalg.eig(full_propergators[-1])
                    #print(np.array_str(w, precision=3, max_line_width=100, suppress_small=True))


                    experimets = np.zeros(operator_size)
                    experimets[1] = 1.
                    # a = np.dot(full_propergators[-1], experimets)
                    # b = np.dot(a, full_propergators[-1].T)
                    # # print(np.array_str(b, precision=3, max_line_width=100, suppress_small=True))

                    intensities = [np.dot(i, experimets) for i in full_propergators]
                    self.relaxometry_inensities[(res1, atom_name1,atom_name2)][i] = [Shuttling.experiment_info[i]['delays'], intensities]

            
    def fit_apprent_rate_to_relaxometry(self, rate_id=1):

        def model(params, x):
            return params['a']*np.e**(-1*params['r']*x)

        def resid(params, x,y):
            return y - model(params, x)

        self.apparent_rates = {}

        for i in self.relaxometry_inensities:
            self.apparent_rates[i] = []
            res1, atom_name1,atom_name2 = i 
            for field in self.relaxometry_inensities[i]:

                #print(field)
                x,y = self.relaxometry_inensities[i][field]
                x = np.array(x)

                #kinda ugle .... 
                y = np.array(y)
                y = y.T[rate_id].T

                params = Parameters()
                params.add('r', value=1)
                params.add('a', value=0.11)
                minner = Minimizer(resid, params, fcn_args=(x,y))
                result = minner.minimize()
                res_params = result.params.valuesdict()
                #report_fit(result)
                self.apparent_rates[i].append([field, res_params['r']])

                plt.title(i)
                plt.scatter(x,y)
                x_model = np.linspace(min(x), max(x), 20)
                #print(x, x_model, min(x), max(x))
                r = res_params['r']
                plt.plot(x_model, model(res_params, x_model), label=f'field: {field:.1f}, Rate: {r:.1f}')

            
            plt.legend()
            plt.savefig(f'{self.path_prefix}_{res1}_{atom_name1}_intensities.pdf')
            plt.clf()

    def plot_relaxometry_coherences(self):

        for i in self.relaxometry_inensities:
            # print(i)
            for field in self.relaxometry_inensities[i]:

                if i[1] == "C1'":
                    labels = ["E/2","C1'z","H1'z","H2'1z","H2'2z","H3'z","H4'z","C1'zH1'z","C1'zH1'","C1'zH2'1z","C1'zH2'2z","C1'zH3'z" ,"C1'zH4'z"]
                elif i[1] == 'C5':
                    labels =   [ "E/2","C5z","H5z","H6z","H41z", "H42z","C5zH5z","C5zH6z","C5zH41z","C5zH42z"]
                x,y = self.relaxometry_inensities[i][field]
                x = np.array(x)
                # print(x)

                #kinda ugle .... 
                y = np.array(y)
                plt.title(f"{i[0]}, {i[1]}, {i[2]}, Field: {field}")

                for indx, yi in enumerate(y.T):
                    x_plot, yi = zip(*sorted(zip(x,yi)))
                    plt.plot(x_plot,yi, label=f"{indx} {labels[indx]}")
            
                plt.legend()
                plt.xlabel("time")
                plt.ylabel('population')
                plt.savefig(f'{i[0]}_{i[1]}_{i[2]}_Field_{field}.pdf')
                plt.close()

    def write_apparent_relaxometry_rates(self, file_name_prefix='', file_name='apparent_relaxometry_rates.dat'):

        '''
        Need to re-write this so the format matches the other rates files, mostly an issue with name tags 
        '''

        file = self.path_prefix + '_calculated_relaxation_rates/' + file_name_prefix + file_name
        print(f'Writing apparent R1 rates to: {file}')
        f = open(file, 'w')
        f.write('#header bondName ')
        header_check = False
        for i in self.apparent_rates:
            resname = self.resid2type[i[0]][1]
            name = resname + str(i[0]) + str(i[2]) + '-' + resname + str(i[0]) + str(i[1])

            data = np.array(self.apparent_rates[i])
            data = data[data[:, 1].argsort()].T

            if header_check == False:
                header_check = True
                f.write(' '.join([str(a) for a in data[0]]) + '\n')

            f.write(name +' ' + ' '.join([str(round(a,4)) for a in data[1]]) + '\n')
        f.close()


    def fit_diffusion_to_r1_r2_hetnoe(self, r1_file, r1_error, r2_file, r2_error, hetNoe_file, hetNoe_errors, spectral_density_file,
                                      fields,x, y='h', blocks=False,dna=False, write_out=False, reduced_noe=False,
                                      error_filter=0.05, PhysQ=PhysQ, model="anisotropic", scale_model='default', out_name= "diffusion_tensor_fitted.dat"):
    
        def resid(params, values, errors, csa, bondlength, cosine_angles, spec_params, fields, csa_cosine_angles, csa_params):

            spec_den = self.spectral_density_anisotropic
            total = []

            for currrent_vals in zip(values, errors, csa, bondlength, cosine_angles, spec_params, csa_cosine_angles,csa_params):
                vali, erri, csai, bondlengthi, angi, speci, csa_cosine_angles, csa_parami = currrent_vals

                # these were from the old main branch 

                # model_r1 = d2e.rates.r1_YX(speci, spec_den, fields, bondlengthi, csai, x, cosine_angles=csa_angi, csa_cosine_angles=csa_angi)
                # model_r2 = d2e.rates.r2_YX(speci, spec_den, fields, bondlengthi, csai, x, cosine_angles=csa_angi, csa_cosine_angles=csa_angi)
                csa_parami_xx, csa_parami_yy, csa_parami_xy = csa_parami
                #print('CSA Angle', csa_cosine_angles)

                #add diffution to the spectral density parameters
                for param_dict in [speci, csa_parami_xx, csa_parami_yy, csa_parami_xy]:
                    param_dict['dx'] = params['dx']
                    param_dict['dy'] = params['dy']
                    param_dict['dz'] = params['dz']

                csa_parami = (csa_parami_xx, csa_parami_yy, csa_parami_xy)

                model_r1 = d2e.rates.r1_YX(speci, spec_den, fields, bondlengthi, csai, x, 
                    cosine_angles=angi, csa_cosine_angles=csa_cosine_angles, csa_params=csa_parami)
                model_r2 = d2e.rates.r2_YX(speci, spec_den, fields, bondlengthi, csai, x, 
                    cosine_angles=angi, csa_cosine_angles=csa_cosine_angles,  csa_params=csa_parami)

                # use the reduced NOE in the fitting? This is to prevent the R1 being pressent twice in the fit 
                # and alows the error in the R1 to be included in the error estimation for the hetNOE 

                if reduced_noe == False:
                    model_noe = d2e.rates.noe_YX(speci, spec_den, fields, bondlengthi, x, model_r1, cosine_angles=angi)
                elif reduced_noe == True:
                    model_noe = d2e.rates.r1_reduced_noe_YX(speci, spec_den, fields, bondlengthi, x, cosine_angles=angi)

                model = np.array([model_r1,model_r2, model_noe])
                #print(model.shape, vali.shape,  erri)
                diffs = (model - vali)/erri
                total.append(diffs)

            total = np.array(total).flatten()
            return total

        # read in the data
        r1, _ = d2e.utils.read_nmr_relaxation_rate(r1_file)
        r2, _ = d2e.utils.read_nmr_relaxation_rate(r2_file) 
        hetnoe, _ = d2e.utils.read_nmr_relaxation_rate(hetNoe_file) 

        r1_err, _ = d2e.utils.read_nmr_relaxation_rate(r1_error)
        r2_err, _ = d2e.utils.read_nmr_relaxation_rate(r2_error)
        hetnoe_err, _ = d2e.utils.read_nmr_relaxation_rate(hetNoe_errors) 
        spectral_density_params_dict = utils.read_fitted_spectral_density(spectral_density_file)

        #read in the CSA parameters 
        csa_spectral_density_params = self.read_csa_spectra_density_params()
        csa_xx_spectral_density_params, csa_yy_spectral_density_params, csa_xy_spectral_density_params = csa_spectral_density_params

        axis = self.uni.select_atoms('all').principal_axes()

        # Quite a few logic checks!
        # should probably turn this into a function, also used in other routines 
        values = []
        errors = []
        csa = []
        bondlengths = []
        cosine_angles = []
        csa_cosine_angles = []
        spectral_density_params = []
        csa_params = []


        for i in r1:

            try:
                # this try statement if you want to look at fewer atoms 
                # need a better fix in future

                #print(i)
                check = False

                # get the atom info earlier in the funtion that I originally thought 
                atom1_letters, atom1_numbers, atom1_res_type, atom1_resid, atom1_type, \
                atom2_letters, atom2_numbers, atom2_res_type, atom2_resid, atom2_type = utils.get_atom_info_from_rate_key(i)
                nucleus_type = atom2_letters[1].lower()
                atom_full_names = (atom2_letters[1] + atom2_numbers[1], atom1_letters[1] + atom1_numbers[1])

                #key for the spectral density
                spec_key = (int(atom1_numbers[0]), atom2_letters[1] + atom2_numbers[1]+','+atom1_letters[1] + atom1_numbers[1])
                
                if i in r2: 
                    if i in hetnoe:
                        # do some sub selection 
                        if spec_key in spectral_density_params_dict:
                            check=True

                if check == True:
                    #do some error filtering
                    if error_filter != None:
                        if np.mean(r1_err[i]/r1[i]) > error_filter:
                            check = False

                        if np.mean(r2_err[i]/r2[i]) > error_filter:
                            check = False

                        if np.mean(hetnoe_err[i]/hetnoe[i]) > error_filter:
                            check = False

                if check == True:

                    #spectral density parameters
                    spectral_density_params.append(spectral_density_params_dict[spec_key])

                    #csa parameters
                    csa_params_temp = [a[spec_key] for a in (csa_xx_spectral_density_params, csa_yy_spectral_density_params, csa_xy_spectral_density_params)]
                    csa_params.append(csa_params_temp)

                    #get the local correlation times but fitting r1, r2, and hetnoe 
                    current_values = np.array([r1[i], r2[i],hetnoe[i]])
                    values.append(current_values)
                    current_errors = np.array([r1_err[i], r2_err[i],hetnoe_err[i]])
                    errors.append(current_errors)

                    csa_key = (atom2_type, atom2_res_type)
                    csa.append(csa_key)
                    bond_lengths_key = (atom2_type, atom1_type)
                    bond_length = PhysQ.bondlengths[bond_lengths_key]
                    bondlengths.append(bond_length)

                    #vec = strucUtils.get_bond_vector(self.uni,atom1_resid, atom1_type, atom2_resid, atom2_type)
                    
                    #calculate the cosine angles. I am not sure how these should be ordered with 
                    #print(self.cosine_angles.keys())
                    angs = self.cosine_angles[ (int(atom2_resid), atom2_type, int(atom1_resid), atom1_type)]
                    csa_angs = self.csa_cosine_angles[ (int(atom2_resid), atom2_type, int(atom1_resid), atom1_type)]
                    cosine_angles.append(angs)
                    csa_cosine_angles.append(csa_angs)

            except KeyError:
                pass


        if model == "anisotropic":
            params = Parameters()
            params.add('dx', min=0, value=1/(6*4e-9))
            params.add('dy', min=0, value=1/(6*4e-9))
            params.add('dz', min=0, value=1/(6*4e-9))

        elif model == 'sphereoid':
            params = Parameters()
            params.add('dx', min=0, value=1/(6*4e-9))
            params.add('dy', min=0, value=1/(6*4e-9), expr='dx')
            params.add('dz', min=0, value=1/(6*4e-9))

        elif model == 'scale':

            if scale_model == 'default':
                dval, diffusion_errors = utils.read_diffusion_tensor(f"{self.path_prefix}_diffusion_rotacf_fit/diffusion_tensor.dat")
            else:
                dval, diffusion_errors = utils.read_diffusion_tensor(scale_model)

            params = Parameters()
            params.add('scalar', min=0, value=1)
            params.add('dxinit', value=dval[0], vary=False)
            params.add('dyinit', value=dval[1], vary=False)
            params.add('dzinit', value=dval[2], vary=False)

            params.add('dx', min=0, expr='dxinit*scalar')
            params.add('dy', min=0, expr='dyinit*scalar')
            params.add('dz', min=0, expr='dzinit*scalar')
        else:
            print('Model selected does not exist')
            sys.exit()

        minner = Minimizer(resid, params, fcn_args=(values, errors, csa, bondlengths, cosine_angles, 
                           spectral_density_params, fields, csa_cosine_angles,csa_params))

        result = minner.minimize()
        res_params = result.params
        report_fit(result)

        diso = (res_params['dx'].value + res_params['dy'].value + res_params['dz'].value)/3
        iso_tc = 1/(6*diso)
        print(f'isotropic tau c: {iso_tc * 1e9:0.2} ns' )
        self.write_diffusion_trace(res_params, out_name)

    # def fit_emf_to_spectral_density(self, atom_names, spectral_density_file, diffusion_file, omega_range=[1e-5, 4e9], omega_number_of_steps=1e3):

    #     '''
    #     This function tries to fit extended model free to the MD spectral density function model but for some 
    #     reason it doesnt produce very stable results. 
    #     '''
    #     atom_info = self.make_atom_pairs_list(atom_names)
    #     spectral_density_params_dict = utils.read_fitted_spectral_density(spectral_density_file)
    #     dval, diffusion_errors = utils.read_diffusion_tensor(diffusion_file)
    #     omega = np.geomspace(omega_range[0], omega_range[1], int(omega_number_of_steps))

    #     def model(params, args):
    #         model_corr = specDens.J_anisotropic_emf(params, args)
    #         return model_corr

    #     def resid(params, resid, args, target):
    #         model_corr = model(params, args)
    #         return model_corr - target

    #     emf_models = {}
    #     for res1, res2, atom_name1, atom_name2 in tqdm(atom_info):

    #         params = copy.copy(spectral_density_params_dict[spec_dens_key])
    #         params['dx']=dval[0]
    #         params['dy']=dval[1]
    #         params['dz']=dval[2]

    #         angles_key = (res1, atom_name1, res2, atom_name2)
    #         ex,ey,ez = self.cosine_angles[angles_key]

    #         args = (time, ex,ey,ez )
    #         model_params = mf.generate_mf_parameters(scale_diffusion=False, diffusion=dval)
    #         bic = None 
    #         mod = model_params[-1]

    #         minner = Minimizer(resid, mod, fcn_args=(mod, args, current_spectral_density))
    #         result = minner.minimize(method="powel")

    #         if bic == None:
    #             bic = result.bic
    #             emf_models[angles_key] = result

    #         elif result.bic < bic:
    #             bic = result.bic
    #             emf_models[angles_key] = result

    #         result = minner.minimize(method='powel')
    #         resdict = result.params.valuesdict()

    #         # if "C1'" in (atom_name1, atom_name2) and "H1'" in (atom_name1, atom_name2):
    #         #     cur_mod = model(emf_models[angles_key].params, args)
    #         #     plt.plot(omega, current_spectral_density, label='tagr')
    #         #     plt.plot(omega, cur_mod, label='mod')
    #         #     plt.legend()
    #         #     plt.show()
    #     try:
    #         os.mkdir(f'{self.path_prefix}_emf_fit/')
    #     except FileExistsError:
    #         pass

    #     fit_results_pickle_file = f'{self.path_prefix}_emf_fit/emf_params.pic'
    #     with open(fit_results_pickle_file, 'wb') as handle:
    #         pic.dump(emf_models, handle)

    def fit_emf_correlation_to_md(self, atom_names, diffusion_file,mf_config,  time_max=20e-9, ignore_list=(), sv=False):

        atom_info = self.make_atom_pairs_list(atom_names)
        dval, diffusion_errors = utils.read_diffusion_tensor(diffusion_file)
        emf_models = {}

        def model(params, args):
            c0, internal, total = correlation_functions.correlation_anisotropic_emf(params, args, sv=sv)
            return c0, internal, total

        def resid(params, args, target):
            _,_, model_spec = model(params, args)
            return model_spec - target


        for res1, res2, atom_name1, atom_name2 in tqdm(atom_info):

            if [atom_name1, atom_name2] not in ignore_list:
                name = f"{self.path_prefix}_rotacf/rotacf_{res1}_{atom_name1}_{res2}_{atom_name2}.xvg"
                pdf_name =f'{self.path_prefix}_rotacf/correlation_func_{res1}_{atom_name1}_{res2}_{atom_name2}_check.pdf'
                temp = self.read_gmx_xvg(name) 

                x = temp.T[0]
                y = temp.T[1]

                y = y[x<time_max]
                x = x[x<time_max]

                if sv == True:
                    y = y[x>0.]
                    x = x[x>0.]

                angles_key = (res1, atom_name1, res2, atom_name2)
                ex,ey,ez = self.cosine_angles[angles_key]
                args = (x,ex,ey,ez)

                model_free_class = mf.ModelFree(*dval, mf_config=mf_config)

                model_params = model_free_class.generate_mf_parameters(scale_diffusion=False)
                c0, _, _ = correlation_functions.correlation_anisotropic_emf(model_params[0], args)
                md_with_diff = y * c0
                mod = model_params[-1]
                
                if sv == True:
                    mod.add('Sv', min=0, max=1, value=0.9)

                minner = Minimizer(resid, mod, fcn_args=(args, md_with_diff))
                result = minner.minimize(method="powel")
                emf_models[angles_key] = result
                report_fit(result)

                current_model_0, mod_internal, mod_total = model(emf_models[angles_key].params, args)

                plt.plot(x,y, label='raw MD')

                plt.plot(x,c0, label='c0')
                plt.plot(x,md_with_diff, label='total')
                plt.plot(x,y,label='internal')
                
                plt.plot(x,current_model_0,':', label='model co')
                plt.plot(x,mod_internal,':', label='model internal')
                plt.plot(x,mod_total,':', label='model total')
                plt.legend()
                plt.savefig(pdf_name)
                plt.close()
                # os.exit

        try:
            os.mkdir(f'{self.path_prefix}_emf_fit/')
        except FileExistsError:
            pass

        fit_results_pickle_file = f'{self.path_prefix}_emf_fit/emf_params_from_correlations_check.pic'
        with open(fit_results_pickle_file, 'wb') as handle:
            pic.dump(emf_models, handle)

    def fit_emf_correlation_to_md_internal_only(self, 
        atom_names, 
        time_max=0.5e-9, 
        ignore_list=(), 
        fit_config='emf_to_md_correlation_function.toml',
        sv=False):

        atom_info = self.make_atom_pairs_list(atom_names)
        emf_models = {}

        def model(params, time, sv):
            internal = correlation_functions.correlation_emf_internal_only(params, time, sv=sv)
            return internal

        def resid(params, time, target, sv):
            internal_model = model(params, time, sv)
            return (internal_model - target)/(time*1e6 + 0.01) 

        fitting_starting_conditions = toml.load(fit_config)

        for res1, res2, atom_name1, atom_name2 in tqdm(atom_info):

            if [atom_name1, atom_name2] not in ignore_list:
                
                name = f"{self.path_prefix}_rotacf/rotacf_{res1}_{atom_name1}_{res2}_{atom_name2}.xvg"
                pdf_name =f'{self.path_prefix}_rotacf/correlation_func_{res1}_{atom_name1}_{res2}_{atom_name2}_internal.pdf'
                temp = self.read_gmx_xvg(name) 

                x = temp.T[0]
                y = temp.T[1]

                y = y[x<time_max]
                x = x[x<time_max]

                if sv == True:
                    y = y[x>0.]
                    x = x[x>0.]

                key = (res1, atom_name1, res2, atom_name2)

                Ss_grid = np.linspace(0.4, 1, 4)
                Sf_grid = np.linspace(0.4, 1, 4)
                tau_f = np.geomspace(1e-13, 1e-9, 10)
                tau_s = np.geomspace(0.5e-9, 10e-9, 10)

                if sv == True:
                    sv_array = np.linspace(0.7, 1, 10)
                else:
                    sv_array = np.array([1.])

                print('doing the grid search')
                lowest_rmsd = 1e100

                for i in itertools.product(Ss_grid, Sf_grid, tau_f, tau_s, sv_array):
                    Ss_curr, Sf_curr, tau_f_curr, tau_s_curr, Sv_current = i
                    if tau_f_curr < tau_s_curr:
                        values = {
                        'tau_f':tau_f_curr, 
                        'tau_s':tau_s_curr,
                        'Ss': Ss_curr, 
                        'Sf': Sf_curr,}

                        if sv == True:
                            values['Sv'] = Sv_current

                        rmsd_current = resid(values, x, y, sv)
                        rmsd_current = np.sum(rmsd_current**2)

                        if rmsd_current < lowest_rmsd:
                            best_model = i
                            lowest_rmsd = rmsd_current

                Ss_best, Sf_best, tau_f_best, tau_s_best, Sv_best = best_model

                print(best_model)

                params = Parameters()
                params.add('tau_f', value=tau_f_best, min=0, max=1e-9)
                params.add('tau_s', value=tau_s_best, min=0, max=5e-9)
                params.add('Ss', value=Ss_best, min=0.4, max=1)
                params.add('Sf', value=Sf_best, min=0.4, max=1)

                if sv == True:
                    params.add('Sv', min=0, max=1, value=Sv_best)

                minner = Minimizer(resid, params, fcn_args=(x, y, sv))
                result = minner.minimize(method="powel")
                emf_models[key] = result
                #report_fit(result)

                tauf = result.params['tau_f'].value * 1e12
                taus = result.params['tau_s'].value * 1e9

                Ss = result.params['Ss'].value
                Sf = result.params['Sf'].value

                mod_internal = model(params, x, sv= sv)
                plt.title(f'{res1} {atom_name1} {atom_name2} tauf: {tauf:0.2f}ps  Sf: {Sf:0.2f}   taus: {taus:0.2f}ns   S2: {Ss:0.2f}')
                plt.plot(x,y,label='MD raw')
                plt.plot(x, mod_internal, label='fit')
                plt.ylim(0.5, 1.1)
                plt.legend()
                plt.savefig(pdf_name)
                plt.show()
                plt.close()
                # os.exit

        try:
            os.mkdir(f'{self.path_prefix}_emf_fit/')
        except FileExistsError:
            pass

        fit_results_pickle_file = f'{self.path_prefix}_emf_fit/emf_params_from_correlations_interal.pic'
        with open(fit_results_pickle_file, 'wb') as handle:
            pic.dump(emf_models, handle)

    def fit_emf_correlation_three_tau_internal_only(self, 
        atom_names, 
        time_max=0.5e-9, 
        ignore_list=(), 
        fit_config='emf_to_md_correlation_function.toml'):

        atom_info = self.make_atom_pairs_list(atom_names)
        emf_models = {}

        def model(params, time):
            internal = correlation_functions.correlation_emf_three_tau_internal_only(params, time)
            return internal

        def resid(params, time, target):
            internal_model = model(params, time)
            return (internal_model - target)/(time*1e6 + 0.01) 

        fitting_starting_conditions = toml.load(fit_config)

        for res1, res2, atom_name1, atom_name2 in tqdm(atom_info):

            if [atom_name1, atom_name2] not in ignore_list:
                
                name = f"{self.path_prefix}_rotacf/rotacf_{res1}_{atom_name1}_{res2}_{atom_name2}.xvg"
                pdf_name =f'{self.path_prefix}_rotacf/correlation_func_{res1}_{atom_name1}_{res2}_{atom_name2}_three_tau_internal.pdf'
                temp = self.read_gmx_xvg(name) 

                x = temp.T[0]
                y = temp.T[1]

                y = y[x<time_max]
                x = x[x<time_max]

                key = (res1, atom_name1, res2, atom_name2)

                amp_vf = np.linspace(0, 0.2, 10)
                amp_f  = np.linspace(0, 0.2, 10)
                amp_s  = np.linspace(0, 0.4, 10)
                
                tau_vf =  np.geomspace(0.05e-12, 10e-12, 20)
                tau_f = np.geomspace(10e-12, 500e-12, 20)
                tau_s = np.geomspace(0.2e-9, 10e-9, 20)


                print('doing the grid search')
                lowest_rmsd = 1e10000

                for i in tqdm(itertools.product(amp_vf, amp_f, amp_s, tau_vf, tau_f, tau_s)):
                    
                    amp_vf_i, amp_f_i, amp_s_i, tau_vf_i, tau_f_i, tau_s_i = i
                    
                    sum_check = amp_vf_i + amp_f_i + amp_s_i
                    
                    run_point = True

                    if tau_vf_i > tau_f_i:
                        run_point = False
                    elif tau_f_i > tau_s_i:
                        run_point = False
                    elif sum_check > 1.:
                        run_point = False

                    if run_point:
                        #print('here', amp_vf_i, amp_f_i, amp_s_i, tau_vf_i, tau_f_i, tau_s_i)

                        values = {
                        'tau_vf':tau_vf_i, 
                        'tau_f':tau_f_i, 
                        'tau_s':tau_s_i,
                        'amp_vf': amp_vf_i, 
                        'amp_f': amp_f_i,
                        'amp_s': amp_s_i,
                        'S_long': 1-amp_vf_i-amp_f_i-amp_s_i}

                        rmsd_current = resid(values, x, y)
                        rmsd_current = np.sum(rmsd_current**2)

                        if rmsd_current < lowest_rmsd:
                            best_model = i
                            lowest_rmsd = rmsd_current

                amp_vf_best, amp_f_best, amp_s_best, tau_vf_best, tau_f_best, tau_s_best = best_model

                params = Parameters()
                params.add('tau_vf', value=tau_vf_best, min=0, max=1e-9)
                params.add('tau_f', value=tau_f_best, min=0, max=1e-9)
                params.add('tau_s', value=tau_s_best, min=0, max=5e-9)
                
                params.add('amp_vf', value=amp_vf_best, min=0, max=0.6)
                params.add('amp_f', value=amp_f_best, min=0, max=0.6)
                params.add('amp_s', value=amp_s_best, min=0, max=0.6)
                params.add('S_long', expr='1-amp_vf-amp_f-amp_s', min=0, max=1)

                minner = Minimizer(resid, params, fcn_args=(x, y))
                result = minner.minimize(method="powel")
                emf_models[key] = result
                report_fit(result)

                tauvf = result.params['tau_vf'].value * 1e12
                tauf = result.params['tau_f'].value * 1e12
                taus = result.params['tau_s'].value * 1e9

                ampvf = result.params['amp_vf'].value
                ampf = result.params['amp_f'].value
                amps = result.params['amp_s'].value

                mod_internal = model(params, x)
                plt.title(f'{res1} {atom_name1} {atom_name2} tauvf: {tauvf:0.2f}ps  Avf: {ampvf:0.2f} tauf: {tauf:0.2f}ps  Af: {ampf:0.2f}   taus: {taus:0.2f}ns   As: {amps:0.2f}')
                plt.plot(x,y,label='MD raw')
                plt.plot(x, mod_internal, label='fit')
                plt.ylim(0.5, 1.1)
                plt.legend()
                plt.savefig(pdf_name)
                #plt.show()
                plt.close()
                # os.exit

        try:
            os.mkdir(f'{self.path_prefix}_emf_fit/')
        except FileExistsError:
            pass

        fit_results_pickle_file = f'{self.path_prefix}_emf_fit/emf_params_from_correlations_interal_three_tau.pic'
        with open(fit_results_pickle_file, 'wb') as handle:
            pic.dump(emf_models, handle)



    def plot_spectral_densities(self,r1_file, spectral_density_file, diffusion_file,):
        '''
        This function calculates the spectral density for a single trajecotry. 
        https://pubs.acs.org/doi/pdf/10.1021/ja001129b
        '''

        try:
            os.mkdir(f'{self.path_prefix}_plot_spec_dens/')
        except FileExistsError:
            pass

        # this is a pretty ugly way of doing things but maybe it will work for now ...
        r1, _ = d2e.utils.read_nmr_relaxation_rate(r1_file)
        dval, diffusion_errors = utils.read_diffusion_tensor(diffusion_file)
        spectral_density_params_dict = utils.read_fitted_spectral_density(spectral_density_file)
        
        for i in r1:
            try:
                # this try statement if you want to look at fewer atoms 
                # need a better fix in future
                check = False

                # get the atom info earlier in the funtion that I originally thought 
                atom1_letters, atom1_numbers, atom1_res_type, atom1_resid, atom1_type, \
                atom2_letters, atom2_numbers, atom2_res_type, atom2_resid, atom2_type = utils.get_atom_info_from_rate_key(i)
                nucleus_type = atom2_letters[1].lower()
                atom_full_names = (atom2_letters[1] + atom2_numbers[1], atom1_letters[1] + atom1_numbers[1])

                #key for the spectral density
                spec_key = (int(atom1_numbers[0]), atom2_letters[1] + atom2_numbers[1]+','+atom1_letters[1] + atom1_numbers[1])
                angs = self.cosine_angles[ (int(atom2_resid), atom2_type, int(atom1_resid), atom1_type)]

                params = spectral_density_params_dict[spec_key]
                params['dx'] = dval[0]
                params['dy'] = dval[1]
                params['dz'] = dval[2]

                model_x = np.linspace(0.1, 2*np.pi*3e9+0.1, 200)-0.1
                args = [model_x, angs[0], angs[1], angs[2]]
                model_y = self.spectral_density_anisotropic(params, args)
                name = f'{self.path_prefix}_plot_spec_dens/{atom1_resid}_{atom_full_names[0]}_{atom_full_names[1]}.pdf'
                plt.plot(model_x, model_y)
                
                for field in (11.7467, 14.1, 18.7929, 23.4904):
                    h = PhysQ.calc_omega('h', field)
                    c = PhysQ.calc_omega('c', field)
                    omegas = np.array([0, c, h-c, c + h, h])
                    txt = ['J(0)', 'J(c)', 'J(h-c)', 'J(h+c)', 'J(h)']
                    #print(omegas)
                    args = [omegas, angs[0], angs[1], angs[2]]
                    omega_model = self.spectral_density_anisotropic(params, args)
                    plt.scatter(omegas, omega_model, label=str(field)+'T')

                for i,j,k in zip(omegas, txt, omega_model):
                    plt.text(i+0.2e9, k,j)
                #plt.xscale('symlog')
                plt.yscale('log')
                plt.legend()
                plt.savefig(name)
                plt.close()

            except KeyError:
                pass

