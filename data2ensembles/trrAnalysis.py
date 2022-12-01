
import data2ensembles.utils as utils
import data2ensembles.mathFuncs as mathFunc
import data2ensembles.rates
import data2ensembles as d2e
import data2ensembles.structureUtils as strucUtils

import os
import numpy as np
#import sys
import random as r
import copy
from tqdm import tqdm
import MDAnalysis as md
from MDAnalysis import transformations

from lmfit import Minimizer, Parameters, report_fit
from tqdm import tqdm
import enlighten

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


        # the number of curves to use when fitting the internal correlation functions
        self.curve_count = 3
        # This is a dummy correlation time used to help fit the internal correlation times
        # note that this should be seconds
        self.dummy_tauc = 5e-6

        # this dictionary contains the transform information from a local geometry to
        # the principle axis of the CSA tensor
        # this is created by  calc_cosine_angles
        self.csa_tensor_transform = None

    def write_diffusion_trace(self, params, file):

        name = f"{self.path_prefix}_diffusion_rotacf_fit/{file}"
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

        all_atom_res_types = self.uni.atoms.residues.resnames
        all_atom_res_ids = self.uni.atoms.residues.resids

        for i,j in zip(all_atom_res_ids, all_atom_res_types):
            if i not in self.resid2type:
                self.resid2type[i] = j



    def make_ndx_file(self, atom_info, index_name,supress=True):
        '''
        make the index file for the gmx rotacf command
        '''
        indx = open(index_name,'w')
        indx.write('[indx]\n')
        atom_info = self.make_atom_pairs_list(atom_info)

        for res1, res2, atom_name1, atom_name2 in atom_info:
            #print(f'resid {res1} and name {atom_name1}\nresid {res2} and name {atom_name2}')
            a = self.uni.select_atoms(f'resid {res1} and name {atom_name1}')
            b = self.uni.select_atoms(f'resid {res2} and name {atom_name2}')

            #python numbering!
            a_indx = a[0].ix+1
            b_indx = b[0].ix+1

            indx.write(f'{a_indx} {b_indx}\n')
        indx.close()

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
        for i in fi.readlines():
            if i[0] not in ('#', '@'):
                s = i.split()
                values.append([float(j) for j in s])

        fi.close()
        return np.array(values)

    def make_atom_pairs_list(self, atoms):
        '''
        Many functions want to iterate over residues and atom types. This function created this list
        This function returns a list where of atom pair infos

        [[res1 res2 atomname1 atomname2] ... N]
        '''

        # I could probably think of a better wat to do this
        atom_info = []
        for i in self.uni.residues:
            res1_check = False
            res2_check = False

            uni_res_atom_names = [k.name for k in  i.atoms]
            for a in atoms:
                if a[0] in uni_res_atom_names:
                    if a[1] in uni_res_atom_names:
                        atom_info.append([i.resid, i.resid, a[0], a[1]])

        return atom_info

    def calc_cosine_angles(self,
                        atom_names, 
                        calc_csa_angles=True, 
                        skip=1000, 
                        calc_average_structure=True, 
                        write_out_angles=False):

        # try calculating the principle axis with gromacs
        # then average them and determine all the angles and 
        # average those

        # def average_principle_axis(a):

        #     a = np.average(a,axis=0) 
        #     a = a[1:4]
        #     return a

        # calculate average structure
        average_pdb = self.path_prefix+'_average.pdb'

        if calc_average_structure == True:
            gmx_command = f'{self.gmx} rmsf -f {self.xtc} -s {self.gro} -dt 1000 -ox {average_pdb} << EOF\n 0 \nEOF'
            os.system(gmx_command)

        self.average_uni = md.Universe(average_pdb)

        # here the larges axis is axis[2] this should be considered to be 'Dz' from the diffusion tensor
        # or in the case of a sphereoid D||
        axis = self.average_uni.select_atoms('all').principal_axes()

        #this file can be written out as a check if desired
        if write_out_angles == True:
            angles_out = open(self.path_prefix + '_dipolar_angles.dat', 'w')
            
            #write out the principle axis
            pas_out = open(self.path_prefix + '_principle_axis.dat', 'w')
            labels = ['large', 'middle', 'small']
            for i,j in zip(axis, labels):
                #this should go x,y,x
                pas_out.write(f'{j} {i[0]:0.2f} {i[1]:0.2f} {i[2]:0.2f}\n')
            pas_out.close()

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
            self.cosine_angles[key] = list(angs)

            if write_out_angles == True:

                # explicit for my own understanding ...
                angx = np.rad2deg(np.arccos(angs[0]))
                angy = np.rad2deg(np.arccos(angs[1]))
                angz = np.rad2deg(np.arccos(angs[2]))
                angles_out.write(f'{res1}\t{atom_name1}\t{angx}\t{angy}\t{angz}\n')

            if calc_csa_angles == True:
                
                # here we calculate all the CSA tensor angles 
                # but I think we only need the cosine angles for d11 - Need to check
                d_selected, (d11, d22, d33) = mathFunc.calc_csa_axis(res1, atom_name1, 
                    self.csa_tensor_transform[atom1_resname][atom_name1], 
                    self.average_uni)

                csa_angs = mathFunc.cosine_angles(d_selected, axis)
                self.csa_cosine_angles[key] = list(csa_angs)

        if write_out_angles == True:
            angles_out.close()

    def calc_rotacf(self, indx_file, atom_names, b=None, e=None, dt=None, xtc=None, timestep=2e-12):
        '''
        calculate the rotational correlation function using gromacs 

        the time step is to account for the fact that in the gromacs .xvg files for rotacf it gives the time step 
        not the actual time.
        '''
        if xtc == None:
            xtc = self.xtc

        try:
            os.mkdir(f'{self.path_prefix}_rotacf')
        except FileExistsError:
            pass

        out_name = f'{self.path_prefix}_rotacf/rotacf.xvg'
        print('out name:', out_name)
        command = f'{self.gmx} rotacf -s {self.tpr} -f {xtc} -o {out_name}.xvg -n {indx_file} -P 2 -d -noaver -xvg none'
        print('>>>>' + command)
        if b != None:
            command = command + f' -b {b} '
        if e != None:
            command = command + f' -e {e} '
        if dt != None:
            command = command + f' -dt {dt}'
        print(f'Running the command: \n {command}')
        os.system(command)

        print('reading %s'%(out_name))
        xvg = open(out_name+'.xvg')
        time = []
        total = []

        data = []
        time_check = False
        for i in xvg.readlines():
            s = i.split()


            if s[0] == '&':
                check = True
                total.append(data)
                data = []
            else:
                data.append(float(s[1]))

                if time_check == False:
                    time.append(float(s[0]))

        print('time step', timestep)
        time = timestep * np.array(time)
        xvg.close()
        total = np.array(total)

        atom_info = self.make_atom_pairs_list(atom_names)
        for indx , (res1, res2, atom_name1, atom_name2)  in enumerate(atom_info):
            f = open(f'{self.path_prefix}_rotacf/rotacf'+f'_{res1}_{atom_name1}_{res2}_{atom_name2}.xvg', 'w')

            for ti, coeffi in zip(time, total[indx]):
                f.write(f'{ti} {coeffi}\n')
            f.close()

        os.remove(out_name+'.xvg')

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



    def correlation_function(self, Params, x):

        '''
        An internal correlation function. This is the same one that is used
        by Kresten in the absurder paper.
        '''

        total = Params['S_long']
        for i in range(self.curve_count):
            i = i+1
            amp = 'amp_%i'%(i)
            time = 'time_%i'%(i)

            total = total + Params[amp]*(np.e**(-1*x/Params[time]))

        return total

    def spectral_density_fuction(self,params, omega, dummy_tauc=5e-6):

        term2 = params['S_long']*dummy_tauc/(1+(omega*dummy_tauc)**2)
        total = term2

        for i in range(self.curve_count):
            i = i + 1

            #correction for fitting the correlation times in ps
            taui = params['time_%i'%(i)]
            ampi = params['amp_%i'%(i)]


            tau_eff = taui*dummy_tauc/(dummy_tauc+taui)
            term1 = ampi*tau_eff/(1+(omega*tau_eff)**2)
            total = total + term1

        return (2/5.)*total
    
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
        The aim of this function is to take internal correlation funstion from MD 
        (see self.spectral_density_fuction and spectral_density_fuction) and to apply and anisotropic diffusion 
        so that one could then calculate NMR relaxation rates.

        maybe I should move this to the spectral density module. 
        '''

        # unpack varriables
        dx = params['dx']
        dy = params['dy']
        dz = params['dz']

        omega, ex,ey,ez, = args 
        taus, amps = mathFunc.calculate_anisotropic_d_amps(dx,dy,dz,ex,ey,ez)

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

            #now we do the internal motions 
            for i in range(self.curve_count):
                i = i + 1

                #correction for fitting the correlation times in ps
                tau_internal = params['time_%i'%(i)]
                amp_internal = params['amp_%i'%(i)]

                tau_eff = taui*tau_internal/(tau_internal+taui)
                term_internal = amp_internal*ampi*tau_eff/(1+(omega*tau_eff)**2)
                
                total = total + term_internal

        return (2./5.)*total

    def fit_correlation_function(self,x,y, log_time_min=5e-13, log_time_max=10e-9, blocks=True):
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

        def func2min(Params, x,y):

            '''
            The function we want to minimize. The fitting is carried out both in s and hz.
            '''

            correlation_diff =  y - correlation_function(Params, x)

            x = x
            dt = x[1]-x[0]
            tumbling = 0.4*(np.e**(-x/dummy_tauc))
            total = tumbling * y

            j_fft = scipy.fft.fft(total)*dt*2
            j_fft = scipy.fft.fftshift(j_fft)
            j_fft = j_fft.real
            j_fft_freq = scipy.fft.fftfreq(len(x), d=dt)
            j_fft_freq = scipy.fft.fftshift(j_fft_freq)
            j_fft_freq = np.pi*j_fft_freq*2
            spec_difference = j_fft - self.spectral_density_fuction(Params, j_fft_freq,  dummy_tauc = dummy_tauc)

            all_diffs = np.concatenate([spec_difference, correlation_diff])
            return all_diffs.flatten()


        # create a set of Parameters
        Params = Parameters()
        s_guess = np.mean(y[int(len(y)*0.75):])
        Params.add('S_long', value=s_guess, min=0, max=1)

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
        minner = Minimizer(func2min, Params, fcn_args=(x, y))
        result = minner.minimize()
        report_fit(result)
        return result

    
    def fit_all_correlation_functions(self,
        atom_names, time_threshold=10e-9,
        log_time_min=50, log_time_max=5000,blocks=True):

        atom_info = self.make_atom_pairs_list(atom_names)
        atom1 = atom_names[0][0]
        atom2 = atom_names[0][1]

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

        if blocks==True:
            block_count = self.get_number_of_blocks(atom_names)
        else:
            block_count = 1

        #manager = enlighten.get_manager()
        #blocks_pbar = manager.counter(total=block_count, desc="Blocks", unit="Ticks", color="red")
        #residues_pbar = manager.counter(total=len(self.uni.residues)*block_count, desc="residues", unit="Tocks", color="blue")

        for block in range(block_count):

            params_out = open(f'{self.path_prefix}_fit_params/internal_correlations_{block}.dat', 'w')
            params_out.write('# residue:atoms:S2:amps:times\n')
            
            
            for res1, res2, atom_name1, atom_name2 in atom_info:

                if blocks == True:
                    name = f"{self.path_prefix}_rotacf/rotacf_block_{block}_{res1}_{atom_name1}_{res2}_{atom_name2}.xvg"
                else:
                    name = f"{self.path_prefix}_rotacf/rotacf_{res1}_{atom_name1}_{res2}_{atom_name2}.xvg"

                print('>fitting rotacf for:', name)

                #Sometimes we dont want to fit the whole Ci(t) as it will often plateau. This time is in ps
                temp = self.read_gmx_xvg(name)
                x = temp.T[0]
                y = temp.T[1]

                #print(min(x), max(x), time_threshold)
                y = y[x<time_threshold]
                x = x[x<time_threshold]

                #print('starting fit ...')
                result = self.fit_correlation_function(x,y)
                values = result.params.valuesdict()
                x_model = np.linspace(0, max(x), 10000)
                y_model = self.correlation_function(values, x_model)

                #print('plotting...')
                plt.plot(x,y, label='raw acf')
                plt.plot(x_model, y_model, label='fit')
                plt.legend()
                plt.xscale('symlog')
                plt.savefig(f'{self.path_prefix}_fits/rot_acf_{block}_{res1}_{atom_name1}.pdf')
                plt.close()
                #residues_pbar.update()

                amps = [str(values['amp_%i'%(i+1)]) for i in range(self.curve_count)]
                amps = ','.join(amps)
                times = [str(values['time_%i'%(i+1)]) for i in range(self.curve_count)]
                times = ','.join(times)
                slong = values['S_long']
                line = f'{res1}:{atom_name1},{atom_name2}:{slong}:{amps}:{times}\n'
                params_out.write(line)
                params_out.flush()

            #locks_pbar.update()
        #manager.stop()

    
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
        cosine_angles = []
        #axis = self.uni.select_atoms('all').principal_axes()

        for res1, res2, atom_name1, atom_name2 in atom_info:
            if blocks == True:
                name = f"{self.path_prefix}_diffusion_rotacf/rotacf_block_{block}_{res1}_{atom_name1}_{res2}_{atom_name2}.xvg"
            else:
                name = f"{self.path_prefix}_diffusion_rotacf/rotacf_{res1}_{atom_name1}_{res2}_{atom_name2}.xvg"

            #print(name)
            #print(name)
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

        #define the model 
        def model(params, time, cosine_angles):

            dx = params['dx']
            dy = params['dy']
            dz = params['dz']
            ds = (dx,dy,dz) 
            return self.correlation_function_anisotropic(time, ds, cosine_angles)

        def residual(params, cosine_angles, correlation_functions, time):

            diffs = []

            for angles, corri in zip(cosine_angles, correlation_functions):
                current_model = model(params, time, angles)
                current_diffs = corri - current_model
                diffs.append(current_diffs)


            return np.array(diffs)

        Params = Parameters()
        Params.add('dx', value=1/(3e-9), min=0,)
        Params.add('dy', value=1/(3e-9), min=0,)
        Params.add('dz', value=1/(5e-9), min=0,)

        minner = Minimizer(residual, Params, fcn_args=(cosine_angles, correlation_functions, x))
        result = minner.minimize()
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
            plt.xscale('symlog')
            plt.savefig(name)
            plt.close()

        self.write_diffusion_trace(res_params, "diffusion_tensor.dat")
    
    
    def calculate_r1_r2_hetnoe(self, atom_names, diffusion_file, fields,x, y='h', blocks=False,dna=False, write_out=False, prefix=''):
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
        else:
            print('LUCAS YOU NEED TO CODE THIS!')

        spectrail_density_params = utils.read_fitted_spectral_density(spec_file)

        if write_out == True:


            print(f'Writing to:')
            print(f'{self.path_prefix}_calculated_relaxation_rates/{prefix}r1.dat)')
            print(f'{self.path_prefix}_calculated_relaxation_rates/{prefix}r2.dat')
            print(f'{self.path_prefix}_calculated_relaxation_rates/{prefix}hetnoe.dat')
            
            r1_out = open(f'{self.path_prefix}_calculated_relaxation_rates/{prefix}r1.dat', 'w')
            r1_out.write('#header bondName \n' )
            r2_out = open(f'{self.path_prefix}_calculated_relaxation_rates/{prefix}r2.dat', 'w')
            r2_out.write('#header bondName \n' )
            hetnoe_out = open(f'{self.path_prefix}_calculated_relaxation_rates/{prefix}hetnoe.dat', 'w')
            hetnoe_out.write('#header bondName \n' )

        for res1, res2, atom_name1, atom_name2 in atom_info:

            #write out location
            if blocks == True:
                name = f"{self.path_prefix}_diffusion_rotacf/rotacf_block_{block}_{res1}_{atom_name1}_{res2}_{atom_name2}.xvg"
            else:
                name = f"{self.path_prefix}_diffusion_rotacf/rotacf_{res1}_{atom_name1}_{res2}_{atom_name2}.xvg"


            angs = self.cosine_angles[(res1, atom_name1, res2, atom_name2)]
            csa_angs = self.csa_cosine_angles[(res1, atom_name1, res2, atom_name2)]
            rxy = PhysQ.bondlengths[atom_name1, atom_name2]
            spectral_density_key = (res1, atom_name1+','+atom_name2)
            params = spectrail_density_params[spectral_density_key]

            #now we need to add the diffusion parameters \
            params['dx'] = diffusion_values[0]
            params['dy'] = diffusion_values[1]
            params['dz'] = diffusion_values[2]

            if dna == True:
                csa_atom_name = (atom_name1, self.resid2type[res1][1])
                resname = self.resid2type[res1][1]
            else:
                csa_atom_name = (atom_name1, self.resid2type[res1])
                resname = self.resid2type[res1]



            spectral_density = self.spectral_density_anisotropic
            r1 = d2e.rates.r1_YX(params, spectral_density, fields,rxy, csa_atom_name, x, y='h', 
                                cosine_angles = angs, csa_cosine_angles=csa_angs)

            r2 = d2e.rates.r2_YX(params, spectral_density, fields,rxy, csa_atom_name, x, y='h', 
                                cosine_angles = angs, csa_cosine_angles=csa_angs)
            hetnoe = d2e.rates.noe_YX(params, spectral_density, fields,
                               rxy, x, r1, y='h', cosine_angles=angs)
            
            if write_out == True:
                key = resname+str(res2)+atom_name2+'-'+resname+str(res1)+atom_name1
                #print(res1, res2, atom_name1, atom_name2, write_out, key)
                r1_str = ' '.join([str(a) for a in r1])
                r2_str = ' '.join([str(a) for a in r2])
                hetnoe_str = ' '.join([str(a) for a in hetnoe])
                r1_out.write(f"{key} {r1_str}\n")
                r2_out.write(f"{key} {r2_str}\n")
                hetnoe_out.write(f"{key} {hetnoe_str}\n")

        if write_out == True:
            r1_out.close()
            r2_out.close()
            hetnoe_out.close()

    def fit_diffusion_to_r1_r2_hetnoe(self, r1_file, r1_error, r2_file, r2_error, hetNoe_file, hetNoe_errors, spectral_density_file,
                                      fields,x, y='h', blocks=False,dna=False, write_out=False, reduced_noe=False,
                                      error_filter=0.05, PhysQ=PhysQ, model="anisotropic", scale_model='default'):
    
        def resid(params, values, errors, csa, bondlength, cosine_angles, spec_params, fields, csa_cosine_angles):

            spec_den = self.spectral_density_anisotropic
            total = []

            for vali, erri, csai, bondlengthi, angi, speci, csa_angi in zip(values, errors, csa, bondlength, cosine_angles, spec_params, csa_cosine_angles):
                speci['dx'] = params['dx']
                speci['dy'] = params['dy']
                speci['dz'] = params['dz']

                model_r1 = d2e.rates.r1_YX(speci, spec_den, fields, bondlengthi, csai, x, cosine_angles=angi, csa_cosine_angles=csa_angi)
                model_r2 = d2e.rates.r2_YX(speci, spec_den, fields, bondlengthi, csai, x, cosine_angles=angi, csa_cosine_angles=csa_angi)

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

                    spectral_density_params.append(spectral_density_params_dict[spec_key])

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
                    csa_cosine_angles.append(angs)
            
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
                           spectral_density_params, fields, csa_cosine_angles))
        result = minner.minimize()
        res_params = result.params
        report_fit(result)

        diso = (res_params['dx'].value + res_params['dy'].value + res_params['dz'].value)/3
        iso_tc = 1/(6*diso)
        print(f'isotropic tau c: {iso_tc * 1e9:0.2} ns' )
        self.write_diffusion_trace(res_params, "diffusion_tensor_fitted.dat")

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

