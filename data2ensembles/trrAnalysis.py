
import data2ensembles.utils as utils
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

    def load_trr(self,):
        '''
        load the trajectory
        '''

        if self.xtc == None:
            uni = md.Universe(self.gro)
        elif self.xtc != None and self.tpr != None:
            uni = md.Universe(self.gro, self.xtc)
        else:
            print('Something went wrong ... not sure what to load in')
        self.uni = uni

    def make_ndx_file(self, atom_selection_pairs, index_name,supress=True):
        '''
        make the index file for the gmx rotacf command
        '''
        indx = open(index_name,'w')
        indx.write('[indx]\n')
        for i,j in atom_selection_pairs:

            a = self.uni.select_atoms(i)
            b = self.uni.select_atoms(j)

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

    def calc_rotacf(self, indx_file, out_name, atom_selection_pairs, b=None, e=None, dt=None, xtc=None):
        '''
        calculate the rotational correlation function using gromacs 
        '''
        if xtc == None:
            xtc = self.xtc

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

        print('reading %s'%(out_name+'.xvg'))
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

        xvg.close()
        total = np.array(total)

        for indx, i in enumerate(atom_selection_pairs):
            s0 = i[0].split()
            s1 = i[1].split()
            f = open(out_name+f'_{s0[1]}_{s0[4]}_{s1[1]}_{s1[4]}.xvg', 'w')

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


    def extract_diffusion_rotacf(self, internal_dir, total_dir):
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

        for internal_f in internal_motion_files:
            file = internal_f.split('/')[-1]
            total_f = total_dir + '/' + file

            #print(internal_f, total_f)

            if total_f in total_motion_files:

                internal_x, internal_y = self.read_gmx_xvg(internal_f).T
                total_x, total_y = self.read_gmx_xvg(total_f).T
                diffusion_y = total_y/internal_y
                file_name = f'{self.path_prefix}_diffusion_rotacf/{file}'

                f = open(file_name, 'w')
                for i,j in zip(internal_x, diffusion_y):
                    f.write(f'{i} {j}\n')
                f.close()

                plt.plot(internal_x, internal_y, label='internal')
                plt.plot(total_x, total_y, label='total')
                plt.plot(total_x, diffusion_y, label='diffusion')
                plt.legend()
                pdf_name = file.split('.')[0]+'.pdf'
                plt.savefig(f'{self.path_prefix}_diffusion_rotacf_pdfs/{pdf_name}')
                plt.close()

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

    def fit_correlation_function(self,x,y, log_time_min=50, log_time_max=5000):
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

        def calc_jw(time, internal_corelations,):

            '''
            This function is used to calculate an numerical spectral density function
            here we assume an overall isotropic tumbling. In reality this is just a way to focus the fitting
            on a region of interest.

            Here the spectral density is calculated numerically
            '''

            dt = time[1]-time[0]
            tumbling = 0.2*(np.e**(-time/dummy_tauc))
            total = tumbling * internal_corelations

            j_fft = scipy.fft.fft(total)*dt*2
            j_fft = scipy.fft.fftshift(j_fft)
            j_fft_freq = scipy.fft.fftfreq(len(time), d=dt)
            j_fft_freq = scipy.fft.fftshift(j_fft_freq)
            j_fft_freq = np.pi*j_fft_freq*2
            return j_fft_freq, j_fft

        def spectral_density_fuction(params, omega):

            term2 = params['S_long']*dummy_tauc/(1+(omega*dummy_tauc)**2)
            total = term2

            for i in range(curve_count):
                i = i + 1

                #correction for fitting the correlation times in ps
                taui = params['time_%i'%(i)]
                ampi = params['amp_%i'%(i)]


                tau_eff = taui*dummy_tauc/(dummy_tauc+taui)
                term1 = ampi*tau_eff/(1+(omega*tau_eff)**2)
                total = total + term1

            return (2/5.)*total


        def func2min(Params, x,y):

            '''
            The function we want to minimize. The fitting is carried out both in s and hz.
            '''

            correlation_diff =  y - correlation_function(Params, x)

            x = x
            dt = x[1]-x[0]
            tumbling = 0.2*(np.e**(-x/dummy_tauc))
            total = tumbling * y

            j_fft = scipy.fft.fft(total)*dt*2
            j_fft = scipy.fft.fftshift(j_fft)
            j_fft = j_fft.real
            j_fft_freq = scipy.fft.fftfreq(len(x), d=dt)
            j_fft_freq = scipy.fft.fftshift(j_fft_freq)
            j_fft_freq = np.pi*j_fft_freq*2

            spec_difference = j_fft - spectral_density_fuction(Params, j_fft_freq)

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

            # create a set of Parameters
            Params.add('amp_%i'%(i), value=1/curve_count, min=0, max=1.)
            amp_list.append('amp_%i'%(i))
            Params.add('time_%i'%(i), value=taui, min=0)
            i = 1+1

        #these final parameters are used to ensure the sums are all correct
        expression = '1 - (' + ' + '.join(amp_list) + ' + S_long)'
        Params.add('amp_%i'%(curve_count), expr=expression, min=0, max=1.)
        Params.add('time_%i'%(curve_count), value=log_times[-1], min=0)

        # do fit, here with the default leastsq algorithm

        minner = Minimizer(func2min, Params, fcn_args=(x, y))
        result = minner.minimize(xtol=np.mean(x)*1e-7)
        #report_fit(result)
        return result

    def fit_all_correlation_functions(self,
        atom_names, time_threshold=10000,
        log_time_min=50, log_time_max=5000,):

        atom1 = atom_names[0]
        atom2 = atom_names[1]

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

        block_count = self.get_number_of_blocks(atom_names)
        manager = enlighten.get_manager()
        blocks_pbar = manager.counter(total=block_count, desc="Blocks", unit="Ticks", color="red")
        residues_pbar = manager.counter(total=len(self.uni.residues)*block_count, desc="residues", unit="Tocks", color="blue")

        for block in range(block_count):

            params_out = open(f'{self.path_prefix}_fit_params/{block}.dat', 'w')
            params_out.write('# residue:atoms:S2:amps:times\n')
            for i in self.uni.residues:
                name = f"{self.path_prefix}_rotacf/rotacf_block_{block}_{i.resid}_{atom1}_{i.resid}_{atom2}.xvg"

                #Sometimes we dont want to fit the whole Ci(t) as it will often plateau. This time is in ps
                temp = self.read_gmx_xvg(name)
                x = temp.T[0]
                y = temp.T[1]

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
                plt.savefig(f'{self.path_prefix}_fits/rot_acf_{block}_{i.resid}_{atom1}.pdf')
                plt.close()
                residues_pbar.update()

                amps = [str(values['amp_%i'%(i+1)]) for i in range(self.curve_count)]
                amps = ','.join(amps)
                times = [str(values['time_%i'%(i+1)]) for i in range(self.curve_count)]
                times = ','.join(times)
                slong = values['S_long']
                line = f'{i.resid}:{atom1},{atom2}:{slong}:{amps}:{times}\n'
                params_out.write(line)
                params_out.flush()

            blocks_pbar.update()
        manager.stop()

    def calc_diffusion_tensor_from_trr_msd(self,seg_blocks=None,trestart=1, use_unaligned_xtc=False):

        '''
        In this function we use gmx msd to calculate the diffusion tensor.
        I am not completely sure what frame that we get the tensor in is so we need to rotate it to
        our reference frame

        Following this we align the diffusion tensor to the axis
        reference_system = np.array([[1,0,0],[0,1,0],[0,0,1]])

        since this is the axis system that the moment of inertia is aligned along they should match.
        '''

        # make the output directory
        try:
            os.mkdir(f'{self.path_prefix}_diffusion_tensors')
        except FileExistsError:
            pass

        # use the unaligned trajectory ? - You probably want to do this ...
        if use_unaligned_xtc == True:
            if self.unaligned_xtc == None:
                raise("Please set self.unaligned_xtc = 'the path to the unaligned trrr' ")
            else:
                trr = self.unaligned_xtc
        else:
            trr = self.xtc


        base_command = f'{self.gmx} msd -f {trr} -s {self.gro} -ten -trestart {trestart} '

        if seg_blocks == None:
            print('calculating diffusion tensor for the whole trr!')
            indx = 1
            outname = f'{self.path_prefix}_diffusion_tensors/block_{indx}.xvg'
            command = base_command + f' -o {outname} <<< 0 \n'
            os.system(command)

        else:
            print('calculating diffusion tensor for segments')
            for indx, i in enumerate(seg_blocks):
                outname = f'{self.path_prefix}_diffusion_tensors/block_{indx}.xvg'
                command = base_command + f' -o {outname} -b {i[0]} -e {i[1]} <<< 0 \n'
                os.system(command)

        print('Rotating the diffusion tensors!')
        try:
            os.mkdir(f'{self.path_prefix}_diffusion_tensors_aligned')
        except FileExistsError:
            pass

