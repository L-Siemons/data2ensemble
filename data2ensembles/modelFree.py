import data2ensembles.utils as utils
import data2ensembles.trrAnalysis as trrAnalysis
import data2ensembles.mathFuncs as mathFunc
import data2ensembles.spectralDensity as specDens
import data2ensembles.rates
import data2ensembles as d2e
import data2ensembles.structureUtils as strucUtils
import data2ensembles.shuttling as shuttling
import data2ensembles.relaxation_matricies as relax_mat
import data2ensembles.adaptive_sampler as adaptive

from timeit import default_timer as timer
from datetime import timedelta
import toml

import scipy
import scipy.interpolate

import matplotlib
import matplotlib.cm as cm
from matplotlib.colors import LogNorm

import multiprocessing as mp
import pickle as pic
import re
import os
import numpy as np
import random as r
import copy
from tqdm import tqdm
import glob
import time
import math 
import pkg_resources
import MDAnalysis as md
import itertools

from lmfit import Minimizer, Parameters, report_fit
from tqdm import tqdm
import enlighten

import matplotlib.pyplot as plt
import sys

PhysQ = utils.PhysicalQuantities()


#print(data2ensembles.rates.__file__)

class ModelFree():
    """This class impliments model free calculations

    Parameters:
    -----------
    dx : float
        Dxx component of the diffusion tensor
    dy : float
        Dyy component of the diffusion tensor
    dz : float
        Dzz component of the diffusion tensor
    mf_config : str 
        default or string to a toml file with the parameters for the model free

    Attributes
    ----------
    r1_path : str, None
        path to r1 data

    r2_path : str, None
        path to r2 data

    r1_err_path : str, None
        path to r1 errors

    r2_err_path : str, None
        path to r2 errors

    hetnoe_path : str, None
        path to hetronuclear NOE data

    hetnoe_err_path : str, errors
        path to hetronuclear NOE data

    path_prefix : str 
        this is a prefix used for the output files

    sigma_path : str, None
        path to sigma NOE data

    sigma_err_path : str, errors
        path to sigma NOE data

    PhysQ : utils.PhysicalQuantities()
        This attribute is a class containing information about 
        physical qunatities. It can be changed to a new / edited copy if desired. 

    diffusion_tensor : None, list
        diffusion tensor. Currently it seems like this is not in use 

    high_fields : ndarray 
        the high fields for the R1, R2 and het NOE experiments

    model_high_fields : ndarray
        high fields used for drawing a smooth model line

    spectral_density : function from data2ensembles.spectralDensity
        This is the model spectral density function. The default is 
        data2ensembles.spectralDensity.J_anisotropic_emf

    scale_diffusion : bool 
        apply a linear scaling to the diffusion tensor when fitting extended model free
    
    diffusion : list 
        contains the diagonalised diffusion tensor (dx,dy,dz)


    Methods
    -------
    load_data()
        loads in the high field data after the paths are set

    load_low_field_info(file, distance_incrementing=0.5)
        this loads the low field information

    load_sigma()
        load the files containing sigma (reduced NOE)

    calc_cosine_angles_and_distances(atom_names, dist_skip=1)
        This calculates the cosine angles and distances from the reference structure

    get_atom_info_from_tag(i)
        gets the atom info from the tag 

    get_c1p_keys()
        gets the keys for the C1' atoms in DNA/RNA

    load_low_field_data(prefix)
        This loads the intensities of relaxometry datasets that have been fit
        with peakfit. 

    set_low_field_errors(fields, errors)
        set the low field errors if you want different values to what is in the 
        peak fit files

    read_pic_for_plotting(model_pic, atom_name)
        reads the picle file and returns some information 

    model_hf(params,res_info, fields)
        models the highfield data

    model_lf_single_field_intensities(params, low_field, res_info, protons)
        models the lowfield intensities

    subtracts_and_divide_by_error(a,b,c)
        calculates (a-b)/c

    residual_hf(self, params, r1, r2, hetnoe, r1_err, r2_err, noe_err, res_info)
        calculates the residual of the high field data and the model
        for leastsquares fitting

    residual_lf(self, params, low_field, resinfo, protons, intensities, intensity_errors)
        calculates the residual of the low field intensities and the model
        for leastsquares fitting    

    residual_hflf(self, params, hf_data, hf_error, res_info, low_fields, protons,intensities, intensity_errors)
        calculates a residual for both high field and low field data

    fit_single_residue(i)
        fits model free to a single residue

    per_residue_emf_fit(lf_data_prefix='', lf_data_suffix='', scale_diffusion=False, select_only_some_models=False, 
        fit_results_pickle_file='model_free_parameters.pic', cpu_count=-1)
        Fits model free to all residues

    plot_model_free_parameters( atom_name, model_pic='model_free_parameters.pic', plot_name='model_free_params.pdf', showplot=False)
        plots the model free parameters

    plot_r1_r2_noe(atom_name, model_pic='model_free_parameters.pic', fields_min=14, fields_max=25, plot_ranges=[[0,3.5], [10,35], [1, 1.9]], per_residue_prefix='hf_rates'):
        plots the highfield data and the model 

    plot_relaxometry_intensities(atom_name, protons,model_pic='model_free_parameters.pic',folder='relaxometry_intensities/')
        plots the experimental and modeled relaxometry data
    """

    def __init__(self,dx,dy,dz, mf_config='default'):

        self.r1_path = None
        self.r2_path = None 

        self.r1_err_path = None
        self.r2_err_path = None 

        self.hetnoe_path = None 
        self.hetnoe_err_path = None 

        self.sigma_path = None 
        self.sigma_err_path = None

        self.relaxometry_mono_exponencial_fits_path = None
        self.relaxometry_mono_exponencial_fits_err_path = None 

        self.path_prefix = 'model'
        self.PhysQ = PhysQ

        #fields = None
        self.diffusion_tensor = None
        self.pdb = None
        self.high_fields = None
        self.low_fields = None

        self.model_high_fields = None
        self.model_all_fields = None
        self.spectral_density = specDens.J_anisotropic_emf

        self.scale_diffusion=False, 
        self.diffusion = [dx,dy,dz]
        self.fit_results_pickle_file='model_free_parameters.pic'

        # set the file for the configurations
        if mf_config == 'default':
            self.emf_toml = pkg_resources.resource_filename('data2ensembles', 'config/extended_model_free.toml')
        else:
            self.emf_toml = mf_config

        # now read in the config
        self.emf_config = toml.load(self.emf_toml)
        
        # the structure
        self.universe = None

    def load_data(self):
        '''
        This function laods the rates and errors specified in:

        self.r1_path
        self.r2_path 
        self.hetnoe_path 
        self.r1_err_path
        self.r2_err_path 
        self.hetnoe_err_path
        '''

        rates = [self.r1_path, self.r2_path, self.hetnoe_path, 
                self.r1_err_path, self.r2_err_path, self.hetnoe_err_path]
        
        labels = ['r1', 'r2', 'hetnoe', 'r1 error', 'r2 error', 'hetnoe error']

        # check if the rates are laoded
        for i,j in zip(rates, labels):
            check = False
            if i == None:
                print('No rates loaded: please set the path for {j}')
                check=True
            
            if check == True:
                sys.exit()

        print('loading rates')
        self.r1 ,_ = utils.read_nmr_relaxation_rate(self.r1_path)
        self.r2 ,_ = utils.read_nmr_relaxation_rate(self.r2_path)
        self.hetnoe ,_ = utils.read_nmr_relaxation_rate(self.hetnoe_path)

        self.r1_err ,_ = utils.read_nmr_relaxation_rate(self.r1_err_path)
        self.r2_err ,_ = utils.read_nmr_relaxation_rate(self.r2_err_path)
        self.hetnoe_err ,_ = utils.read_nmr_relaxation_rate(self.hetnoe_err_path)

        # load the structure
        self.universe = md.Universe(self.pdb)
        
        # check = False
        # if self.relaxometry_mono_exponencial_fits_path != None:
        #     if self.relaxometry_mono_exponencial_fits_err_path !=  None:
        #         check=True


    def load_sigma(self,):

        self.sigma, _ = utils.read_nmr_relaxation_rate(self.sigma_path)
        self.sigma_err, _ = utils.read_nmr_relaxation_rate(self.sigma_err_path)

    def load_low_field_info(self, file, field_increment=0.5):
        '''
        This function loads the low field information and sets the distance increment
        for the shuttling trajectory. 

        This function defines the attributes: 
        self.ShuttleTrajectory

        Parameters
        ----------
        file : str
            file with the low field information
        
        distance_incrementing : float
            the distance increment used to calculate the shuttling trajectory

        '''
        self.ShuttleTrajectory = shuttling.ShuttleTrajectory(file, field_increment = field_increment)
        self.ShuttleTrajectory.construct_all_trajectories()

    def unpack_emf_config(self, tag, config):
        upper = config[f'{tag}_upper']
        lower = config[f'{tag}_lower']
        start = config[f'{tag}_start']
        return start, upper, lower

    def calc_cosine_angles_and_distances(self, atom_names, dist_skip=1):
        '''
        This function determines the cosine angles

        This function defines the attributes: 
        self.cosine_angles
        self.distances

        Parameters
        ----------
        atom_names : list 
            list of atom pairs 

        dist_skip : int 
            with which stride do you want to calculate the distances over the 
            MDAnalysis trajectory

        '''
        structure_analysis = trrAnalysis.AnalyseTrr(self.pdb, self.pdb, './')
        structure_analysis.load_trr()

        # this is not really needed but is used by calc_cosine_angles()
        # could make a change to that function to remvoe it from here 
        structure_analysis.atom_pair_distance_cutoff = 20e-10
        structure_analysis.calc_inter_atom_distances(atom_names, skip=1)
        structure_analysis.path_prefix = self.path_prefix

        structure_analysis.calc_cosine_angles(atom_names, 
                        calc_csa_angles=False, 
                        calc_average_structure=False,
                        reference_gro_file = self.pdb, 
                        delete_rmsf_file=True, 
                        dt = 1, 
                        use_reference=True, 
                        print_axis=True)

        structure_analysis.calc_inter_atom_distances(atom_names, skip=dist_skip)

        self.cosine_angles = structure_analysis.cosine_angles
        self.distances = structure_analysis.md_distances

    def residual_to_chi2(self,a):
        '''
        Parameters
        ---------
        a : ndarray
            converts the residual to a chisquare 
        '''
        return np.mean(a**2)

    def get_atom_info_from_tag(self, i):
        '''
        convert the atom tag to the atom name
        '''
        return utils.get_atom_info_from_tag(i)

    def get_c1p_keys(self,):
        '''
        Get all the keys in the R1 dataset that have the C1' atom 
        '''
        c1p_keys = []
        for i in self.r1:
            if "C1'" in i:
                c1p_keys.append(i)
        return c1p_keys

    def load_peakfit_file(self, file):
        array = np.genfromtxt(file, comments="#")
        delays, intensities, errors = array.T
        key = file.split('/')[-1].split('.')[0]
        return delays, intensities, errors, key

    def remove_peakfit_outs(self, files):
        f_out = []
        for f in files:
            if f.split('/')[-1] not in ('logs.out' , 'shifts.out'):
                f_out.append(f)

        return f_out


    def load_low_field_data(self, prefix):
        '''
        This function loads all the low field intensities from peakfit output files. 

        Parameters
        ----------
        prefix : str 
            prefix for the directory where the data is stored 
        '''
        
        # the delays are saved in the Shuttling class, here we will read it in again 
        # but likely not use it.  
        
        self.low_field_intensities = {}
        self.low_field_intensity_errors = {}
        self.low_field_delays = {}
        keys = self.get_c1p_keys()

        # iterate ove fields
        for f in self.ShuttleTrajectory.experiment_info:
            self.low_field_intensities[f] = {}
            self.low_field_intensity_errors[f] = {}
            self.low_field_delays[f] = {}

            # seach for peakfit files
            search = prefix + f"r1_c1p_{int(round(f,0))}T/Fits/*out"
            files = glob.glob(search)

            # read in the files and assign to the correspondin dictionary
            files = self.remove_peakfit_outs(files)
            for fi in files:
                delays, intensities, errors, key = self.load_peakfit_file(fi)
                self.low_field_intensities[f][key] = intensities
                self.low_field_intensity_errors[f][key] = errors
                #print(f'errors< ', errors)
                self.low_field_delays[f][key] = delays

    def set_low_field_errors(self, fields, errors):
        '''
        This function takes a list of fields and then changes the errors. 
        you might want to do this if you used the incorrect error when fitting the peaks 
        and dont want to re-run the whole peak fitting - its a but quick and dirty, but atleast its well 
        documented! 

        Note that this will set it for all the atom types/pairs 

        Parameters
        ----------
        fields : list 
            list of low fields 

        errors : list 
            list of errors 
        '''

        for f, err in zip(fields, errors):

            if f not in self.low_field_intensity_errors:
                print(f'Warning: Not changing the error for field: {f}')

            else:
                for key in self.low_field_intensity_errors[f]:
                    error_list = [err for _ in self.low_field_intensity_errors[f][key]]
                    self.low_field_intensity_errors[f][key] = error_list


    def read_pic_for_plotting(self,model_pic, atom_name):
        '''
        Read the pic file containing the model free parameters for plotting 
        
        Parameters
        ----------
        model_pic : str 
            path to the model free parameters in a picle file 

        atom_name : list 
            pairs of atom names 

        Returns
        -------
        models : dictionary of Results class from Lmfit
            contains the Result from fitting in Lmfit

        models_resid : dictionary
            contains models indexed by the residue number 

        sorted_keys : set, list
            keys sorted but size 

        model_resinfo : dictionary
            key is the residue id and the entry contains the atom information
        '''
        # load the model
        models = utils.load_pickle(model_pic)
        models_resid, model_resinfo = utils.dict_with_full_keys_to_resid_keys(models, atom_name)
        sorted_keys = sorted(list(models_resid.keys()))
        return models, models_resid, sorted_keys, model_resinfo

    def generate_mf_parameters(self, scale_diffusion=False, diffusion=None, model_selector=None, simple=False):

        '''
        This function takes makes parameter objects for each of the models
        '''
        dx,dy,dz = self.diffusion
        config = copy.deepcopy(self.emf_config['emf'])
        
        # models
        if simple == False:
            #varriable order S2, tau_s, Sf, tau_f
            models_state = [[True,  False, False, False],
                            [True,  True,  False, False],
                            [True,  True,  True,  False],
                            [True,  True,  True,  True],
                            [False,  False, True, True]]

        if simple == True:
            models_state = [[False, False, False, False,]]

        #do we want to fit the diffusion ? 
        params_lists = []
        if scale_diffusion == False:
            fit_diff = [False]

        elif scale_diffusion == True:
            fit_diff = [False, True]        

            if simple==True:
                fit_diff = [True]

        for i in fit_diff:
            for mod in models_state:
                params = Parameters()

                # now we add Sf 
                tag = 'Sf'
                if mod[2] == True:
                    start, upper, lower = self.unpack_emf_config(tag, config)
                    params.add(tag, value=start, min=lower, max=upper, vary=True)
                else:
                    params.add(tag, value=1, vary=False)

                #always have a S2
                tag = 'S2'
                if mod[0] == True:
                    start, upper, lower = self.unpack_emf_config(tag, config)
                    params.add(tag, value=start, min=lower, max=upper, vary=True)
                else:
                    if mod[2] != True:
                        params.add(tag, value=1, vary=False ,min= 0,max=1)
                    else:
                        params.add(tag, expr='Sf')

                # add Ss, which depends on S2 and Sf
                params.add('Ss', expr='S2/Sf', min=0, max=1)

                # add tau_s 
                tag = 'tau_s'
                if mod[1] == True:
                    start, upper, lower = self.unpack_emf_config(tag, config)
                    params.add(tag, value=start, min=lower, max=upper, vary=True)
                else:
                    params.add(tag, value=0, vary=False)

                # add tau_f 
                tag = 'tau_f'
                if mod[3] == True:
                    start, upper, lower = self.unpack_emf_config(tag, config)
                    params.add(tag, value=start, min=lower, max=upper, vary=True)
                else:
                    params.add(tag, value=0, vary=False)

                if diffusion == None:
                    params.add('dx_fix',  min=0, value=dx, vary=False)
                    params.add('dy_fix',  min=0, value=dy, vary=False)
                    params.add('dz_fix',  min=0, value=dz, vary=False)
                    params.add('diff_scale', min=0, value=1, vary=i)

                else:
                    params.add('dx_fix',  min=0, value=diffusion[0], vary=False)
                    params.add('dy_fix',  min=0, value=diffusion[1], vary=False)
                    params.add('dz_fix',  min=0, value=diffusion[2], vary=False)
                    params.add('diff_scale', min=0, value=1, vary=i)

                params.add('dx',  expr='dx_fix*diff_scale')
                params.add('dy',  expr='dy_fix*diff_scale')
                params.add('dz',  expr='dz_fix*diff_scale')
                params_lists.append(params)

        # for i in params_lists[0]:
        #     print(i, params_lists[0][i])
        return params_lists


    def model_hf(self, params,res_info, fields):
        '''
        Model for fitting high field R1, R2 and hetro nuclear NOE

        Parameters
        ----------
        params : lmfit parameters class, dictionary
            contains the parameters for the selected spectral density class 

        res_info : list 
            list containing the residue information

        fields : ndarray 
            contains the fields for the data stored in the R1,R2, and HetNOE files. 

        Returns
        -------
        model_r1 : ndarray 
            array of model R1 values
        model_r2 : ndarray 
            array of model R2 values
        model_noe : ndarray 
            array of model NOE values
        '''

        resid, resname, atom1, atom2 = res_info
        rxy = self.PhysQ.bondlengths[atom1, atom2]
        csa_atom_name = (atom1, resname)
        x = 'c'
        angles = self.cosine_angles[(resid, atom1 , resid, atom2)]

        model_r1 = d2e.rates.r1_YX(params, self.spectral_density, fields, rxy, 
            csa_atom_name, x, y='h', cosine_angles=angles, model='axially symmetric')
        
        model_r2 = d2e.rates.r2_YX(params, self.spectral_density, fields, rxy, 
            csa_atom_name, x, y='h', cosine_angles=angles, model='axially symmetric')

        model_noe = d2e.rates.noe_YX(params, self.spectral_density, fields, rxy, 
            x, model_r1, cosine_angles=angles)

        return model_r1, model_r2, model_noe

    def model_sigma(self, params,res_info, fields):
        '''
        Model for sigma (reduced NOE) NOTE: Here we have hard coded the C-H pair

        Parameters
        ----------
        params : lmfit parameters class, dictionary
            contains the parameters for the selected spectral density class 

        res_info : list 
            list containing the residue information

        fields : ndarray 
            contains the fields for the data stored in the R1,R2, and HetNOE files. 

        Returns
        -------
        model_sigma : ndarray 
            array of model sigma values

        '''

        resid, resname, atom1, atom2 = res_info
        rxy = self.PhysQ.bondlengths[atom1, atom2]
        csa_atom_name = (atom1, resname)
        x = 'c'
        angles = self.cosine_angles[(resid, atom1 , resid, atom2)]    

        model_sigma = d2e.rates.r1_reduced_noe_YX(params, self.spectral_density, 
            fields, rxy, x, y='h', cosine_angles=angles,)

        return model_sigma


    def model_lf_single_field_intensities(self, params, low_field, res_info, protons):
        '''
        Model for fitting low field intensities

        Parameters
        ----------
        params : lmfit parameters class, dictionary
            contains the parameters for the selected spectral density class 

        low_field : float
            the low field we are simulating at

        res_info : list 
            list containing the residue information

        protons : list 
            list of protons we want to consider in our interactions 

        Returns
        -------
        intensities : ndarray 
            list of intensities at each delay 

        '''
        time_dict = {}

        resid, restype, x_spin, y_spin = res_info
        traj = self.ShuttleTrajectory.trajectory_time_at_fields[low_field]
        
        forwards_field, forwards_time, backwards_field, backwards_time = traj 
        operator_size = 2 + len(protons)*2

        # forwards 
        relaxation_matricies_forwards = relax_mat.relaxation_matrix_emf_c1p(params,
                resid, 
                self.spectral_density, 
                forwards_field, 
                self.distances, 
                self.cosine_angles, 
                restype,
                operator_size, 
                x_spin,
                y_spin,
                protons, 
                x = 'c',
                y = 'h',)

        # backwards 
        relaxation_matricies_backwards = relax_mat.relaxation_matrix_emf_c1p(params,
                resid, 
                self.spectral_density, 
                backwards_field, 
                self.distances, 
                self.cosine_angles, 
                restype,
                operator_size, 
                x_spin,
                y_spin,
                protons, 
                x = 'c',
                y = 'h',)

        #stabalisation 
        relaxation_matricies_stablaization_delay = relax_mat.relaxation_matrix_emf_c1p(params,
                resid, 
                self.spectral_density, 
                np.array([self.ShuttleTrajectory.experiment_info[low_field]['high_field']]), 
                self.distances, 
                self.cosine_angles, 
                restype,
                operator_size, 
                x_spin,
                y_spin,
                protons, 
                x = 'c',
                y = 'h',)

        forwrds_propergators = mathFunc.construct_operator(relaxation_matricies_forwards, forwards_time)
        
        backwards_operators = mathFunc.construct_operator(relaxation_matricies_backwards, backwards_time)         
        stablaization_operator = mathFunc.construct_operator(relaxation_matricies_stablaization_delay,
            np.array([self.ShuttleTrajectory.experiment_info[low_field]['stabalisation_delay']]), product=False)
        
        stablaization_operator = np.array(stablaization_operator)[0]

        #this is the matrix that evolves with the varriable delay
        low_field_relaxation_matrix = relax_mat.relaxation_matrix_emf_c1p(params,
                resid, 
                self.spectral_density, 
                np.array([low_field]), 
                self.distances, 
                self.cosine_angles, 
                restype,
                operator_size, 
                x_spin,
                y_spin,
                protons, 
                x = 'c',
                y = 'h',)

        #str_mat = np.array2string(low_field_relaxation_matrix.T[0], precision=3, separator='\t',suppress_small=False) 

        # code to debug / print a matrix
        # print(low_field)
        # print(str_mat)
        # sys.exit()

        delay_propergators = mathFunc.construct_operator(low_field_relaxation_matrix, 
                                self.ShuttleTrajectory.experiment_info[low_field]['delays'], product=False)

        # for i in range(100):
        #     #mathFunc.construct_operator(low_field_relaxation_matrix, 
        #     #                    self.ShuttleTrajectory.experiment_info[low_field]['delays'], product=False)

            

        #     roll = np.rollaxis(low_field_relaxation_matrix, 2)
        #     mathFunc.matrix_exp(roll[0])

        # this could probably be shortened by pre-caclulating 
        # stablaization_operator*backwards_operators
        full_propergators = []
        for i in delay_propergators:
            operators = [stablaization_operator, backwards_operators, i, forwrds_propergators]
            i = np.linalg.multi_dot(operators)
            full_propergators.append(i)


        # full_propergators = [stablaization_operator.dot(backwards_operators).dot(i).dot(forwrds_propergators) for i in delay_propergators]
        
        experimets = np.zeros(operator_size)
        experimets[1] = 1.
        #print(np.array_str(full_propergators[0].dot(experimets), precision=2, suppress_small=True))
        #print(len(full_propergators), full_propergators[0].shape)

        intensities = [np.matmul(i, experimets) for i in full_propergators]
        return intensities

    def subtracts_and_divide_by_error(self, a,b,c):
        '''
        Parameters
        ----------
        a : float
        b : float 
        c : float 
        '''
        return (a-b)/c

    def residual_hf(self, params, r1, r2, hetnoe, r1_err, r2_err, noe_err, res_info):
        '''
        calculates the residuals for the high field experiments: R1, R2, hetNOE

        Parameters
        ----------
        params : lmfit parameters class, dictionary
            contains the parameters for the selected spectral density class 

        r1 : ndarray 
            experimental R1 values

        r2 : ndarray 
            experimental r2 values 

        hetnoe : ndarray 
            experimental hetNOE values 

        r1_err : ndarray 
            experimental R1 errors

        r2_err : ndarray 
            experimental r2 errors 

        hetnoe_err : ndarray 
            experimental hetNOE errors 

        res_info : list 
            list containing the residue information

        Returns 
        -------
        ndarray : residuals
        '''
        
        model_r1, model_r2, model_noe = self.model_hf(params,res_info, self.high_fields)
        
        r1_diff = self.subtracts_and_divide_by_error(r1, model_r1,r1_err)
        r2_diff = self.subtracts_and_divide_by_error(r2, model_r2,r2_err)
        noe_diff = self.subtracts_and_divide_by_error(hetnoe, model_noe,noe_err)
        constraint = 0

        if params['tau_s'] < params['tau_f']:
            constraint = (params['tau_s'] - params['tau_f'])*1e12

        return np.concatenate([r1_diff.flatten(), r2_diff.flatten(), noe_diff.flatten(), [constraint]])

    def residual_hf_diffusion(self, params, r1, r2, hetnoe, r1_err, r2_err, noe_err, res_info):
        '''
        calculates the residuals for the high field experiments: R1, R2, hetNOE

        Parameters
        ----------
        params : lmfit parameters class, dictionary
            contains the parameters for the selected spectral density class 

        r1 : ndarray 
            experimental R1 values

        r2 : ndarray 
            experimental r2 values 

        hetnoe : ndarray 
            experimental hetNOE values 

        r1_err : ndarray 
            experimental R1 errors

        r2_err : ndarray 
            experimental r2 errors 

        hetnoe_err : ndarray 
            experimental hetNOE errors 

        res_info : list 
            list containing the residue information

        Returns 
        -------
        ndarray : residuals
        '''
        
        model_r1, model_r2, model_noe = self.model_hf(params,res_info, self.high_fields)
        
        diff_r1  = self.subtracts_and_divide_by_error(r1, model_r1,r1_err)
        diff_r2  = self.subtracts_and_divide_by_error(r2, model_r2,r2_err)
        diff_noe = self.subtracts_and_divide_by_error(hetnoe, model_noe,noe_err)
        model_ratio = model_r1/model_r2
        exp_ratio = r1/r2
        exp_error = r1_err/r2_err

        diff = self.subtracts_and_divide_by_error(exp_ratio, model_ratio,exp_error)
        constraint = 0

        if params['tau_s'] < params['tau_f']:
            constraint = (params['tau_s'] - params['tau_f'])*1e12

        return diff
        #return np.array([diff_r1, diff_r2])

    def residual_lf(self, params, low_field, resinfo, protons, intensities, intensity_errors):

        '''
        calculates the residuals for the relaxometry inensities

        Parameters
        ----------
        params : lmfit parameters class, dictionary
            contains the parameters for the selected spectral density class 

        low_field : float
            the low field we are simulating at

        res_info : list 
            list containing the residue information

        protons : list 
            list of protons we want to consider in our interactions 

        intensities : ndarray 
            experimental intensities 

        intensities_error s: ndarray 
            experimental intensity errors

        Returns 
        -------
        diffs : ndarray
            residuals between experimental and modeled errors
        '''

        diffs = []
        if type(params) is not dict:
            params = params.valuesdict()
        
        for field, exp_intensity, exp_error in zip(low_field, intensities, intensity_errors):
            
            intensities = self.model_lf_single_field_intensities(params, field, resinfo, protons)
            # the model returns back all the populations 
            intensities = np.array([i[1] for i in intensities])
            exp_error = np.array(exp_error)
            len_error = len(exp_error)
            exp_intensity = np.array(exp_intensity)
            # here we need a prefactor to scale our calculated intensities since we do not 
            # know what the zero point is. 

            factor = np.mean(exp_intensity)/np.mean(intensities)
            diff = (factor*intensities - exp_intensity)/(exp_error*len_error)
            diffs = diffs + diff.tolist()

        return diffs

    def residual_hflf(self, params, hf_data, hf_error, res_info, low_fields, protons,intensities, intensity_errors):
        '''
        calculates the residuals for highfield data and low field data

        Parameters
        ----------
        params : lmfit parameters class, dictionary
            contains the parameters for the selected spectral density class 

        hf_data : tupple 
            contains ndarrays for the high field data 

        hf_error : tupple 
            contains ndarrays for the high field errors 

        res_info : list 
            list containing the residue information      

        low_fields : list
            list of the low fields

        protons : list 
            list of protons we want to consider in our interactions 

        intensities : list 
            experimental intensities for each low field 

        intensities_error s: list 
            experimental intensity errors for each low field

        Returns 
        -------
        diffs : ndarray
            residuals between experimental and modeled errors

        '''
        # high field data
        
        r1, r2, hetnoe = hf_data
        r1_err, r2_err, noe_err = hf_error

        # for i in range(50):
        hf_residual = self.residual_hf(params, r1, r2, hetnoe, r1_err, r2_err, noe_err, res_info)
        #return hf_residual
        
        lf_residual = self.residual_lf(params, low_fields, res_info, protons, intensities, intensity_errors)
        return np.concatenate([hf_residual, lf_residual])

    def chi_square(self, residual_func, resid_args):
        '''
        This function returns the chi square statistic from the residuals
        '''
        diffs = residual_func(*resid_args)
        squares = np.square(diffs)
        return np.sum(squares)

    def residual_selector(self, i, residual_type, res_info):

        if 'hf' in residual_type:
            hf_data = (self.r1[i], self.r2[i], self.hetnoe[i])
            hf_errors = (self.r1_err[i], self.r2_err[i], self.hetnoe_err[i])

        if 'lf' in residual_type:
            protons = ("H1'", "H2'", "H2''")
            low_fields = list(self.ShuttleTrajectory.experiment_info.keys())
            intensities = []
            intensity_errors = []

            for f in low_fields:
                intensities.append(self.low_field_intensities[f][i])
                intensity_errors.append(self.low_field_intensity_errors[f][i])

        if residual_type == 'hflf':
            # do fit, here with the default leastsq algorithm
            fcn_args = (hf_data, hf_errors, res_info, low_fields, 
                        protons,intensities, intensity_errors)
            
            residual_func = self.residual_hflf

            #minner = Minimizer(self.residual_hflf, params, fcn_args=())

        elif residual_type == 'hf':
            # do fit, here with the default leastsq algorithm
            fcn_args = (*hf_data, *hf_errors, res_info)
            residual_func = self.residual_hf
            #minner = Minimizer(self.residual_hf, params, fcn_args=(fcn_args))

        elif residual_type == 'hf_r2/r1':
            # do fit, here with the default leastsq algorithm
            fcn_args = (*hf_data, *hf_errors, res_info)
            residual_func = self.residual_hf_diffusion
            #minner = Minimizer(self.residual_hf_diffusion, params, fcn_args=(fcn_args))

        elif residual_type == 'lf':
            print('This has not been implimented yet ...')
            os.exit()

        else:
            print('This residual_type is not recognised.')

        return fcn_args, residual_func

    def fit_single_residue(self, i, provided_diffusion=None, residual_type='hflf', model_select=True, minimistion_method='powell'):
        '''
        This function fits all the models available in model free (set by generate_mf_parameters())
        and selects the best one based on the lowest BIC. 

        Paramteres
        ----------
        i : str
            the residue tag
        
        provided_diffusion : bool, list, array
            this argument is used to provide an alternative diffusion tensor to 
            self.diffusion. If False self.diffusion is used otherwise a list like object 
            [dx,dy,dz] is used. 

        residual_type : str
            determines which residual type is used. The residual can calculate only low field data
            'lf', only high field data 'hf' or both 'hflf'.  

        model_select : bool 
            If True then the best model is selected based on the bic. If False all models are returned

        '''
        # print(f'fitting {i} with method {minimistion_method}')
        # get the atom info
        res_info = utils.get_atom_info_from_tag(i)
        resid, resname, atom1, atom2 = res_info
        all_models = []

        # probably dont need a dictionary for the bic any more 
        bics = {} 
        bics[i] = 10e1000

        # the rows are Sf, Ss, tauf, taus, might be best to take this out and put it in its own function
        # if provided_diffusion == False:
        #     dx,dy,dz = self.diffusion
        # else:
        #     dx,dy,dz = provided_diffusion

        if residual_type=='hf_r2/r1':
            simple = True
        else:
            simple = False

        params_models = self.generate_mf_parameters(scale_diffusion=self.scale_diffusion, 
            diffusion=provided_diffusion, simple=simple)
        
        # probably can do this logic a little better
        for params in params_models:

            resid, resname, atom1, atom2 = res_info
            key = (resid, atom1 , resid, atom2)
            # args = (np.array([23,16])*self.PhysQ.gamma['c'],self.cosine_angles[key][0], self.cosine_angles[key][1], self.cosine_angles[key][2], )

            fcn_args, residual_func = self.residual_selector(i, residual_type, res_info)
            start_time = time.time()
            minner = Minimizer(residual_func, params, fcn_args=(fcn_args))
            result = minner.minimize(method=minimistion_method)
            resdict = result.params.valuesdict()
            end_time = time.time()
            elapsed_time = end_time - start_time
            all_models.append(result)

            if result.bic < bics[i]:
                model = result
                bics[i] = result.bic

        #report_fit(model)
        if model_select == True:
            return (i, model)

        if model_select == False:
            return (i, all_models)

    def wrapper_fit_single_residue(self, args):
        args, kwargs = args
        #print('args', args)
        #print('kwargs', kwargs)
        res = self.fit_single_residue(args,**kwargs)
        return res


    def per_residue_emf_fit(self, 
        lf_data_prefix='', 
        lf_data_suffix='', 
        select_only_some_models=False, 
        cpu_count=-1, 
        residual_type='hflf',
        provided_diffusion=None, 
        writeout=True, 
        minimistion_method='powell'):

        '''
        This funtion fits a spectral density function to each residue using the
        extended model free formalism. Since Each residue is treated individually
        the diffusion tensor is held as fixed. 
        '''

        data = {}
        models = {}

        if cpu_count <= 0 :
            cpu_count = mp.cpu_count()

        args = self.get_c1p_keys()
        kwargs = {}
        kwargs['residual_type'] = residual_type
        kwargs['minimistion_method'] = minimistion_method

        #allows a provided diffution tensor to be passed to the fitting. 
        if provided_diffusion != None:
            kwargs['provided_diffusion'] = provided_diffusion
        
        total_args = [(i, kwargs) for i in args]

        start_time = time.time()
        if cpu_count == 1:
            for i in total_args:
                res = self.wrapper_fit_single_residue(i)
                models[res[0]] = res[1] 
        else:

            with mp.Pool(cpu_count) as pool:
                results = pool.map(self.wrapper_fit_single_residue, total_args)

            for i in results:
                models[i[0]] = i[1]

        end_time = time.time()
        elapsed_time = end_time - start_time
        print("extended model free fitting Elapsed time: ", elapsed_time) 
        
        # save the file
        if writeout == True:
            with open(self.fit_results_pickle_file, 'wb') as handle:
                pic.dump(models, handle)

        return models


    def wrapper_global_residual(self, args):
        i, current_diffusion, residual_type = args
        
        # could probably re-write this with *args and **kwargs if needed
        res = self.fit_single_residue(i, 
            provided_diffusion=current_diffusion, 
            residual_type=residual_type, 
            model_select=False)
        
        return res

    def global_emf_fit(self, 
        lf_data_prefix='', 
        lf_data_suffix='', 
        select_only_some_models=False, 
        cpu_count=-1, 
        residual_type='hf', 
        diffusion_type='linear'):

        '''
        This funtion fits a spectral density function to each residue using the
        extended model free formalism. Since Each residue is treated individually
        the diffusion tensor is held as fixed. 

        WARNING: this somehow is not working well at the moment ...
        '''

        def global_residual(params, cpu_count):

            # diffusion tensor
            dx = params['dx'].value
            dy = params['dy'].value
            dz = params['dz'].value
            current_diffusion = (dx,dy,dz)

            #residue keys
            c1p_keys = self.get_c1p_keys()
            
            # args for the wrapper 
            args = [[i, current_diffusion, residual_type] for i in c1p_keys]

            #store info
            data = {}
            models = {}
            resids = []

            #parallel part
            if cpu_count <= 0 :
                cpu_count = mp.cpu_count()

            # do the calculation
            start_time = time.time()
            if cpu_count == 1:
                for i in c1p_keys:
                    res = self.wrapper_global_residual(args)
                    models[res[0]] = res[1] 
            else:

                with mp.Pool(cpu_count) as pool:
                    results = pool.map(self.wrapper_global_residual, args)

                for i in results:
                    models[i[0]] = i[1]

            # blend the residuals for the models based on the aics
            for i in models:
                # get all the AICS
                #aics = [mod.aic for mod in models[i]]
                #min_aic = min(aics)

                # simpler seleciton of the models - is it better ? Not sure
                best_aic = 10e10
                for mod in models[i]:
                    # get all the AICS
                    if mod.bic < best_aic:
                        best_aic = mod.bic
                        current_resids = mod.residual

                resids.append(current_resids)
                    
                    # probability_of_min_info_loss = np.e**((min_aic-mod.aic)/2)
                    # weighted_residuals = probability_of_min_info_loss*mod.residual
                    # resids.append(weighted_residuals)

            return np.array(resids)

        # sanity check 
        if self.scale_diffusion==True:
            print('This will not work with self.scale_diffusion=True')
            print('because we update the diffusion tensor in the outer and inner fits')

        # set up the Parameters object
        params = Parameters()

        if diffusion_type == 'linear':
            params.add('dx_fix',  min=0, value=self.diffusion[0], vary=False)
            params.add('dy_fix',  min=0, value=self.diffusion[1], vary=False)
            params.add('dz_fix',  min=0, value=self.diffusion[2], vary=False)


            #scale the diffusion tensor by a single value 
            params.add('diff_scale', min=0, value=1, vary=True)
            params.add('dx',  expr='dx_fix*diff_scale')
            params.add('dy',  expr='dy_fix*diff_scale')
            params.add('dz',  expr='dz_fix*diff_scale')

        elif diffusion_type == 'axially_symetric':

            dperp = (self.diffusion[0] + self.diffusion[1])/2
            params.add('dx_fix',  min=0, value=dperp, vary=False)
            params.add('dy_fix',  expr='dx_fix', vary=False)
            params.add('dz_fix',  min=0, value=self.diffusion[2], vary=False)


            #scale the diffusion tensor by a single value 
            params.add('diff_scale_long', min=0, value=1, vary=True)
            params.add('diff_scale_short', min=0, value=1, vary=True)
            params.add('dx',  expr='dx_fix*diff_scale_short')
            params.add('dy',  expr='dy_fix*diff_scale_short')
            params.add('dz',  expr='dz_fix*diff_scale_long')

        else:
            print('Params object for fitting has not been set correctly.')
            print('check the diffusion_type kwarg')
            sys.exit()

        args = [cpu_count]
        minner = Minimizer(global_residual, params, fcn_args=args)
        result = minner.minimize(method='powel')
        report_fit(result)

    def scale_hydro_nmr_diffusion_tensor(self, 
        lf_data_prefix='', 
        lf_data_suffix='', 
        select_only_some_models=False, 
        cpu_count=-1, 
        residual_type='hf', 
        diffusion_type='linear'):

        '''
        This function scales the hydroNMR diffusion tensor to NMR data using an approach similar
        https://sci-hub.hkvisa.net/10.1126/science.7754375
        Long-Range Motional Restrictions in a Multi domain Zinc-Finger Protein from Anisotropic Tumbling

        and 

        Rotational diffusion anisotropy of proteins from simultaneous analysis of 15N and 13Cα nuclear spin relaxation
        '''

        def q_model(cosine_angles, q):
            cosine_angles = np.array(cosine_angles)
            part_1 = np.matmul(q, cosine_angles)
            part_2 = np.matmul(cosine_angles, part_1)
            return part_2

        def resid(params, diso):

            # set up the matrix q
            dx = params['dx']
            dy = params['dy']
            dz = params['dz']

            qx = (dy + dz)/2
            qy = (dx + dz)/2
            qz = (dx + dy)/2

            q = np.zeros([3,3])
            q[0][0] = qx
            q[1][1] = qy
            q[2][2] = qz

            # determine the diffusions
            diffs = []
            for i in diso:
                res_info = utils.get_atom_info_from_tag(i)
                resid, resname, atom1, atom2 = res_info
                key = (resid, atom1 , resid, atom2)

                cosine_angles = self.cosine_angles[key]
                diffs.append(diso[i] - q_model(cosine_angles, q))

            return np.array(diffs)

        #step one is to get all local tau_c 
        # this will give us an isotropic model
        diffusion = [1e7, 1e7, 1e7]

        original_value = copy.copy(self.scale_diffusion)
        print(f'Setting self.scale_diffusion = True, from {original_value}')
        
        self.scale_diffusion = True
        models = self.per_residue_emf_fit(residual_type=diffusion_type,
            provided_diffusion=diffusion, writeout=False)

        print(f'Setting self.scale_diffusion back to: {original_value}')
        self.scale_diffusion = original_value
        print(f'value now: {self.scale_diffusion}')

        # get Di
        diso_local = {}
        for i in models:
            #check it was isotropic 
            assert models[i].params['dx'] == models[i].params['dy'] , "dx != dy"
            assert models[i].params['dx'] == models[i].params['dz'] , "dx != dz"
            diso_local[i] = np.mean([models[i].params[j] for j in ('dx','dy','dz')])

        #get moment of inertia
        moment_of_inertia = []
        for ts in self.universe.trajectory:
            moment_of_inertia.append(self.universe.select_atoms('all').moment_of_inertia())

        # I dont think I need to transform my cosine angles if they are the angles 
        # between the principle axis ... 

        # now we set up the parameters for the diffusion calculation
        params = Parameters()
        params.add('dx_fix', value=self.diffusion[0], vary=False)
        params.add('dy_fix', value=self.diffusion[1], vary=False)
        params.add('dz_fix', value=self.diffusion[2], vary=False)


        #scale the diffusion tensor by a single value 
        params.add('diff_scale', min=0, value=1, vary=True)
        params.add('dx',  expr='dx_fix*diff_scale')
        params.add('dy',  expr='dy_fix*diff_scale')
        params.add('dz',  expr='dz_fix*diff_scale')

        minner = Minimizer(resid,params, fcn_args=[diso_local])
        result = minner.minimize(method='powel')
        report_fit(result)

        inputs = []
        for i in diso_local:
            res_info = utils.get_atom_info_from_tag(i)
            resid_, resname, atom1, atom2 = res_info
            key = (resid_, atom1 , resid_, atom2)

            cosine_angles = self.cosine_angles[key]
            inputs.append(diso_local[i])

        inputs = np.array(inputs)

        plt.title('local tauc and model')
        plt.plot(inputs, label='data')
        plt.plot(inputs+result.residual, label='model')
        plt.xlabel('residue')
        plt.ylabel('$D$')
        plt.legend()
        plt.savefig('diffusion_model_vs_data.pdf')
        plt.show()

        res_dict = result.params.valuesdict()
        diso = np.mean([res_dict['dx'], res_dict['dy'], res_dict['dz']])
        print(f'Diso = {diso}')
        print(f'tauc iso = {1/(6*diso)}')

        # now calculate chi square surface
        scale_axis = np.linspace(-1.2, 1.2, 300) + res_dict['diff_scale']
        chi_squares = []
        
        for i in scale_axis:
            params = {'dx' : self.diffusion[0] * i, 
                      'dy' : self.diffusion[1] * i, 
                      'dz' : self.diffusion[2] * i}

            residuals = resid(params, diso_local)
            chi_squares.append(np.sum(residuals**2))

        a = res_dict['diff_scale']
        plt.title(f'best diffusion scalar {a}')
        plt.plot(scale_axis, chi_squares)
        plt.xlabel('diffusion scalar')
        plt.ylabel('chi square value')
        plt.savefig('diffusion_chi_square.pdf')
        plt.show()


    def global_connected_emf_fit(self, atom_name, 
        cpu_count=-1, 
        residual_type='hflf',
        provided_diffusion=None, 
        writeout=True, 
        model_pic='model_free_parameters.pic'):

        '''
        Here I do a global fit starting from some parameters from a local fit. 
        '''

        

        # load the previous models
        models, models_resid, sorted_keys, model_resinfo = self.read_pic_for_plotting(model_pic, atom_name)

        #build the parameters object
        params = Parameters()

        varriables = ['S2', 'Sf', 'Ss', 'tau_s', 'tau_f']

        #iterate over the data points, could maybe change this loop to give tag directly
        for i in sorted_keys:
            res_info = model_resinfo[i]
            resid, resname, atom1, atom2 = res_info
            tag = utils.resinto_to_tag(*model_resinfo[i])

            current_params = models[tag].params
            for var in varriables:
                current_params_var = current_params[var]
                var_name = f'{var}_res_{resid}'

                params_kwargs = {}
                params_kwargs['value'] = current_params_var.value
                params_kwargs['vary'] =  current_params_var.vary
                params_kwargs['min'] =  current_params_var.min
                params_kwargs['max'] =  current_params_var.max
                
                if current_params_var.expr != None:

                    current_str = current_params_var.expr
                    current_str = current_str.replace('S2', f'S2_res_{resid}')
                    current_str = current_str.replace('Sf', f'Sf_res_{resid}')
                    print(current_str)

                    params_kwargs['expr'] = current_str
                
                params.add(var_name, **params_kwargs)

        # now add the diffusion values
        params.add('dx_fix', value=self.diffusion[0], vary=False)
        params.add('dy_fix', value=self.diffusion[1], vary=False)
        params.add('dz_fix', value=self.diffusion[2], vary=False)


        #scale the diffusion tensor by a single value 
        params.add('diff_scale', min=0, value=1, vary=True)
        params.add('dx',  expr='dx_fix*diff_scale')
        params.add('dy',  expr='dy_fix*diff_scale')
        params.add('dz',  expr='dz_fix*diff_scale')

        print(f'In total there are {len(params.keys())} parameters')

        return models


    def get_mono_exponencial_fits(self, 
        atom_name, 
        protons,
        model_pic='model_free_parameters.pic', 
        plot=False, 
        plots_directory='monoexp_fits/'):
        '''
        This function fits monoexponencial decays to intensities 
        for the relaxometry data, experimental and calculated
        '''

        def mono_exp_model(params, time):

            return params['a']*np.e**(-1*params['b']*time)

        def residual(params, time, data, errors):
            # time = np.array(time)
            # data = np.array(data)
            # errors = np.array(errors)
            return (data - mono_exp_model(params,time))/errors

        def do_fit(time, data, errors):
            '''
            does the fit
            '''
            params = Parameters()
            params.add('a', value=np.mean(data))
            params.add('b', value=1)
            minner = Minimizer(residual, params, fcn_args=(time, data, errors))
            result = minner.minimize()
            return result

        os.makedirs(plots_directory, exist_ok=True)

        # load the model
        models, models_resid, sorted_keys, model_resinfo = self.read_pic_for_plotting(model_pic, atom_name)

        # define attributes where we store the data
        self.relaxometry_mono_exp_fits = {}
        self.relaxometry_mono_exp_fits_err = {}

        self.relaxometry_mono_calc_fits = {}
        self.relaxometry_mono_calc_fits_err = {}

        #iterate over the data points, could maybe change this loop to give tag directly
        for i in sorted_keys:
            res_info = model_resinfo[i]
            tag = utils.resinto_to_tag(*model_resinfo[i])
            resid, resname, atom1, atom2 = res_info

            self.relaxometry_mono_exp_fits[tag] = []
            self.relaxometry_mono_exp_fits_err[tag] = []

            self.relaxometry_mono_calc_fits[tag] = []
            self.relaxometry_mono_calc_fits_err[tag] = []

            for f in self.low_fields:

                # the experimental fit
                intensities = self.low_field_intensities[f][tag]
                intensity_errors = self.low_field_intensity_errors[f][tag]
                delays =  self.ShuttleTrajectory.experiment_info[f]['delays']
                exp_result = do_fit(delays, intensities, intensity_errors)

                # the fit of the calculated intensities, and pull out the coherence we want
                calc_intensities = self.model_lf_single_field_intensities(models[tag].params, f, res_info, protons)
                calc_intensities = np.array([j[1] for j in calc_intensities])
                errors = [1 for i in intensities]
                calc_result = do_fit(delays, calc_intensities, errors)

                sim_x = np.linspace(min(delays), max(delays), 100)
                
                if plot == True:
                    rate = exp_result.params['b'].value
                    exp_fit = mono_exp_model(exp_result.params, sim_x)
                    plt.scatter(delays, intensities, label=f'experimenal points R: {rate:0.2f}', c='C1')
                    plt.plot(sim_x, exp_fit, c='C1')

                    calc_fit = mono_exp_model(calc_result.params, sim_x)
                    # scalling
                    rate = calc_result.params['b'].value
                    scaling_factor = np.mean(intensities)/np.mean(calc_intensities)
                    calc_intensities_temp = calc_intensities * scaling_factor
                    calc_fit_temp = calc_fit * scaling_factor
                    plt.scatter(delays, calc_intensities_temp, label=f'calc points R:{rate:0.2f}', c='C2')
                    plt.plot(sim_x, calc_fit_temp, c='C2')
                    plt.legend()
                    plt.savefig(f'{plots_directory}check_{tag}_{f}.pdf')
                    plt.close()

                # save the results
                self.relaxometry_mono_exp_fits[tag].append(exp_result.params['b'].value)
                self.relaxometry_mono_exp_fits_err[tag].append(exp_result.params['b'].stderr)

                self.relaxometry_mono_calc_fits[tag].append(calc_result.params['b'].value)
                self.relaxometry_mono_calc_fits_err[tag].append(calc_result.params['b'].stderr)

    def chi_square_surface_resid_wrapper(self, 
        param1_value, 
        param2_value, 
        param1_name, 
        param2_name, 
        params, 
        resid_args, 
        resid_func,):

        params = copy.copy(params.valuesdict())
        params[param1_name] = param1_value
        params[param2_name] = param2_value

        resid_args = [params, *resid_args]
        chi_square = self.chi_square(resid_func, resid_args)
        return chi_square

    def chi_square_likelyhood(self, 
        param1_value, 
        param2_value, 
        param1_name, 
        param2_name, 
        params, 
        resid_args, 
        resid_func,
        chi_0,):

        params = copy.copy(params.valuesdict())
        params[param1_name] = param1_value
        params[param2_name] = param2_value

        resid_args = [params, *resid_args]
        chi_square = self.chi_square(resid_func, resid_args)
        likelyhood = np.e**(-(chi_square-chi_0)/2)
        return likelyhood

    def plot_chi_likelyhood_surfs(self, 
        atom_name, 
        protons,
        residual_type='hf', 
        model_pic='model_free_parameters.pic', 
        plots_directory='chi_square_surf/', 
        sampler_resolution=200, 
        surface_resolution=50, 
        init_grid=10):
        '''
        This function fits monoexponencial decays to intensities 
        for the relaxometry data, experimental and calculated
        '''

        def apply_log(tag):
            if 'tau' in tag:
                return 'log'
            else:
                return 'linear'

        def adpative_sampler_metric(x,y,z_x,z_y,dim1_bounds, dim2_bounds,dim1_scale, dim2_scale):

            # dim_1_co_ords = np.array([x[0], y[0]])
            # dim_2_co_ords = np.array([x[1], y[1]])

            # #rescale 
            # if dim1_scale == 'log':
            #     dim_1_co_ords = np.log(dim_1_co_ords)
           
            # if dim2_scale == 'log':
            #     dim_2_co_ords = np.log(dim_2_co_ords)

            # # get the difference
            # dim1_diff = dim_1_co_ords[0] - dim_1_co_ords[1]
            # dim2_diff = dim_2_co_ords[0] - dim_2_co_ords[1]

            # # divide by the range
            # dim1_diff = dim1_diff/(dim1_bounds[0]-dim1_bounds[1])
            # dim2_diff = dim2_diff/(dim2_bounds[0]-dim2_bounds[1])

            # dim1_diff = dim1_diff**2
            # dim2_diff = dim2_diff**2

            # #final distance we use 
            # distance_measure = np.sqrt(dim1_diff + dim2_diff)

            scalar = np.array([dim1_bounds[0]-dim1_bounds[1], dim2_bounds[0]-dim2_bounds[1]])
            scalar = np.absolute(scalar)
            # print(scalar)

            x = x/scalar
            y = y/scalar

            distance_measure = np.linalg.norm(x-y)
            #esitmate the derivative 
            derivative = z_x - z_y

            return abs(distance_measure * derivative)

        def adpative_sampler_transform(points, dim1_bounds, dim2_bounds):

            new_points = []

            scalar = np.array([dim1_bounds[0]-dim1_bounds[1], dim2_bounds[0]-dim2_bounds[1]])
            scalar = np.absolute(scalar)*0.1
            minimum =  np.array([np.min(dim1_bounds), np.min(dim2_bounds)])
            for i in points:
                new_points.append((i-minimum)/scalar)

            return new_points

        def get_stderr_bounds(best, params, tag):

            # the smallest value
            std = params[tag].stderr
            minimum = best - std*2
            maximum = best + std*2

            if minimum < params[tag].min:
                minimum = params[tag].min
            
            if maximum > params[tag].max:
                maximum = params[tag].max

            return minimum*0.9, maximum*1.1

        os.makedirs(plots_directory, exist_ok=True)

        # load the model
        models, models_resid, sorted_keys, model_resinfo = self.read_pic_for_plotting(model_pic, atom_name)

        #iterate over the data points, could maybe change this loop to give tag directly
        for i in sorted_keys:
            res_info = model_resinfo[i]
            tag = utils.resinto_to_tag(*model_resinfo[i])
            resid, resname, atom1, atom2 = res_info

            current_param = models[tag].params
            varriables = []
            for var in current_param:
                if current_param[var].vary ==True:
                    varriables.append(var)

            # get the args for the residual function
            residual_args, residual_func = self.residual_selector(tag, residual_type, res_info)
            
            # calculate chi minimum 
            # chi_square_args = (param1_tag, param2_tag, current_param, residual_args, residual_func)
            resid_args_with_params = [current_param, *residual_args]
            chi_minimum = self.chi_square(residual_func, resid_args_with_params)

            combinations = list(itertools.combinations(varriables, 2))
            for comb in combinations:

                print(i, comb)

                param1_tag = comb[0]
                param1_best = current_param[param1_tag].value
                param2_tag = comb[1]
                param2_best = current_param[param2_tag].value

                param1_min = current_param[param1_tag].min
                param1_max = current_param[param1_tag].max

                param2_min = current_param[param2_tag].min
                param2_max = current_param[param2_tag].max

                #set up some axis for later on x_values for initial points of sampler and 
                #xgrid for interpolation
                para1_scale_max = param1_max*1.1
                para1_scale_min = param1_min*0.9

                para2_scale_max = param2_max*1.1
                para2_scale_min = param2_min*0.9
                fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
                fig.suptitle(f'Residue: {resid} {param1_tag} {param2_tag}')
                
                if 'tau' in comb[0]:
                    x_values = np.geomspace(para1_scale_min, para1_scale_max,init_grid)
                    xgrid = np.geomspace(para1_scale_min, para1_scale_max,surface_resolution)
                    ax1.set_xscale('log')
                    ax2.set_xscale('log')
                else:            
                    x_values = np.linspace(para1_scale_min, para1_scale_max,init_grid)
                    xgrid = np.linspace(para1_scale_min, para1_scale_max,surface_resolution)

                if 'tau' in comb[1]:
                    y_values = np.geomspace(para2_scale_min, para2_scale_max,init_grid)
                    ygrid = np.geomspace(para2_scale_min, para2_scale_max,surface_resolution)
                    ax1.set_yscale('log')
                    ax2.set_yscale('log')
                else:   
                    y_values = np.linspace(para2_scale_min, para2_scale_max,init_grid)                 
                    ygrid = np.linspace(para2_scale_min, para2_scale_max,surface_resolution)

                # param1_std = current_param[param1_tag].stderr
                
                # param1_min, param1_max = get_stderr_bounds(param1_best, current_param, param1_tag)
                # param2_min, param2_max = get_stderr_bounds(param2_best, current_param, param2_tag)
 
                # x_values = np.linspace(param1_min, param1_max,5)
                # xgrid = np.linspace(param1_min, param1_max,surface_resolution)
                
                # y_values = np.linspace(param2_min, param2_max,5)                 
                # ygrid = np.linspace(param2_min, param2_max,surface_resolution)

                points = []
                initial_evaluations = []

                #do some logic for the args
                dim1_scale = apply_log(param1_tag)
                dim2_scale = apply_log(param2_tag)
                
                dim1_bounds = [param1_min, param1_max]
                dim2_bounds = [param2_min, param2_max]
                sampler_metric_args = [dim1_bounds, dim2_bounds ,dim1_scale, dim2_scale]
                sampler_points_transform_args = [dim1_bounds, dim2_bounds]

                initial_points = []
                for ii in x_values:
                    for jj in y_values:
                        initial_points.append([ii,jj])
                
                initial_points.append([param1_best, param2_best])
                initial_points = np.array(initial_points)

                likelyhood_args = (param1_tag, param2_tag, current_param, residual_args, residual_func, chi_minimum)

                #Now we do the adaptive sampling
                Sampler = adaptive.AdaptiveSampler(self.chi_square_likelyhood,initial_points)
                Sampler.objective_args = likelyhood_args
                Sampler.user_metric = adpative_sampler_metric
                Sampler.user_metric_args = sampler_metric_args
                
                Sampler.points_transform = adpative_sampler_transform 
                Sampler.points_transform_args = sampler_points_transform_args
                
                Sampler.evalute_all_points()
                Sampler.run_n_cycles(sampler_resolution)
                Sampler.interpolate(resolution=sampler_resolution)

                #now we want to interpolate the data we have and plot it

                points = Sampler.points
                values = Sampler.evaluated_objective_funtion
                x = points[:,0]
                y = points[:,1]

                xgrid, ygrid = np.meshgrid(xgrid, ygrid)
                
                interpolate = scipy.interpolate.griddata((x,y),
                        Sampler.evaluated_objective_funtion,
                        (xgrid, ygrid), method='linear')
                
                # now we plot the surfaces
                transformer_points = adpative_sampler_transform(Sampler.points, *sampler_points_transform_args)
                tri = scipy.spatial.Delaunay(transformer_points)

                X, Y = np.meshgrid(xgrid, ygrid)
                levels = np.linspace(np.min(interpolate), np.max(interpolate), 50)

                interpolate[np.isnan(interpolate)] = np.min(interpolate) - 1 
                interpolate[np.isinf(interpolate)] = np.min(interpolate) - 1 

                ax1.contourf(xgrid, ygrid, interpolate, levels=20, cmap='Blues')
                ax1.scatter(param1_best, param2_best, s=20, facecolors='none', edgecolors='r')
                
                ax2.contourf(xgrid, ygrid, interpolate, levels=20, cmap='Blues' )
                ax2.scatter(param1_best, param2_best, s=20, facecolors='none', edgecolors='r')
                
                ax2.triplot(points[:,0], points[:,1], tri.simplices)
                ax2.plot(points[:,0], points[:,1], 'o')

                ax1.set_xlabel(f'{param1_tag}')
                ax1.set_ylabel(f'{param2_tag}')

                ax2.set_xlabel(f'{param1_tag}')
                ax2.set_ylabel(f'{param2_tag}')
                
                name = f'{plots_directory}/{resid}_{param1_tag}_{param2_tag}.pdf'
                plt.tight_layout()
                plt.savefig(name)
                plt.close('all')


    def plot_chi_square_surfs_brute(self, 
        atom_name, 
        protons,
        residual_type='hf', 
        model_pic='model_free_parameters.pic', 
        plots_directory='chi_square_surf_brute/', 
        brute_resolution=100, ):
        '''
        This function fits monoexponencial decays to intensities 
        for the relaxometry data, experimental and calculated
        '''

        os.makedirs(plots_directory, exist_ok=True)

        # load the model
        models, models_resid, sorted_keys, model_resinfo = self.read_pic_for_plotting(model_pic, atom_name)

        #iterate over the data points, could maybe change this loop to give tag directly
        for i in sorted_keys:
            res_info = model_resinfo[i]
            tag = utils.resinto_to_tag(*model_resinfo[i])
            resid, resname, atom1, atom2 = res_info

            current_param = models[tag].params
            varriables = []
            for var in current_param:
                if current_param[var].vary ==True:
                    varriables.append(var)
            
            combinations = list(itertools.combinations(varriables, 2))
            
            # the residual 
            residual_args, residual_func = self.residual_selector(tag, residual_type, res_info)
            for comb in combinations:

                brute_params = copy.deepcopy(current_param)
                for brute_p in brute_params:
                    if brute_p in comb:
                        brute_params[brute_p].vary = True
                        brute_step = abs(brute_params[brute_p].max-brute_params[brute_p].min)/brute_resolution
                        brute_params[brute_p].brute_step = brute_step
                    else:
                        brute_params[brute_p].vary = False

                minner = Minimizer(residual_func, brute_params, fcn_args=(residual_args))
                result = minner.minimize(method='brute')

                grid_x, grid_y = (np.unique(par.ravel()) for par in result.brute_grid)

                #now we want to interpolate the data we have and plot it
                
                plt.title(f'Residue: {resid} {comb[0]} {comb[1]}')
                levels = np.geomspace(np.min(result.brute_Jout), np.max(result.brute_Jout), 100)
                plt.contourf(grid_x, grid_y, result.brute_Jout.T, levels=levels , norm = LogNorm() )
               
                plt.xlabel(f'{comb[0]}')
                plt.ylabel(f'{comb[1]}')
                plt.colorbar()
                plt.scatter(current_param[comb[0]].value, current_param[comb[1]].value)


                name = f'{plots_directory}/{resid}_{comb[0]}_{comb[1]}.pdf'
                plt.tight_layout()
                plt.savefig(name)
                plt.close()
            

    def emcee(self, 
        atom_name, 
        residual_type='hf', 
        model_pic='model_free_parameters.pic',
        model_pic_emcee='model_free_parameters.pic', 
        plots_directory='emcee/'):

        '''
        This function fits monoexponencial decays to intensities 
        for the relaxometry data, experimental and calculated
        '''

        os.makedirs(plots_directory, exist_ok=True)
        print(f'using the residual {residual_type}')

        # load the model
        models, models_resid, sorted_keys, model_resinfo = self.read_pic_for_plotting(model_pic, atom_name)
        emcee_results = {}
        #iterate over the data points, could maybe change this loop to give tag directly
        
        for i in sorted_keys:
            res_info = model_resinfo[i]
            tag = utils.resinto_to_tag(*model_resinfo[i])
            resid, resname, atom1, atom2 = res_info
            current_param = models[tag].params
            residual_args, residual_func = self.residual_selector(tag, residual_type, res_info)

            minner = Minimizer(residual_func, current_param, fcn_args=(residual_args))
            result = minner.minimize(method='emcee')
            report_fit(result)
            emcee_results[tag] = result

        if writeout == True:
            with open(model_pic_emcee, 'wb') as handle:
                pic.dump(emcee_results, handle)







    def fit_proton_noes(self, atom_name1, protons, data_dir, fields, 
                        out_folder='proton_proton_buildup', 
                        model_pic='model_free_parameters.pic', 
                        delay_min=0.005, 
                        delay_max=0.03): 
        '''
        takes a file of relaxation rates and plots it against the elements from the relaxation 
        matrix
        '''

        os.makedirs(out_folder, exist_ok=True)
        models, models_resid, sorted_keys, model_resinfo = self.read_pic_for_plotting(model_pic, atom_name1)
        
        search = data_dir + '/*out'
        total_protons = [atom_name1] + protons
        files = glob.glob(search)
        intensities = {}
        delays = {}
        errors = {}
        operator_size = 2 + len(protons)*2

        # read in the files and assign to the correspondin dictionary
        files = self.remove_peakfit_outs(files)
        for fi in files:
            current_delays, current_intensities, current_errors, key = self.load_peakfit_file(fi)

            selector = np.logical_and(current_delays>delay_min ,current_delays<delay_max)
            intensities[key] = current_intensities[selector]
            delays[key] = current_delays[selector]
            errors[key] = current_errors[selector]

        for key in models:
            models, models_resid, sorted_keys, model_resinfo = self.read_pic_for_plotting(model_pic, atom_name1)
            
            res_info = utils.get_atom_info_from_tag(key)
            resid, resname, atom1, atom2 = res_info
            atom_1_c = atom1.replace("H", "C")

            
            # with motions            
            full_matrix = relax_mat.relaxation_matrix_emf_c1p(models[key].params,
                resid, 
                self.spectral_density, 
                fields, 
                self.distances, 
                self.cosine_angles, 
                resname,
                operator_size, 
                atom_1_c,
                atom_name1,
                protons, 
                x = 'c',
                y = 'h',)

            str_mat = np.array2string(full_matrix[:,:,0], separator='\t',suppress_small=True, 
                        formatter={"float": utils.np2string_formatter}, max_line_width=1e5) 
            # print(key)
            # print(str_mat)
            # print(intensities.keys())

            small_matrix = np.zeros([len(total_protons), len(total_protons)])
            
            for i in range(len(protons)):
                big_mat_i = i + 2
                i = i + 1
                for j in range(len(protons)):
                    big_mat_j = j + 2
                    j = j + 1

                    if i == j: 
                        small_matrix[i][j] = full_matrix[big_mat_i][big_mat_j]
                    else:
                        small_matrix[i][j] = full_matrix[big_mat_i][big_mat_j]

            small_matrix = small_matrix[:, :, np.newaxis]

            #without motions 
            full_matrix = relax_mat.relaxation_matrix_emf_c1p_no_long_range_motion(models[key].params,
                resid, 
                self.spectral_density, 
                fields, 
                self.distances, 
                self.cosine_angles, 
                resname,
                operator_size, 
                atom_1_c,
                atom_name1,
                protons, 
                x = 'c',
                y = 'h',)

            small_matrix_no_motion = np.zeros([len(total_protons), len(total_protons)])
            
            for i in range(len(protons)):
                big_mat_i = i + 2
                i = i + 1
                for j in range(len(protons)):
                    big_mat_j = j + 2
                    j = j + 1

                    if i == j: 
                        small_matrix_no_motion[i][j] = full_matrix[big_mat_i][big_mat_j]
                    else:
                        small_matrix_no_motion[i][j] = full_matrix[big_mat_i][big_mat_j]

            small_matrix_no_motion = small_matrix_no_motion[:, :, np.newaxis]

            key1 = f"{resname}{resid}{protons[2]}-{resname}{resid}{atom_name1}"
            key2 = f"{resname}{resid}{protons[1]}-{resname}{resid}{atom_name1}"

            check = True
            if key1 not in delays:
                check = False

            if key2 not in delays:
                check = False

            if check == True:
                current_delays1 = delays[key1]
                current_delays2 = delays[key2]
                synth_delays = np.linspace(min(current_delays1)*0.95, max(current_delays1)*1.05, 50)

                simlation = []
                
                #with motions 
                delay_propergators = mathFunc.construct_operator(small_matrix,synth_delays, product=False)
                experimets = np.zeros(len(protons)+1)
                experimets[1] = 1.
                calc_intensities = [np.matmul(i, experimets) for i in delay_propergators]
                calc_intensities = np.array(calc_intensities)

                #without motions
                delay_propergators = mathFunc.construct_operator(small_matrix_no_motion,synth_delays, product=False)
                experimets = np.zeros(len(protons)+1)
                experimets[1] = 1.
                calc_intensities_no_motion = [np.matmul(i, experimets) for i in delay_propergators]
                calc_intensities_no_motion = np.array(calc_intensities_no_motion)

                #with motions
                scale = np.mean(np.absolute([intensities[key1], intensities[key2]]))
                factor1 = np.mean([calc_intensities[:,2], calc_intensities[:,3]])/scale

                #without motions
                factor1_no_motion = np.mean([calc_intensities_no_motion[:,2], calc_intensities_no_motion[:,3]])/scale

                plt.title(resid)
                plt.plot(synth_delays, calc_intensities[:,2]/factor1, ':', c="C1", label=protons[1])
                plt.plot(synth_delays, calc_intensities[:,3]/factor1, ':', c="C2", label=protons[2])

                plt.plot(synth_delays, calc_intensities_no_motion[:,2]/factor1_no_motion,  c="C1", label=protons[1])
                plt.plot(synth_delays, calc_intensities_no_motion[:,3]/factor1_no_motion,  c="C2",label=protons[2])

                plt.legend()
                
                plt.errorbar(delays[key1], intensities[key1], yerr=errors[key1], c="C1", fmt='o')
                plt.errorbar(delays[key2], intensities[key2], yerr=errors[key2], c="C2", fmt='o')
                plt.xlabel('time (s)')
                plt.ylabel('intensity')
                plt.savefig(f'{out_folder}/{key1}_{key2}_{fields[0]}.pdf')
                plt.close()


        # for key in intensities.keys():
        #     res_info = utils.get_atom_info_from_tag(key)
        #     resid, resname, atom1, atom2 = res_info
        #     carbon_name = atom1.replace('H', 'C')

        #     if atom1 == atom_name1:

        #         model = models

    def get_coherence_population_with_time(self, 
        atom_name, 
        protons, 
        model_pic='model_free_parameters.pic', 
        plot=False, 
        plots_directory='coherence_populations/'):
        '''
        This function fits monoexponencial decays to intensities 
        for the relaxometry data, experimental and calculated
        '''

        os.makedirs(plots_directory, exist_ok=True)

        # load the model
        models, models_resid, sorted_keys, model_resinfo = self.read_pic_for_plotting(model_pic, atom_name)

        coherence_names = list(atom_name) + protons

        #iterate over the data points, could maybe change this loop to give tag directly
        for i in sorted_keys:
            res_info = model_resinfo[i]
            tag = utils.resinto_to_tag(*model_resinfo[i])
            resid, resname, atom1, atom2 = res_info

            for f in self.low_fields:
                calc_intensities = self.model_lf_single_field_intensities(models[tag].params, f, res_info, protons)
                delays =  self.ShuttleTrajectory.experiment_info[f]['delays']

                for indx, coherence in enumerate(coherence_names):
                    indx = indx + 1

                    current = []
                    for delay in calc_intensities:
                        current.append(delay[indx])
                    
                    plt.scatter(delays, current, label=coherence)

                plt.xlabel('delay')
                plt.ylabel('intensity')
                plt.legend()

                plt.savefig(f'{plots_directory}{tag}_ {f}_coherence.pdf')
                plt.close()

                # calc_intensities = np.array([j[1] for j in calc_intensities])

    def print_final_residuals(self, protons, pickle_path='default'):

        '''
        This function takes the output from an extended model free 
        fitting and then prints out the residuals

        Parameters
        ----------

        load_pickle : str
            if pickle_path is defined as 'default' then the atribute 
            self.fit_results_pickle_file is used. Otherwise it will take the 
            path to the pickle file and load this. This should be similar to the 
            output for per_residue_emf_fit()
         '''

        if pickle_path=='default':
            pickle_path = self.fit_results_pickle_file

        with open(pickle_path, "rb") as input_file:
            models = pic.load(input_file)

        print('ID\tlf_X2\thf_X2')
        for i in models:

            # this section could probably be grouped into a function 
            # as it is copied from fit_single_residue()

            res_info = utils.get_atom_info_from_tag(i)
            resid, resname, atom1, atom2 = res_info

            low_fields = list(self.ShuttleTrajectory.experiment_info.keys())
            intensities = []
            intensity_errors = []

            for f in low_fields:
                intensities.append(self.low_field_intensities[f][i])
                intensity_errors.append(self.low_field_intensity_errors[f][i])

            lf_residual = self.residual_lf(models[i].params, low_fields, res_info, protons, intensities, intensity_errors)
            lf_chisq = self.residual_to_chi2(np.array(lf_residual))

            hf_residual_args = [models[i].params, self.r1[i], self.r2[i], self.hetnoe[i], 
                                self.r1_err[i], self.r2_err[i], self.hetnoe[i], res_info]
            
            hf_residual = self.residual_hf(*hf_residual_args)
            hf_chisq = self.residual_to_chi2(np.array(hf_residual))
            print(f"{i}\t{lf_chisq:.2e}\t{hf_chisq:.2e}\t")

    def plot_relaxation_matrix_element_vs_measured(self, element_co_ords, file_values, file_errors, fields, out_folder, atom_name1, atom_name2, protons, model_pic='model_free_parameters.pic'): 
        '''
        takes a file of relaxation rates and plots it against the elements from the relaxation 
        matrix
        '''

        os.makedirs(out_folder, exist_ok=True)
        rates,_ = utils.read_nmr_relaxation_rate(file_values)
        errors,_ = utils.read_nmr_relaxation_rate(file_errors)
        models, models_resid, sorted_keys, model_resinfo = self.read_pic_for_plotting(model_pic, atom_name1)
        
        operator_size = 2 + len(protons)*2
        selected_keys = []
        for i in rates.keys():
            if atom_name1 in i:
                selected_keys.append(i)

        calc = []
        exp = []
        error = []

        for i in selected_keys: 

            res_info = utils.get_atom_info_from_tag(i)
            resid, resname, atom1, atom2 = res_info

            relaxation_matricies = relax_mat.relaxation_matrix_emf_c1p(models[i].params,
                resid, 
                self.spectral_density, 
                fields, 
                self.distances, 
                self.cosine_angles, 
                resname,
                operator_size, 
                atom_name1,
                atom_name2,
                protons, 
                x = 'c',
                y = 'h',)
            
            calc.append(relaxation_matricies[element_co_ords[0], element_co_ords[1],:])
            exp.append(rates[i])
            error.append(errors[i])

        calc = np.array(calc).T
        exp = np.array(exp).T
        error = np.array(error).T

        for ci, expi, erri, fi in zip(calc, exp, error, fields):

            maxi = np.max([ci, expi])
            mini = np.min([ci, expi])
            print([mini, mini], [maxi, maxi])
            plt.plot([mini, maxi], [mini, maxi])
            plt.errorbar(ci, expi, yerr=erri, label=f'{fi:0.2f} T', fmt='o')
            plt.legend()
            plt.xlabel('calc rate')
            plt.ylabel('exp rate')
            plt.tight_layout()
            # plt.show()
            plt.savefig(out_folder+f"/rate_{fi:0.2f}.pdf")
            plt.close()


    def write_out_relaxation_matricices_c1p(self,atom_name1, atom_name2,
        protons,
        matrix_directory='lowfield_relaxation_matricies/',
        operator_directory='lowfield_relaxation_operator/',
        model_pic='model_free_parameters.pic'):

        #make the directory
        os.makedirs(matrix_directory, exist_ok=True)
        os.makedirs(operator_directory, exist_ok=True)
        models, models_resid, sorted_keys, model_resinfo = self.read_pic_for_plotting(model_pic, atom_name1)
        
        operator_size = 2 + len(protons)*2

        for i in sorted_keys:
            tag = utils.resinto_to_tag(*model_resinfo[i])
            res_info = model_resinfo[i]
            resid, resname, atom1, atom2 = res_info
            low_field_relaxation_matrix = relax_mat.relaxation_matrix_emf_c1p(models[tag].params,
                    resid, 
                    self.spectral_density, 
                    np.array([self.low_fields]).flatten(), 
                    self.distances, 
                    self.cosine_angles, 
                    resname,
                    operator_size, 
                    atom_name1,
                    atom_name2,
                    protons, 
                    x = 'c',
                    y = 'h',)

            high_field_relaxation_matrix = relax_mat.relaxation_matrix_emf_c1p(models[tag].params,
                    resid, 
                    self.spectral_density, 
                    np.array([self.high_fields]).flatten(), 
                    self.distances, 
                    self.cosine_angles, 
                    resname,
                    operator_size, 
                    atom_name1,
                    atom_name2,
                    protons, 
                    x = 'c',
                    y = 'h',)
            
            file_name = matrix_directory + f"{tag}_relax_matrix.txt"
            file_op_name = operator_directory + f"{tag}_relax_matrix.txt"
            
            # f_op = open(file_op_name, 'w')
            f = open(file_name, 'w')

            matricies = [low_field_relaxation_matrix, high_field_relaxation_matrix]
            field_list = [self.low_fields, self.high_fields]

            for matrix_list, field_list in zip(matricies, field_list):
                for matrix, field in zip(matrix_list.T,  field_list):

                    str_mat = np.array2string(matrix, separator='\t',suppress_small=True, 
                        formatter={"float": utils.np2string_formatter}, max_line_width=1e5) 
                    f.write(f'==== {field} ===\n')
                    f.write(str_mat)
                    f.write('\n')

                # operator = mathFunc.construct_operator([low_field_relaxation_matrix[0]], [100e-3], product=False) 
                # print(operator)
                # str_mat_op = np.array2string(operator, precision=3, separator='\t',suppress_small=False) 
                # f_op.write(f'==== {field} ===\n')
                # f_op.write(str_mat_op)
                # f_op.write('\n')
                        
            f.close()
            #f_op.close()


    def print_trajectory_info(self):

        for field in self.low_fields:
            traj = self.ShuttleTrajectory.trajectory_time_at_fields[field]
            forwards_field, forwards_time, backwards_field, backwards_time = traj
            pretty_fields = np.array2string(forwards_field, precision=2, separator=' , ') 
            pretty_time = np.array2string(forwards_time*1000, precision=2, separator=' , ')
            print(f'==== {field} ====')
            print(f"forwards fields {pretty_fields}")
            print(f"forwards time (ms) {pretty_time}")
            print(f'total time travelled: {np.sum(forwards_time)*1000:0.2f}ms')


    def plot_model_free_parameters(self, atom_name, model_pic='model_free_parameters.pic', plot_name='model_free_params.pdf', showplot=False):

        # load the model
        models, models_resid, sorted_keys, model_resinfo = self.read_pic_for_plotting(model_pic, atom_name)
        model_array = []
        model_err_array = []
        vary_status = []


        print('sorted keys:', sorted_keys)

        for i in sorted_keys:
            values = [models_resid[i].params[j].value for j in models_resid[i].params]
            stds = [models_resid[i].params[j].stderr for j in models_resid[i].params]
            vary = [models_resid[i].params[j].vary for j in models_resid[i].params]

            model_array.append(values)
            model_err_array.append(stds)
            vary_status.append(vary)

        model_array = np.array(model_array)
        vary_status = np.array(vary_status)
        model_err_array = np.array(model_err_array)
        
        #model_err_array[np.isnan(model_err_array)] = 0
        model_err_array[model_err_array==None]=0


        number_of_params = model_err_array.shape[1]
        param_names = [j for j in models_resid[i].params]

        fig_width = math.ceil(number_of_params/3)
        fig, ax = plt.subplots(nrows=3, ncols=fig_width, figsize=(3*fig_width, 9))
        ax = ax.flatten()

        for i in range(number_of_params):

            colors = []
            for ii in vary_status.T[i]:
                if ii == True:
                    colors.append('#1f77b4')
                else:
                    colors.append('#ff7f0e')



            ax[i].set_ylabel(param_names[i])
            ax[i].set_xlabel('residue')
            ax[i].errorbar(sorted_keys, model_array.T[i], yerr=model_err_array.T[i],fmt='|',color='k', zorder=1)
            ax[i].scatter(sorted_keys, model_array.T[i], color=colors, zorder=2)
            if 'tau' in param_names[i]:
                ax[i].set_yscale('log')

            if "S" in param_names[i]:
                ax[i].set_ylim(0,1.1)
        plt.tight_layout()
        plt.savefig(plot_name)
        if showplot == True:
            plt.show()
        plt.close()

    def plot_r1_r2_noe(self,atom_name, 
        model_pic='model_free_parameters.pic', 
        fields_min=14, 
        fields_max=25, 
        plot_ranges=[[0,3.5], [10,35], [1, 1.9]], 
        per_residue_prefix='hf_rates',
        plot_per_residue=True):

        # load the model
        models, models_resid, sorted_keys, model_resinfo = self.read_pic_for_plotting(model_pic, atom_name)

        r1_exp_array = []
        r1_exp_array_err = []
        r1_model = []
        
        r2_exp_array = []
        r2_exp_array_err = []
        r2_model = []

        noe_array = []
        noe_array_err = []
        noe_model = []


        for i in sorted_keys:

            res_info = model_resinfo[i]
            resid, resname, atom1, atom2 = res_info

            tag = utils.resinto_to_tag(resid, resname, atom1, atom2)
            model_r1, model_r2, model_noe = self.model_hf(models_resid[i].params,res_info, self.model_high_fields)
            model_r1_v2, model_r2_v2, model_noe_v2 = self.model_hf(models_resid[i].params,res_info, self.high_fields)

            r1_exp_array.append(self.r1[tag])
            r2_exp_array.append(self.r2[tag])
            noe_array.append(self.hetnoe[tag])

            r1_exp_array_err.append(self.r1_err[tag])
            r2_exp_array_err.append(self.r2_err[tag])
            noe_array_err.append(self.hetnoe_err[tag])

            r1_model.append(model_r1_v2)
            r2_model.append(model_r2_v2)
            noe_model.append(model_noe_v2)

            if plot_per_residue == True:

                fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
                ax0.set_title(f'R1 {i}')
                ax0.errorbar(self.high_fields, self.r1[tag],yerr=self.r1_err[tag], fmt='o')
                ax0.plot(self.model_high_fields, model_r1)
                ax0.set_ylim(*plot_ranges[0])
                ax0.set_xlabel('field (T)')
                ax0.set_ylabel('R_{1}(Hz)')

                ax1.set_title('R2')
                ax1.errorbar(self.high_fields, self.r2[tag],yerr=self.r2_err[tag], fmt='o')
                ax1.plot(self.model_high_fields, model_r2)
                ax1.set_ylim(*plot_ranges[1])
                ax1.set_xlabel('field (T)')
                ax1.set_ylabel('R_{2} (Hz)')

                ax2.set_title('NOE')
                ax2.errorbar(self.high_fields, self.hetnoe[tag],yerr=self.hetnoe_err[tag], fmt='o')
                ax2.plot(self.model_high_fields, model_noe)
                ax2.set_ylim(*plot_ranges[2])
                ax2.set_ylabel('I_{sat}/I_{0}')

                plt.tight_layout()
                plt.savefig(f"{per_residue_prefix}_{atom_name}_{resid}.pdf")
                plt.close()


        fig, (ax0,ax1,ax2) = plt.subplots(nrows=3, ncols=1, figsize=(12, 8), sharex=True)

        r1_exp_array = np.array(r1_exp_array)
        r1_exp_array_err = np.array(r1_exp_array_err)

        r2_exp_array = np.array(r2_exp_array)
        r2_exp_array_err = np.array(r2_exp_array_err)
        
        noe_array = np.array(noe_array)
        noe_array_err = np.array(noe_array_err)

        noe_model = np.array(noe_model)
        r1_model = np.array(r1_model)
        r2_model = np.array(r2_model)

        for i in range(len(self.high_fields)):

            ax0.errorbar(sorted_keys, r1_exp_array.T[i],yerr=r1_exp_array_err.T[i], fmt='o', c=f'C{i}')
            ax0.plot(sorted_keys, r1_model.T[i], color=f'C{i}')
            ax0.set_ylabel('R_{1}(Hz)')

            ax1.errorbar(sorted_keys, r2_exp_array.T[i],yerr=r2_exp_array_err.T[i], fmt='o', c=f'C{i}')
            ax1.plot(sorted_keys, r2_model.T[i], color=f'C{i}')
            ax1.set_ylabel('R_{2} (Hz)')

            ax2.errorbar(sorted_keys, noe_array.T[i], yerr=noe_array_err.T[i], fmt='o', c=f'C{i}')
            ax2.plot(sorted_keys, noe_model.T[i], color=f'C{i}')
            ax2.set_xlabel('residue')
            ax2.set_ylabel('I_{sat}/I_{0}')

        plt.tight_layout()
        plt.savefig(f"{atom_name}_rates.pdf")
        plt.close()

    def plot_sigma(self,atom_name, 
        model_pic='model_free_parameters.pic', 
        fields_min=14, 
        fields_max=25, 
        per_residue_prefix='sigma_rates'):

        # load the model
        models, models_resid, sorted_keys, model_resinfo = self.read_pic_for_plotting(model_pic, atom_name)

        sigma_array = []
        sigma_array_error = []
        sigma_model = []

        for i in sorted_keys:

            res_info = model_resinfo[i]
            resid, resname, atom1, atom2 = res_info

            tag = utils.resinto_to_tag(resid, resname, atom1, atom2)
            model_sigma = self.model_sigma(models_resid[i].params,res_info, self.model_high_fields)
            
            fig, (ax0) = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
            ax0.set_title(f'R1 {i}')
            ax0.errorbar(self.high_fields, self.sigma[tag],yerr=self.sigma_err[tag], fmt='o')
            ax0.plot(self.model_high_fields, model_sigma)
            ax0.set_xlabel('field (T)')
            ax0.set_ylabel('R_{1}(Hz)')

            plt.tight_layout()
            plt.savefig(f"{per_residue_prefix}_{atom_name}_{resid}.pdf")
            plt.close()

    def plot_relaxometry_intensities_together(self,
        atom_name, protons,
            model_pic='model_free_parameters.pic',
            folder='relaxometry_intensities/', 
            cmap='Blues', fancy=False):

        if fancy == True:
            plt.style.use(['science','nature'])

        try:
            os.mkdir(folder)
        except FileExistsError:
            pass 

        models, models_resid, sorted_keys, model_resinfo = self.read_pic_for_plotting(model_pic, atom_name)
        low_fields = sorted(self.ShuttleTrajectory.experiment_info.keys())
        print('plotting relaxometry_intensities')

        norm = matplotlib.colors.Normalize(0, 10, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap)

        for i in tqdm(sorted_keys):
            tag = utils.resinto_to_tag(*model_resinfo[i])

            for f in low_fields:

                color = mapper.to_rgba(f)
                delays =  self.ShuttleTrajectory.experiment_info[f]['delays']
                intensities = self.low_field_intensities[f][tag]
                intensity_errors = self.low_field_intensity_errors[f][tag]
                
                model = self.model_lf_single_field_intensities(models_resid[i].params, f, model_resinfo[i], protons)
                model =  np.array([i[1] for i in model])
                
                factor = np.mean(intensities/model)

                exp_intensity_scalled = intensities/factor
                exp_error_scalled = intensity_errors/factor
                plt.title(f'{tag}')

                model = [x for _,x in sorted(zip(delays,model))]
                model_delays = sorted(delays)
                
                plt.plot(model_delays, model, color=color,zorder=1)
                plt.errorbar(delays, exp_intensity_scalled, yerr=np.absolute(exp_error_scalled),fmt='|', 
                    c=color,zorder=2)
                plt.scatter(delays, exp_intensity_scalled, color=color, label=f"{f:0.1f} T",edgecolor='black',zorder=3)
            
            plt.ylabel('Intensity')
            plt.xlabel('delay (s)')
            plt.legend()
            plt.savefig(f'{folder}field_{tag}_together.pdf')
            plt.close()

    def plot_hf_relaxometry(self,
        atom_name, protons,
        fields_min=14, 
        fields_max=25, 
        plot_ranges=[[0,3.5], [10,35], [1, 1.9]], 
        folder='folder',
        data_folder='data_for_figures/',
        model_pic='model_free_parameters.pic', 
        fancy=False,
        marker_size=15,
        text_size = 10,
        tick_size=8, 
        cmap='Blues'):

        if fancy == True:
            plt.style.use(['science','nature'])


        try:
            os.mkdir(folder)
        except FileExistsError:
            pass 

        try:
            os.mkdir(data_folder)
        except FileExistsError:
            pass 

        models, models_resid, sorted_keys, model_resinfo = self.read_pic_for_plotting(model_pic, atom_name)
        
        if hasattr(self, 'ShuttleTrajectory'):
            low_fields = sorted(self.ShuttleTrajectory.experiment_info.keys())
        else:
            print('No low fields')
            low_fields = []



        print('plotting relaxometry_intensities')

        norm = matplotlib.colors.Normalize(0, 10, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap)

        data = {}

        for i in tqdm(sorted_keys):

            data[i] = {}
            data[i]['r1'] = {}
            data[i]['r2'] = {}
            data[i]['noe'] = {}
            data[i]['relaxometry_intensity'] = {}

            tag = utils.resinto_to_tag(*model_resinfo[i])
            res_info = model_resinfo[i]
            model_r1, model_r2, model_noe = self.model_hf(models_resid[i].params,res_info, self.model_high_fields)
            model_r1_v2, model_r2_v2, model_noe_v2 = self.model_hf(models_resid[i].params,res_info, self.high_fields)

            fig, (ax0, ax1, ax2,ax3) = plt.subplots(nrows=1, ncols=4, figsize=(16*0.66, 4*0.66))
            #ax0.set_title('$R_{1}$')
            ax0.errorbar(self.high_fields, self.r1[tag],yerr=self.r1_err[tag], fmt='|', zorder=2)
            ax0.scatter(self.high_fields, self.r1[tag], edgecolor='black', zorder=3, s=marker_size)
            ax0.plot(self.model_high_fields, model_r1, zorder=1)
            ax0.set_ylim(*plot_ranges[0])
            ax0.set_xlabel('Field (T)', fontsize=text_size)
            ax0.set_ylabel('$R_{1}$ (Hz)', fontsize=text_size)
            ax0.tick_params(axis='both', labelsize=tick_size)

            data[i]['r1']['exp'] = [self.high_fields, self.r1[tag], self.r1_err[tag]]
            data[i]['r1']['model'] = [self.model_high_fields, model_r1]
            data[i]['r1']['model_hf_points'] = [self.high_fields, model_r1_v2]

            #ax1.set_title('$R_{2}$')
            ax1.errorbar(self.high_fields, self.r2[tag],yerr=self.r2_err[tag], fmt='|', zorder=2,)
            ax1.scatter(self.high_fields, self.r2[tag], edgecolor='black', zorder=3, s=marker_size)
            ax1.plot(self.model_high_fields, model_r2, zorder=1)
            ax1.set_ylim(*plot_ranges[1])
            ax1.set_xlabel('Field (T)', fontsize=text_size)
            ax1.set_ylabel('$R_{2}$ (Hz)', fontsize=text_size)
            ax1.tick_params(axis='both', labelsize=tick_size)

            data[i]['r2']['exp'] = [self.high_fields, self.r2[tag], self.r2_err[tag]]
            data[i]['r2']['model'] = [self.model_high_fields, model_r2]
            data[i]['r2']['model_hf_points'] = [self.high_fields, model_r2_v2]

            #ax2.set_title('$I_{sat}/I_{0}$')
            ax2.errorbar(self.high_fields, self.hetnoe[tag],yerr=self.hetnoe_err[tag], fmt='|', zorder=2)
            ax2.scatter(self.high_fields, self.hetnoe[tag], edgecolor='black', zorder=3, s=marker_size)
            ax2.plot(self.model_high_fields, model_noe, zorder=1)
            ax2.set_ylim(*plot_ranges[2])
            ax2.set_ylabel('$I_{sat}/I_{0}$', fontsize=text_size)
            ax2.set_xlabel('Field (T)', fontsize=text_size)
            ax2.tick_params(axis='both', labelsize=tick_size)

            data[i]['noe']['exp'] = [self.high_fields, self.hetnoe[tag], self.hetnoe_err[tag]]
            data[i]['noe']['model'] = [self.model_high_fields, model_noe]
            data[i]['noe']['model_hf_points'] = [self.high_fields, model_noe_v2]

            for f in low_fields:

                color = mapper.to_rgba(f)
                delays =  self.ShuttleTrajectory.experiment_info[f]['delays']
                intensities = self.low_field_intensities[f][tag]
                intensity_errors = self.low_field_intensity_errors[f][tag]
                
                model = self.model_lf_single_field_intensities(models_resid[i].params, f, model_resinfo[i], protons)
                model =  np.array([i[1] for i in model])
                
                factor = np.mean(intensities/model)

                exp_intensity_scalled = intensities/factor
                exp_error_scalled = intensity_errors/factor

                model = [x for _,x in sorted(zip(delays,model))]
                model_delays = sorted(delays)
                
                ax3.plot(model_delays, model, color=color,zorder=1)
                print('key ', i, 'errors: ', exp_error_scalled)
                ax3.errorbar(delays, exp_intensity_scalled, yerr=np.absolute(exp_error_scalled),fmt='|', c=color,zorder=2)
                ax3.scatter(delays, exp_intensity_scalled, color=color, label=f"{f:0.1f} T",edgecolor='black',zorder=3, s=marker_size)

                data[i]['relaxometry_intensity'][f] = {}
                data[i]['relaxometry_intensity'][f]['exp'] = [delays, exp_intensity_scalled, exp_error_scalled]
                data[i]['relaxometry_intensity'][f]['model'] = [model_delays, model]

            #ax3.set_title('$Relaxometry$')
            ax3.set_ylabel('Intensity', fontsize=text_size)
            ax3.set_xlabel('delay (s)', fontsize=text_size)
            ax3.legend(fontsize=tick_size)
            ax3.tick_params(axis='both', labelsize=tick_size)
            plt.tight_layout()
            plt.savefig(f'{folder}_{tag}_together.pdf')
            plt.close()

        pic_out = open(data_folder + 'hf_rates_lf_intensities.pic', 'wb')
        pic.dump(data, pic_out)
        pic_out.close()




    def plot_relaxometry_rates(self,
        atom_name,
        folder='relaxometry_plots',
        model_pic='model_free_parameters.pic', 
        fancy=False,
        marker_size=15,
        text_size = 10,
        tick_size=8, 
        data_folder='data_for_figures/',):

        if fancy == True:
            plt.style.use(['science','nature'])

        try:
            os.mkdir(folder)
        except FileExistsError:
            pass 
        
        try:
            os.mkdir(data_folder)
        except FileExistsError:
            pass 

        models, models_resid, sorted_keys, model_resinfo = self.read_pic_for_plotting(model_pic, atom_name)
        print('plotting relaxometry_intensities')
        data = {}
        for i in tqdm(sorted_keys):

            data[i] = {}

            tag = utils.resinto_to_tag(*model_resinfo[i])
            res_info = model_resinfo[i]
            model_r1, model_r2, model_noe = self.model_hf(models_resid[i].params,res_info, self.model_all_fields)
            
            fig, (ax0) = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))

            ax0.plot(self.model_all_fields, model_r1, zorder=1, c='C2', label='Simulated $R_1$')
            data[i]['hf_model'] = [self.model_all_fields, model_r1]
            values = self.relaxometry_mono_exp_fits[tag]
            err = self.relaxometry_mono_exp_fits_err[tag]

            values_calc = self.relaxometry_mono_calc_fits[tag]
            err_calc = self.relaxometry_mono_calc_fits_err[tag]

            ax0.set_title(tag)
            ax0.errorbar(self.low_fields, values,yerr=err, fmt='|', zorder=2, c='C1')
            ax0.scatter(self.low_fields, values, edgecolor='black', zorder=3, s=marker_size, c='C1', label='Low field $R_{1,exp}^{apparent}$')
            data[i]['lf_exp'] = [self.low_fields, values, err]

            ax0.errorbar(self.low_fields, values_calc,yerr=err_calc, fmt='|', zorder=2, c='C4')
            ax0.scatter(self.low_fields, values_calc, edgecolor='black', zorder=3, s=marker_size, c='C4', label='Low field $R_{1,calc}^{apparent}$')
            data[i]['lf_calc'] = [self.low_fields, values_calc, err_calc]

            ax0.errorbar(self.high_fields, self.r1[tag],yerr=self.r1_err[tag], fmt='|', zorder=2)
            ax0.scatter(self.high_fields, self.r1[tag], edgecolor='black', zorder=3, s=marker_size, c='C3', label='High Field $R_1$')
            data[i]['hf'] = [self.high_fields, self.r1[tag], self.r1_err[tag]]
            
            ax0.set_xlabel('Field (T)', fontsize=text_size)
            ax0.set_ylabel('$R_{1}^{app}$ (Hz)', fontsize=text_size)
            ax0.tick_params(axis='both', labelsize=tick_size)
            ax0.set_xscale('log')
            ax0.set_xlim(1, 25)
            ax0.legend()

            plt.tight_layout()
            plt.savefig(f'{folder}_{tag}_together.pdf')
            plt.close()
        
        pic_out = open(data_folder + 'lf_rates.pic', 'wb')
        pic.dump(data, pic_out)
        pic_out.close()

    def plot_relaxometry_intensities(self,atom_name, protons,model_pic='model_free_parameters.pic',folder='relaxometry_intensities/'):

        try:
            os.mkdir(folder)
        except FileExistsError:
            pass 

        models, models_resid, sorted_keys, model_resinfo = self.read_pic_for_plotting(model_pic, atom_name)
        low_fields = self.ShuttleTrajectory.experiment_info.keys()
        
        print('plotting relaxometry_intensities')
        for i in tqdm(sorted_keys):
            tag = utils.resinto_to_tag(*model_resinfo[i])
            for f in low_fields:

                delays =  self.ShuttleTrajectory.experiment_info[f]['delays']
                intensities = self.low_field_intensities[f][tag]
                intensity_errors = self.low_field_intensity_errors[f][tag]
                
                model = self.model_lf_single_field_intensities(models_resid[i].params, f, model_resinfo[i], protons)
                model =  np.array([i[1] for i in model])
                factor = np.mean(intensities/model)

                plt.title(f'{tag} Field: {f:0.2f}T')
                plt.errorbar(delays, intensities,yerr=intensity_errors,  label='experimantal', fmt='o', c="C1")

                model = [x for _,x in sorted(zip(delays,model*factor))]
                delays = sorted(delays)
                plt.plot(delays, model, label='model')
                plt.ylabel('Intensity')
                plt.xlabel('delay (s)')
                plt.legend()
                plt.savefig(f'{folder}field_{f}_{tag}.pdf')
                plt.close()

    def write_out_correction_factors(self,atom_name,
        prefix='',
        model_pic='model_free_parameters.pic'):

        models, models_resid, sorted_keys, model_resinfo = self.read_pic_for_plotting(model_pic, atom_name)
        print('plotting relaxometry_intensities')
        out = open(prefix + 'correction_factors.dat', 'w')
        fields_string = ' '.join([str(a) for a in self.low_fields])
        out.write(f'# {fields_string}\n')
        correction_factors_list = []
        for i in tqdm(sorted_keys):

            tag = utils.resinto_to_tag(*model_resinfo[i])
            res_info = model_resinfo[i]
            model_r1, model_r2, model_noe = self.model_hf(models_resid[i].params,res_info, self.model_all_fields)

            values = self.relaxometry_mono_exp_fits[tag]
            model_r1, model_r2, model_noe = self.model_hf(models_resid[i].params,res_info, np.array(self.low_fields))

            correction_factors = model_r1/values
            correction_factors_list.append(correction_factors)
            correction_string = '\t'.join([f'{a:0.3f}' for a in correction_factors])
            out.write(f'{tag}    {correction_string}\n')

        correction_factors = np.array(correction_factors_list)
        mean_correction_factors = np.mean(correction_factors, axis=0)
        std_correction_factors = np.std(correction_factors, axis=0)

        correction_string = '\t'.join([f'{a:0.3f}' for a in mean_correction_factors])
        out.write(f'mean_correction_factors    {correction_string}\n')

        correction_string = '\t'.join([f'{a:0.3f}' for a in std_correction_factors])
        out.write(f'std_correction_factors    {correction_string}\n')
        
        out.close()

# def generate_mf_parameters(dx,dy,dz,scale_diffusion=False, diffusion=None):
#     if scale_diffusion == False:
#         models_state = [[True, 1, 0, 0, False],
#                   [True, 1, True, 0, False], 
#                   [1, True, 0, True, False], 
#                   [True, True, 0, True, False],
#                   [True, True, True, True, False]]

#     if scale_diffusion == True:
#         models_state = [[True, 1, 0, 0, False],
#                   [True, 1, True, 0, False], 
#                   [1, True, 0, True, False], 
#                   [True, True, 0, True, False],
#                   [True, True, True, True, False],
#                   [True, 1, 0, 0, False],
#                   [True, 1, True, 0, True], 
#                   [1, True, 0, True, True], 
#                   [True, True, 0, True, True],
#                   [True, True, True, True, True]]
    
#     params_lists = []
#     for mod in models_state:

#         params = Parameters()
#         #internal dynamics 
#         if mod[3] == True:
#             params.add('tau_s', value=0.5e-9, vary=True, max=15e-9, min=500e-12)
#         else:
#             params.add('tau_s', value=mod[3], vary=False, min=10e-12)

#         if mod[1] == True:
#             params.add('Ss', value=0.8, min=0.4, vary=True, max=1)
#         else:
#             params.add('Ss', value=mod[1], vary=False,)

#         # add a constraint on the diffeerence between tauf and taus
#         if mod[2] == True:
#             params.add('tau_f', value=100e-12, min=40e-12, vary=True,max=500e-12)
#         else: 
#             params.add('tau_f', value=mod[2], vary=False)    

#         #Sf 
#         if mod[0] == True:
#             params.add('Sf', value=0.9, min=0.4, vary=True, max=1)
#         else:
#             params.add('Sf', value=mod[0], vary=False,)

#         params.add('diff', min=0, expr='tau_s-tau_f*5')

#         #diffusion

#         if diffusion == None:
#             params.add('dx_fix',  min=0, value=dx, vary=False)
#             params.add('dy_fix',  min=0, value=dy, vary=False)
#             params.add('dz_fix',  min=0, value=dz, vary=False)
#             params.add('diff_scale', min=0, value=1, vary=mod[4])

#         else:
#             params.add('dx_fix',  min=0, value=diffusion[0], vary=False)
#             params.add('dy_fix',  min=0, value=diffusion[1], vary=False)
#             params.add('dz_fix',  min=0, value=diffusion[2], vary=False)
#             params.add('diff_scale', min=0, value=1, vary=mod[4])

#         params.add('dx',  expr='dx_fix*diff_scale')
#         params.add('dy',  expr='dy_fix*diff_scale')
#         params.add('dz',  expr='dz_fix*diff_scale')
#         params_lists.append(params)

#     return params_lists
