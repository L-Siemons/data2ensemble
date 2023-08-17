import data2ensembles.utils as utils
import data2ensembles.trrAnalysis as trrAnalysis
import data2ensembles.mathFuncs as mathFunc
import data2ensembles.spectralDensity as specDens
import data2ensembles.rates
import data2ensembles as d2e
import data2ensembles.structureUtils as strucUtils
import data2ensembles.shuttling as shuttling
import data2ensembles.relaxation_matricies as relax_mat

from timeit import default_timer as timer
from datetime import timedelta

import matplotlib
import matplotlib.cm as cm
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

    def __init__(self,dx,dy,dz):

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

        check = False
        if self.relaxometry_mono_exponencial_fits_path != None:
            if self.relaxometry_mono_exponencial_fits_err_path !=  None:
                check=True

        if check == True:
            print('Loading monoexponencial fits of relaxometry decays')
            self.relaxometry_mono_exp_fits, _ = utils.read_nmr_relaxation_rate(self.relaxometry_mono_exponencial_fits_path)
            self.relaxometry_mono_exp_fits_err, _ = utils.read_nmr_relaxation_rate(self.relaxometry_mono_exponencial_fits_err_path)

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

    def calc_cosine_angles_and_distances(self, atom_names, dist_skip=1):
        '''
        This function determines the cosine angles

        This function defines the attributes: 
        self.cosine_angles
        self.distances
        self.uni 

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
        structure_analysis.calc_inter_atom_distances(atom_names, skip=1)
        structure_analysis.path_prefix = self.path_prefix

        structure_analysis.calc_cosine_angles(atom_names, 
                        calc_csa_angles=False, 
                        calc_average_structure=False,
                        reference_gro_file = self.pdb, 
                        delete_rmsf_file=True, 
                        dt = 1, use_reference=True)

        structure_analysis.calc_inter_atom_distances(atom_names, skip=dist_skip)

        self.cosine_angles = structure_analysis.cosine_angles
        self.distances = structure_analysis.md_distances
        self.uni = structure_analysis.uni

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
            for fi in files:
                if fi.split('/')[-1] not in ('logs.out' , 'shifts.out'):
                    array = np.genfromtxt(fi, comments="#")
                    delays, intensities, errors = array.T
                    key = fi.split('/')[-1].split('.')[0]

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

    def filter_atoms(self, resid, atoms, print_out=False):
        '''
        removes the atoms that are not present in the residue
        
        Parameters
        ----------
        resid : int 
            residue ID 

        atoms : list
            list of atom names we want to check

        print_out : bool 
            controls a printout for debugging

        Returns
        -------
        atoms_filtered : list 
            the list of atoms present in the residue
        '''

        atoms_filtered = []
        for i in atoms:
            sele = self.uni.select_atoms(f'resid {resid} and name {i}')

            if len(sele) != 0:
                atoms_filtered.append(i)
            
            elif print_out == True:
                test_sele = self.uni.select_atoms(f'resid {resid}')
                print(f' atom {i} is missing', test_sele[0].resname)

        return atoms_filtered


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


        delay_propergators = mathFunc.construct_operator(low_field_relaxation_matrix, 
                                self.ShuttleTrajectory.experiment_info[low_field]['delays'], product=False)

        # for i in range(100):
        #     #mathFunc.construct_operator(low_field_relaxation_matrix, 
        #     #                    self.ShuttleTrajectory.experiment_info[low_field]['delays'], product=False)

            

        #     roll = np.rollaxis(low_field_relaxation_matrix, 2)
        #     mathFunc.matrix_exp(roll[0])

        full_propergators = [stablaization_operator.dot(backwards_operators).dot(i).dot(forwrds_propergators) for i in delay_propergators]
        experimets = np.zeros(operator_size)
        experimets[1] = 1.
        #print(np.array_str(full_propergators[0].dot(experimets), precision=2, suppress_small=True))
        #print(len(full_propergators), full_propergators[0].shape)

        intensities = [np.dot(i, experimets) for i in full_propergators]
        
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

        # if params['tau_s'] < params['tau_f']:
        #     constraint = (params['tau_s'] - params['tau_f'])*1e12
        # print("model:", model_r1, model_r2, model_noe)

        return np.concatenate([r1_diff.flatten(), r2_diff.flatten(), noe_diff.flatten(), [constraint]])

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

    def fit_single_residue(self, i,protons=(), provided_diffusion=False, residual_type='hflf', model_select=True):
        '''
        This function fits all the models available in model free (set by generate_mf_parameters())
        and selects the best one based on the lowest BIC. 

        Paramteres
        ----------
        i : str
            the residue tag

        protons : tupple 
            This is a tupple of the protons you want to include in the relaxation analyis
            of the low field data. This only needs to be given when analysing low field data 

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
        print(i)
        # get the atom info
        res_info = utils.get_atom_info_from_tag(i)
        resid, resname, atom1, atom2 = res_info
        all_models = []

        # probably dont need a dictionary for the bic any more 
        bics = {} 
        bics[i] = 10e1000

        # the rows are Sf, Ss, tauf, taus, might be best to take this out and put it in its own function
        if provided_diffusion == False:
            dx,dy,dz = self.diffusion
        else:
            dx,dy,dz = provided_diffusion

        # remove the protons not in the residue 
        # protons = self.filter_atoms(resid, protons)

        params_models = generate_mf_parameters(dx,dy,dz, scale_diffusion=self.scale_diffusion)
        
        for params in params_models:

            resid, resname, atom1, atom2 = res_info
            key = (resid, atom1 , resid, atom2)
            # args = (np.array([23,16])*self.PhysQ.gamma['c'],self.cosine_angles[key][0], self.cosine_angles[key][1], self.cosine_angles[key][2], )
            
            if 'hf' in residual_type:
                hf_data = (self.r1[i], self.r2[i], self.hetnoe[i])
                hf_errors = (self.r1_err[i], self.r2_err[i], self.hetnoe_err[i])

            if 'lf' in residual_type:
                if protons == ():
                    print("WARNING there are no protons! exiting - sorry !")
                    os.exit() 
                
                # protons = ("H1'", "H2'", "H2''")
                low_fields = list(self.ShuttleTrajectory.experiment_info.keys())
                intensities = []
                intensity_errors = []

                for f in low_fields:
                    intensities.append(self.low_field_intensities[f][i])
                    intensity_errors.append(self.low_field_intensity_errors[f][i])

            if residual_type == 'hflf':
                # do fit, here with the default leastsq algorithm
                minner = Minimizer(self.residual_hflf, params, fcn_args=(hf_data, 
                    hf_errors, res_info, low_fields, 
                    protons,intensities, intensity_errors))

            elif residual_type == 'hf':
                # do fit, here with the default leastsq algorithm
                fcn_args = (*hf_data, *hf_errors, res_info)
                minner = Minimizer(self.residual_hf, params, fcn_args=(fcn_args))

            elif residual_type == 'lf':
                print('This has not been implimented yet ...')
                os.exit()

            else:
                print('This residual_type is not recognised.')

            start_time = time.time()
            result = minner.minimize(method='powel')
            resdict = result.params.valuesdict()
            end_time = time.time()
            elapsed_time = end_time - start_time
            all_models.append(result)

            if result.bic < bics[i]:
                model = result

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
        protons=(),
        residual_type='hflf'):

        '''
        This funtion fits a spectral density function to each residue using the
        extended model free formalism. Since Each residue is treated individually
        the diffusion tensor is held as fixed. 
        '''

        c1p_keys = self.get_c1p_keys()
        data = {}
        models = {}

        if cpu_count <= 0 :
            cpu_count = mp.cpu_count()


        args = c1p_keys
        kwargs = {}
        kwargs['protons'] = protons
        kwargs['residual_type'] = residual_type
        total_args = [(i, kwargs) for i in args]

        #print(total_args)
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
        
        with open(self.fit_results_pickle_file, 'wb') as handle:
            pic.dump(models, handle)


    def wrapper_global_residual(self, args):
        
        i, current_diffusion, residual_type = args
        #print('in wrapper args', args)
        
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
        residual_type='hf'):

        '''
        This funtion fits a spectral density function to each residue using the
        extended model free formalism. Since Each residue is treated individually
        the diffusion tensor is held as fixed. 
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
            #print('argsa', args[0])
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
                for i in args:
                    res = self.wrapper_global_residual(i)
                    models[res[0]] = res[1] 
            else:

                with mp.Pool(cpu_count) as pool:
                    #print('im here', cpu_count)
                    results = pool.map(self.wrapper_global_residual, args)

                for i in results:
                    models[i[0]] = i[1]

            # blend the residuals for the models based on the aics
            best_aic = 10e100

            # take the residuals of the best model
            for i in models:
                # get all the AICS
                for mod in models[i]:
                    if mod.bic < best_aic:
                        best_aic = mod.aic
                        current_resids = mod.residual
                resids.append(current_resids)

                # aics = [mod.aic for mod in models[i]]
                # min_aic = min(aics)

                # for mod in models[i]:
                #     #this 
                #     probability_of_min_info_loss = np.e**((min_aic-mod.aic)/2)
                #     weighted_residuals = probability_of_min_info_loss*mod.residual
                #     resids.append(weighted_residuals)

            return np.array(resids)

        # sanity check 
        if self.scale_diffusion==True:
            print('This will not work with self.scale_diffusion=True')
            print('because we update the diffusion tensor in the outer and inner fits')

        # set up the Parameters object
        params = Parameters()
        print(self.diffusion)
        params.add('dx_fix',  min=0, value=self.diffusion[0], vary=False)
        params.add('dy_fix',  min=0, value=self.diffusion[1], vary=False)
        params.add('dz_fix',  min=0, value=self.diffusion[2], vary=False)

        #scale the diffusion tensor by a single value 
        params.add('diff_scale', min=0, value=1, vary=True)
        params.add('dx',  expr='dx_fix*diff_scale')
        params.add('dy',  expr='dy_fix*diff_scale')
        params.add('dz',  expr='dz_fix*diff_scale')

        args = [cpu_count]
        print(params)
        minner = Minimizer(global_residual, params, fcn_args=args)
        result = minner.minimize(method='powel')
        report_fit(result)


    def print_final_residuals(self, pickle_path='default'):

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
            protons = ("H1'", "H2'", "H2''")

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

    def plot_model_free_parameters(self, atom_name, model_pic='model_free_parameters.pic', plot_name='model_free_params.pdf', showplot=False):

        # load the model
        models, models_resid, sorted_keys, model_resinfo = self.read_pic_for_plotting(model_pic, atom_name)
        model_array = []
        model_err_array = []

        print('sorted keys:', sorted_keys)

        for i in sorted_keys:
            values = [models_resid[i].params[j].value for j in models_resid[i].params]
            stds = [models_resid[i].params[j].stderr for j in models_resid[i].params]
            model_array.append(values)
            model_err_array.append(stds)

        model_array = np.array(model_array)
        model_err_array = np.array(model_err_array)
        
        #model_err_array[np.isnan(model_err_array)] = 0
        model_err_array[model_err_array==None]=0


        number_of_params = model_err_array.shape[1]
        param_names = [j for j in models_resid[i].params]

        fig_width = math.ceil(number_of_params/3)
        fig, ax = plt.subplots(nrows=3, ncols=fig_width, figsize=(3*fig_width, 9))
        ax = ax.flatten()
        for i in range(number_of_params):

            ax[i].set_ylabel(param_names[i])
            ax[i].set_xlabel('residue')
            ax[i].errorbar(sorted_keys, model_array.T[i], yerr=model_err_array.T[i],fmt='o' )
            
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
        per_residue_prefix='hf_rates'):

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

            r1_exp_array.append(self.r1[tag])
            r2_exp_array.append(self.r2[tag])
            noe_array.append(self.hetnoe[tag])

            r1_exp_array_err.append(self.r1_err[tag])
            r2_exp_array_err.append(self.r2_err[tag])
            noe_array_err.append(self.hetnoe_err[tag])

            r1_model.append(model_r1_v2)
            r2_model.append(model_r2_v2)
            noe_model.append(model_noe_v2)

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
                plt.errorbar(delays, exp_intensity_scalled, yerr=exp_error_scalled,fmt='|', 
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

        models, models_resid, sorted_keys, model_resinfo = self.read_pic_for_plotting(model_pic, atom_name)
        low_fields = sorted(self.ShuttleTrajectory.experiment_info.keys())
        print('plotting relaxometry_intensities')

        norm = matplotlib.colors.Normalize(0, 10, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap)

        for i in tqdm(sorted_keys):

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

            #ax1.set_title('$R_{2}$')
            ax1.errorbar(self.high_fields, self.r2[tag],yerr=self.r2_err[tag], fmt='|', zorder=2,)
            ax1.scatter(self.high_fields, self.r2[tag], edgecolor='black', zorder=3, s=marker_size)
            ax1.plot(self.model_high_fields, model_r2, zorder=1)
            ax1.set_ylim(*plot_ranges[1])
            ax1.set_xlabel('Field (T)', fontsize=text_size)
            ax1.set_ylabel('$R_{2}$ (Hz)', fontsize=text_size)
            ax1.tick_params(axis='both', labelsize=tick_size)

            #ax2.set_title('$I_{sat}/I_{0}$')
            ax2.errorbar(self.high_fields, self.hetnoe[tag],yerr=self.hetnoe_err[tag], fmt='|', zorder=2)
            ax2.scatter(self.high_fields, self.hetnoe[tag], edgecolor='black', zorder=3, s=marker_size)
            ax2.plot(self.model_high_fields, model_noe, zorder=1)
            ax2.set_ylim(*plot_ranges[2])
            ax2.set_ylabel('$I_{sat}/I_{0}$', fontsize=text_size)
            ax2.set_xlabel('Field (T)', fontsize=text_size)
            ax2.tick_params(axis='both', labelsize=tick_size)


            for f in low_fields:

                # high field part 


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
                ax3.errorbar(delays, exp_intensity_scalled, yerr=exp_error_scalled,fmt='|', c=color,zorder=2)
                ax3.scatter(delays, exp_intensity_scalled, color=color, label=f"{f:0.1f} T",edgecolor='black',zorder=3, s=marker_size)

            #ax3.set_title('$Relaxometry$')
            ax3.set_ylabel('Intensity', fontsize=text_size)
            ax3.set_xlabel('delay (s)', fontsize=text_size)
            ax3.legend(fontsize=tick_size)
            ax3.tick_params(axis='both', labelsize=tick_size)
            plt.tight_layout()
            plt.savefig(f'{folder}_{tag}_together.pdf')
            plt.close()

    def plot_relaxometry_rates(self,
        atom_name,
        folder='relaxometry_plots',
        model_pic='model_free_parameters.pic', 
        fancy=False,
        marker_size=15,
        text_size = 10,
        tick_size=8):

        if fancy == True:
            plt.style.use(['science','nature'])

        try:
            os.mkdir(folder)
        except FileExistsError:
            pass 

        models, models_resid, sorted_keys, model_resinfo = self.read_pic_for_plotting(model_pic, atom_name)
        low_fields = sorted(self.ShuttleTrajectory.experiment_info.keys())
        print('plotting relaxometry_intensities')

        for i in tqdm(sorted_keys):

            tag = utils.resinto_to_tag(*model_resinfo[i])
            res_info = model_resinfo[i]
            model_r1, model_r2, model_noe = self.model_hf(models_resid[i].params,res_info, self.model_all_fields)
            
            fig, (ax0) = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))

            ax0.plot(self.model_all_fields, model_r1, zorder=1, c='C2', label='Simulated $R_1$')
            values = self.relaxometry_mono_exp_fits[tag]
            err = self.relaxometry_mono_exp_fits_err[tag]

            ax0.set_title(tag)
            ax0.errorbar(self.low_fields, values,yerr=err, fmt='|', zorder=2, c='C1')
            ax0.scatter(self.low_fields, values, edgecolor='black', zorder=3, s=marker_size, c='C1', label='Low field $R_{1}^{apparent}$')

            ax0.errorbar(self.high_fields, self.r1[tag],yerr=self.r1_err[tag], fmt='|', zorder=2)
            ax0.scatter(self.high_fields, self.r1[tag], edgecolor='black', zorder=3, s=marker_size, c='C3', label='High Field $R_1$')

            ax0.set_xlabel('Field (T)', fontsize=text_size)
            ax0.set_ylabel('$R_{1}$ (Hz)', fontsize=text_size)
            ax0.tick_params(axis='both', labelsize=tick_size)
            ax0.set_xscale('log')
            ax0.set_xlim(1, 25)
            ax0.legend()

            plt.tight_layout()
            plt.savefig(f'{folder}_{tag}_together.pdf')
            plt.close()

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

def generate_mf_parameters(dx,dy,dz,scale_diffusion=False, diffusion=None):


    # models 


    models_state = [[True, True,  False, False],
                    [True, True,  True,  False],
                    [True, True,  True,  True],]

    params_lists = []
    if scale_diffusion == False:
        fit_diff = [False]
    elif scale_diffusion == True:
        fit_diff = [False, True]

    for i in fit_diff:
        for mod in models_state:
            params = Parameters()

            if mod[0] == True:
                params.add('Ss', value=1., min=0.4, max=1., vary=True)

            if mod[1] == True:
                params.add('tau_s', value=0.5e-9, vary=True, max=15e-9, min=500e-12)
            else:
                params.add('tau_s', value=0, vary=False)

            if mod[2] == True:
                params.add('Sf', value=0.9, min=0.4, vary=True, max=1)
            else:
                params.add('Sf', value=0, vary=False)


            if mod[3] == True:
                params.add('tau_f', value=100e-12, min=40e-12, vary=True,max=500e-12)
                #this should ensure that tauf is 5 times smaller than taus 
                params.add('diff', min=0, expr='tau_s-tau_f')
            else:
                params.add('tau_f', value=0, vary=False)

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

    return params_lists
