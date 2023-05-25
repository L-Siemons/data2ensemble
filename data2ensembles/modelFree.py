
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

from lmfit import Minimizer, Parameters, report_fit
from tqdm import tqdm
import enlighten

import matplotlib.pyplot as plt
import sys

PhysQ = utils.PhysicalQuantities()


print(data2ensembles.rates.__file__)

class ModelFree():
    """This is class that contains functions for analysing NMR relaxation rates
    using model free approaches"""

    def __init__(self,):

        self.r1_path = None
        self.r2_path = None 
        self.hetnoe_path = None 
        self.r1_err_path = None
        self.r2_err_path = None 
        self.hetnoe_err_path = None 
        self.path_prefix = 'model'
        self.PhysQ = PhysQ

        fields = None
        self.diffusion_tensor = None
        self.pdb = None
        self.high_fields = None
        self.model_high_fields = None
        self.spectral_density = specDens.J_anisotropic_emf


    def load_data(self):

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

    def load_low_field_info(self, file, distance_incrementing=0.5):
        self.ShuttleTrajectory = shuttling.ShuttleTrajectory(file, distance_incrementing = distance_incrementing)
        self.ShuttleTrajectory.construct_all_trajectories()

    def calc_cosine_angles_and_distances(self, atom_names, dist_skip=1):
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

    def get_atom_info_from_tag(self, i):
        #need to fix this
        return utils.get_atom_info_from_tag(i)

    def get_c1p_keys(self,):
        c1p_keys = []
        for i in self.r1:
            if "C1'" in i:
                c1p_keys.append(i)
        return c1p_keys

    def load_low_field_data(self, prefix):
        
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
                    self.low_field_delays[f][key] = delays

    def read_pic_for_plotting(self,model_pic, atom_name):
        # load the model
        models = utils.load_pickle(model_pic)
        models_resid, model_resinfo = utils.dict_with_full_keys_to_resid_keys(models, atom_name)
        sorted_keys = sorted(list(models_resid.keys()))
        return models, models_resid, sorted_keys, model_resinfo

    def model_hf(self, params,res_info, fields):
        '''
        This is is the model for the R1, R2 and HetNOE data 
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

    def per_residue_emf_fit(self, 
        lf_data_prefix='', 
        lf_data_suffix='', 
        scale_diffusion=False, 
        select_only_some_models=False, 
        fit_results_pickle_file='model_free_parameters.pic'):

        '''
        This funtion fits a spectral density function to each residue using the
        extended model free formalism. Since Each residue is treated individually
        the diffusion tensor is held as fixed. 
        '''

        def divide_by_error(a,b,c):
            return (a-b)/c

        def model_lf_single_field_intensities(params, low_field, res_info, protons):

            # I wanted to profile this function 
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
            
                        
            full_propergators = [stablaization_operator.dot(backwards_operators).dot(i).dot(forwrds_propergators) for i in delay_propergators]

            #print(len(full_propergators), full_propergators[0].shape)
            experimets = np.zeros(operator_size)
            experimets[1] = 1.
            intensities = [np.dot(i, experimets) for i in full_propergators]
            
            return intensities

        def residual_hf(params, r1, r2, hetnoe, r1_err, r2_err, noe_err, res_info):
            
            model_r1, model_r2, model_noe = self.model_hf(params,res_info, self.high_fields)
            r1_diff = divide_by_error(r1, model_r1,r1_err)
            r2_diff = divide_by_error(r2, model_r2,r2_err)
            noe_diff = divide_by_error(hetnoe, model_noe,noe_err)

            constraint = 0
            if params['tau_s'] < params['tau_f']:
                constraint = (params['tau_s'] - params['tau_f'])*1e12

            return np.concatenate([r1_diff.flatten(), r2_diff.flatten(), noe_diff.flatten(), [constraint]])

        def residual_lf(params, low_field, resinfo, protons, intensities, intensity_errors):

            diffs = []
            if type(params) is not dict:
                params = params.valuesdict()
            
            for field, exp_intensity, exp_error in zip(low_fields, intensities, intensity_errors):
                
                intensities = model_lf_single_field_intensities(params, field, resinfo, protons)

                # the model returns back all the populations 
                intensities = np.array([i[1] for i in intensities])
                
                # here we need a prefactor to scale our calculated intensities since we do not 
                # know what the zero point is. 

                factor = np.mean(exp_intensity/intensities)
                diff = (factor*intensities - exp_intensity)/(exp_error*(exp_intensity))
                diffs = diffs + diff.tolist()

            return diffs

        def residual_hflf(params, hf_data, hf_error, res_info, low_fields, protons,intensities, intensity_errors):

            # high field data
            r1, r2, hetnoe = hf_data
            r1_err, r2_err, noe_err = hf_error

            # for i in range(50):
            hf_residual = residual_hf(params, r1, r2, hetnoe, r1_err, r2_err, noe_err, res_info)
            # return hf_residual

            for i in range(10):
                lf_residual = residual_lf(params, low_fields, res_info, protons, intensities, intensity_errors)
            sys.exit()
            return np.concatenate([hf_residual, lf_residual])

        c1p_keys = self.get_c1p_keys()

        data = {}
        models = {}
        models_err = {}
        bics = {} 
        for i in tqdm(c1p_keys):

            print(i)

            # get the atom info
            res_info = utils.get_atom_info_from_tag(i)
            resid, resname, atom1, atom2 = res_info
            params = Parameters()
            bics[i] = 10e10

            # the rows are Sf, Ss, tauf, taus, might be best to take this out and put it in its own function
            if scale_diffusion == False:
                models_state = [[True, 1, 0, 0, False],
                          [True, 1, True, 0, False], 
                          [1, True, 0, True, False], 
                          [True, True, 0, True, False],
                          [True, True, True, True, False]]

            if scale_diffusion == True:
                models_state = [[True, 1, 0, 0, False],
                          [True, 1, True, 0, False], 
                          [1, True, 0, True, False], 
                          [True, True, 0, True, False],
                          [True, True, True, True, False],
                          [True, 1, 0, 0, False],
                          [True, 1, True, 0, True], 
                          [1, True, 0, True, True], 
                          [True, True, 0, True, True],
                          [True, True, True, True, True]]

            for mod in models_state:

                #internal dynamics 
                if mod[3] == True:
                    params.add('tau_s', value=0.5e-9, vary=True, max=10e-9)
                else:
                    params.add('tau_s', value=mod[3], vary=False)

                if mod[1] == True:
                    params.add('Ss', value=1, min=0, vary=True, max=1)
                else:
                    params.add('Ss', value=mod[1], vary=False,)

                # add a constraint on the diffeerence between tauf and taus
                if mod[2] == True:
                    params.add('tau_f', value=50e-12, min=40e-12, vary=True)
                else: 
                    params.add('tau_f', value=mod[2], vary=False)

                if mod[3] == True and mod[2] == True:
                    params.add('diff', max=0, expr='tau_f*5-tau_s')
                    

                #Sf 
                if mod[0] == True:
                    params.add('Sf', value=1, min=0, vary=True, max=1)
                else:
                    params.add('Sf', value=mod[0], vary=False,)

                #diffusion
                params.add('dx_fix',  min=0, value=22689512.7513627, vary=False)
                params.add('dy_fix',  min=0, value=21652874.50933672, vary=False)
                params.add('dz_fix',  min=0, value=38050175.186921805, vary=False)
                params.add('diff_scale', min=0, value=1, vary=mod[4])

                params.add('dx',  expr='dx_fix*diff_scale')
                params.add('dy',  expr='dy_fix*diff_scale')
                params.add('dz',  expr='dz_fix*diff_scale')

                #angles = self.cosine_angles[(resid, atom1, resid, atom2)]
                
                # do fit, here with the default leastsq algorithm
                # minner = Minimizer(residual_hf, params, fcn_args=(self.r1[i], self.r2[i], self.hetnoe[i],
                #  self.r1_err[i], self.r2_err[i], self.hetnoe_err[i], res_info))

                resid, resname, atom1, atom2 = res_info
                key = (resid, atom1 , resid, atom2)
                # args = (np.array([23,16])*self.PhysQ.gamma['c'],self.cosine_angles[key][0], self.cosine_angles[key][1], self.cosine_angles[key][2], )
                
                hf_data = (self.r1[i], self.r2[i], self.hetnoe[i])
                hf_errors = (self.r1_err[i], self.r2_err[i], self.hetnoe_err[i])
                protons = ("H1'", "H2'", "H2''")

                low_fields = list(self.ShuttleTrajectory.experiment_info.keys())
                intensities = []
                intensity_errors = []

                for f in low_fields:
                    intensities.append(self.low_field_intensities[f][i])
                    intensity_errors.append(self.low_field_intensity_errors[f][i])

                # do fit, here with the default leastsq algorithm
                minner = Minimizer(residual_hflf, params, fcn_args=(hf_data, 
                    hf_errors, res_info, low_fields, 
                    protons,intensities, intensity_errors))

                start_time = time.time()
                result = minner.minimize(method='powel')
                resdict = result.params.valuesdict()
                end_time = time.time()
                elapsed_time = end_time - start_time
                print("Elapsed time: ", elapsed_time) 
                report_fit(result)

                if result.bic < bics[i]:
                    models[i] = result

        # save the file
        with open(fit_results_pickle_file, 'wb') as handle:
            pic.dump(models, handle)

    def plot_model_free_parameters(self, atom_name, model_pic='model_free_parameters.pic', plot_name='model_free_params.pdf'):

        # load the model
        models, models_resid, sorted_keys, model_resinfo = self.read_pic_for_plotting(model_pic, atom_name)
        model_array = []
        model_err_array = []

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
            ax[i].errorbar(sorted_keys, model_array.T[i], yerr=model_err_array.T[i], fmt='o')
            
            if 'tau' in param_names[i]:
                ax[i].set_yscale('log')

            if "S" in param_names[i]:
                ax[i].set_ylim(0,1.1)
        plt.tight_layout()
        plt.savefig(plot_name)
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





