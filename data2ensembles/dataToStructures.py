
from . import utils
import numpy as np
import sys
import random as r
import copy
from tqdm import tqdm
import MDAnalysis as md
from lmfit import Minimizer, Parameters, report_fit
import enlighten
import scipy
import scipy.fft
import scipy.interpolate
import glob
import matplotlib.pyplot as plt

class ExperimentalData():
    """A class for reading in the experimental data
    Add experimental restraints with the 'read_x() functions)'

    """
    def __init__(self):
        self.data = {}
        self.data_headers = {}
        self.uncertainty = {}
        pass

    def read_nmr_relaxation_rate(self, file, name):
        '''
        Read in NMR relaxation data
        '''

        data, header = utils.read_nmr_relaxation_rate(file)
        self.data[name] = data
        self.data_headers[name] = header

    def print_data(self):

        for i in self.data:
            print(f'Data type {i}')
            self.data_headers[i]
            for j in self.data[i]:
                print(' '.join(self.data[i][j]))

class CandidateData():
    """CandidateData loads in the data for each candidate structure/ensemble.

    This class is very similar to the ExperimentalData class but uses candidate ids to
    keep track of each of the candidates. The ids need to be provided when the class is initialized.

    In general these functions will expect the data files to be named {prefix}id{suffix}. By default the
    suffix is necessary but the prefix is optional. In addition to this the name matching the experimental
    data name needs to be provided.


    ids : str - A list of candidate ids.
    """
    def __init__(self, candidate_ids):
        self.candidate_ids = candidate_ids
        self.candidate_ids_str = [str(i) for i in candidate_ids]
        self.data = {}

    def read_nmr_relaxation_rate(self, name, suffix, prefix=''):

        for id_ in self.candidate_ids_str:
            if id_ not in self.data:
                self.data[id_] = {}
                self.data[id_][name] = {}

            f = open(f'{prefix}{id_}{suffix}')

            for line in f.readlines():

                s = line.split()
                if s[0][0] != '#':
                    self.data[id_][name][s[0]] = s[1:]


    def print_data(self):
        for i in self.data:
            print(f'Candidate {i}')
            for j in self.data[i]:
                print(f'experiment {j}')
                for k in self.data[i][j]:
                    print(' '.join(self.data[i][j][k]) )

class FitData(object):
    """This Class fits the candidate data to the experimental data.

    There are in principle two approaches available.

    1) N chunks are selected so that the average best represents the data
    2) A maximum entropy approach where the the smallest deviation from the
       candidates needed to model the distribution is used. """
    def __init__(self, ExperimentalData, CandidateData):
        self.experimental_data = ExperimentalData.data
        self.candidate_data = CandidateData.data
        self.candidate_ids = CandidateData.candidate_ids_str

        self.experimental_array = []
        self.candidate_dict = {}
        self.candidate_arrays = []

        for exp in self.experimental_data:
            for residue in self.experimental_data[exp]:
                mask = []

                for item in self.experimental_data[exp][residue]:
                    if item == '-':
                        item_mask_status = False
                    else:

                        try:
                            a = float(item)
                            item_mask_status = True
                        except ValueError:
                            print(f'The entry in {exp}, {residue} is not a float or a \'-\'')
                            sys.exit()

                    mask.append(item_mask_status)

                curr_exp_array = np.array(self.experimental_data[exp][residue], )[mask].astype(float)
                self.experimental_array.append(curr_exp_array)

                for id_ in self.candidate_ids:
                    curr_candidate_array = np.array(self.candidate_data[id_][exp][residue])[mask].astype(float)
                    if id_ in self.candidate_dict:
                        self.candidate_dict[id_].append(curr_candidate_array)
                    else:
                        self.candidate_dict[id_] = [curr_candidate_array]

        # here we just turn all the candidates in the dictionary into a list because it
        # will be faster later on!

        for i in self.candidate_dict:
            self.candidate_arrays.append(self.candidate_dict[i])

    def chi_square(self,exp, obs, error=1):

        # print(exp)
        # print(obs)
        difference = exp-obs
        elements = (difference**2)/error
        return np.sum(elements)


    def monte_carlo(self, candidate_count=3,mc_steps=1000,
        temp_range_type='linear', temp_max=400, temp_min=0.,
        threshold=10e-10):


        def average_candicates(candidate_data, index_list):

            #print([candidate_data[i] for i in index_list])

            total = np.mean([candidate_data[i] for i in index_list], axis=0)
            return total

        # make the data flat.
        exp_data = np.concatenate(self.experimental_array)
        candidate_data = [np.concatenate(i) for i in self.candidate_arrays]

        #set up some values
        candidate_pool_size = len(candidate_data)

        #make the temperature range
        if temp_range_type == 'linear':
            temp_range = np.linspace(temp_max, temp_min, mc_steps)

        before_list = r.sample(range(0, candidate_pool_size), candidate_count)
        before_average = average_candicates(candidate_data, before_list)
        before_chi_square = self.chi_square(exp_data, before_average)


        for step, temp in tqdm(zip(range(mc_steps), temp_range)):

            index_to_change = r.randint(0,candidate_count-1)
            #print(index_to_change, before_list, len(candidate_data))

            a = None

            while a == None:
                test = r.randint(0,candidate_pool_size-1)
                if test not in before_list:
                    a = test

            after_list = copy.deepcopy(before_list)
            after_list[index_to_change] = a
            #print('after_list', after_list)

            after_average = average_candicates(candidate_data, after_list)
            after_chi_square = self.chi_square(exp_data, after_average)
            #print(before_chi_square, after_chi_square)

            pass_probability = np.e**((before_chi_square - after_chi_square)/temp)
            update_list = False

            if pass_probability >= 1.:
                update_list = True
            else:
                random_selector = r.random()
                if random_selector < pass_probability:
                    update_list = True

            if update_list == True:
                before_list = copy.deepcopy(after_list)
                before_chi_square = after_chi_square

            if before_chi_square <= threshold:
                print('The accuracy threshold has been reached - breaking the loop!')
                print(f'The best chi square is {before_chi_square}')
                break
