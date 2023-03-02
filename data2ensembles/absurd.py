
import data2ensembles.utils as utils
import data2ensembles as d2e
import numpy as np
import copy 
import random as r
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob

class AbsrudAnalysis():
    """A class to reweigth relaxation rates using absurd and absurder 
    methods"""
    def __init__(self,exp_rates_paths, exp_errors_paths, calc_rates_search_strings):
        self.exp_rates_paths = exp_rates_paths
        self.exp_errors_paths = exp_errors_paths
        self.calc_rates_search_strings = calc_rates_search_strings

    def load_data(self):

        self.exp_rates = {}
        self.exp_errors = {}
        self.calc_rates = {}
        self.calc_rate_files = {}

        iterators = zip(self.exp_rates_paths, self.exp_errors_paths, self.calc_rates_search_strings)
        
        for i,j, search_string in iterators:
            exp_rates_i, _ = utils.read_nmr_relaxation_rate(i)
            exp_errors_j, _ = utils.read_nmr_relaxation_rate(j)

            self.exp_rates[i] = exp_rates_i
            self.exp_errors[i] = exp_errors_j

            files = glob.glob(search_string)
            current_calc_rates = []

            for fi in files:
                calc_rate, _ = utils.read_nmr_relaxation_rate(fi)
                current_calc_rates.append(calc_rate)

            current_calc_rates = np.array(current_calc_rates)
            self.calc_rates[i] = current_calc_rates
            self.calc_rate_files[i] = files

    def calc_optimised_rates(self,optimisation_type):

        if optimisation_type == 'absurd':
            index_list = self.absurd_indexes
        else:
            print('The optimisation type is not registered here')
            os.exit()

        self.optimised_rates = {}
        for rates in self.exp_rates:
            self.optimised_rates[rates] = {}
            for atom_pair in self.exp_rates[rates]:
                if atom_pair in self.calc_rates[rates][0]:
                    collected_rates = []
                    for i in index_list:
                        #print(self.calc_rates[rates][atom_pair])
                        collected_rates.append(self.calc_rates[rates][i][atom_pair])

                    collected_rates = np.array(collected_rates)
                    collected_rates = np.mean(collected_rates,axis=0)
                    self.optimised_rates[rates][atom_pair] = collected_rates

    def write_optimised_rates(self, suffix='_opt.dat'):

        for rates in self.exp_rates:
            out_name = rates.split('/')[-1][:-4] + suffix
            f = open(out_name, 'w')
            f.write('#header bondName\n')
            for atom_pair in self.optimised_rates[rates]:
                f.write(atom_pair + ' ' + ' '.join(str(i) for i in self.optimised_rates[rates][atom_pair]) + '\n')
            f.close()

    def run_absurd(self, gene_length, mutation_chance=0.1, cycles=1e4):

        def target_function(index_list):
            total_diffs = []
            for rates in self.exp_rates:

                for atom_pair in self.exp_rates[rates]:
                    if atom_pair in self.calc_rates[rates][0]:
                        collected_rates = []
                        for i in index_list:
                            #print(self.calc_rates[rates][atom_pair])
                            collected_rates.append(self.calc_rates[rates][i][atom_pair])

                        collected_rates = np.array(collected_rates)
                        collected_rates = np.mean(collected_rates,axis=0)
                        diffs = (collected_rates-self.exp_rates[rates][atom_pair])**2/self.exp_errors[rates][atom_pair]
                        total_diffs.append(diffs)
            
            total_diffs = np.array(total_diffs)
            diffs_mean = np.mean(total_diffs)          
            return diffs_mean

        keys = list(self.calc_rate_files.keys())
        index_list = [i for i in range(len(self.calc_rate_files[keys[0]]))]
        
        prev_gene = r.sample(index_list, gene_length)
        prev_score = target_function(prev_gene)

        iteration_x = []
        iteration_y = []
        cycles = int(cycles)
        for iteration in range(cycles):

            main_list = list(set(index_list) - set(prev_gene))
            new_gene = copy.copy(prev_gene)

            for i in range(gene_length):
                mutation = r.random() < mutation_chance
                if mutation:
                    new_gene[i] = r.sample(main_list, 1)[0]

            new_score = target_function(new_gene)
            
            if new_score < prev_score:
                iteration_x.append(iteration)
                iteration_y.append(new_score)
            
                prev_score = copy.copy(new_score)
                prev_gene = copy.copy(new_gene)
                print(f'Iteration {iteration} score: {prev_score}')
        
        self.absurd_indexes = new_gene
        plt.plot(iteration_x, iteration_y)
        plt.show()


