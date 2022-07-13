
from .trrAnalysis import *
from .dataToStructures import *
from .spectralDensity import *
from .utils import *
from .rates import *
import numpy as np

# class ExperimentalData():
#     """A class for reading in the experimental data
#     Add experimental restraints with the 'read_x() functions)'

#     """
#     def __init__(self):
#         self.data = {}
#         self.data_headers = {}
#         self.uncertainty = {}
#         pass

#     def read_nmr_relaxation_rate(self, file, name):
#         '''
#         Read in NMR relaxation data
#         '''
#         print('Reading in data from {file}')
#         self.data[name] = {}
#         header_check = False

#         #start reading the file
#         f = open(file)
#         for i in f.readlines():
#             if i.split()[0][0] != '#':
#                 s = i.split()
#                 self.data[name][s[0]] = s[1:]
#             else:
#                 self.data_headers[name] = i
#                 header_check = True

#         # good to label your files correctly!
#         if header_check == False:
#             raise NoHeader(f'The file {file} does not have a header!')

#         f.close()

#     def print_data(self):

#         for i in self.data:
#             print(f'Data type {i}')
#             self.data_headers[i]
#             for j in self.data[i]:
#                 print(' '.join(self.data[i][j]))

# class CandidateData():
#     """CandidateData loads in the data for each candidate structure/ensemble.

#     This class is very similar to the ExperimentalData class but uses candidate ids to
#     keep track of each of the candidates. The ids need to be provided when the class is initialized.

#     In general these functions will expect the data files to be named {prefix}id{suffix}. By default the
#     suffix is necessary but the prefix is optional. In addition to this the name matching the experimental
#     data name needs to be provided.


#     ids : str - A list of candidate ids.
#     """
#     def __init__(self, candidate_ids):
#         self.candidate_ids = candidate_ids
#         self.candidate_ids_str = [str(i) for i in candidate_ids]
#         self.data = {}

#     def read_nmr_relaxation_rate(self, name, suffix, prefix=''):

#         for id_ in self.candidate_ids_str:
#             if id_ not in self.data:
#                 self.data[id_] = {}
#                 self.data[id_][name] = {}

#             f = open(f'{prefix}{id_}{suffix}')

#             for line in f.readlines():

#                 s = line.split()
#                 if s[0][0] != '#':
#                     self.data[id_][name][s[0]] = s[1:]



#     def print_data(self):

#         for i in self.data:
#             print(f'Candidate {i}')
#             for j in self.data[i]:
#                 print(f'experiment {j}')
#                 for k in self.data[i][j]:
#                     print(' '.join(self.data[i][j][k]) )

# class FitData(object):
#     """This Class fits the candidate data to the experimental data.

#     There are in principle two approaches available.

#     1) N chunks are selected so that the average best represents the data
#     2) A maximum entropy approach where the the smallest deviation from the
#        candidates needed to model the distribution is used. """
#     def __init__(self, ExperimentalData, CandidateData):
#         self.experimental_data = ExperimentalData.data
#         self.candidate_data = CandidateData.data
#         self.candidate_ids = CandidateData.candidate_ids_str

#         self.experimental_array = []
#         self.candidate_dict = {}
#         self.candidate_arrays = []

#         for exp in self.experimental_data:
#             for residue in self.experimental_data[exp]:
#                 mask = []

#                 for item in self.experimental_data[exp][residue]:
#                     if item == '-':
#                         item_mask_status = False
#                     else:

#                         try:
#                             a = float(item)
#                             item_mask_status = True
#                         except ValueError:
#                             print(f'The entry in {exp}, {residue} is not a float or a \'-\'')
#                             sys.exit()

#                     mask.append(item_mask_status)

#                 curr_exp_array = np.array(self.experimental_data[exp][residue], )[mask].astype(float)
#                 self.experimental_array.append(curr_exp_array)

#                 for id_ in self.candidate_ids:
#                     curr_candidate_array = np.array(self.candidate_data[id_][exp][residue])[mask].astype(float)
#                     if id_ in self.candidate_dict:
#                         self.candidate_dict[id_].append(curr_candidate_array)
#                     else:
#                         self.candidate_dict[id_] = [curr_candidate_array]

#         # here we just turn all the candidates in the dictionary into a list because it
#         # will be faster later on!

#         for i in self.candidate_dict:
#             self.candidate_arrays.append(self.candidate_dict[i])

#     def chi_square(self,exp, obs, error=1):

#         # print(exp)
#         # print(obs)
#         difference = exp-obs
#         elements = (difference**2)/error
#         return np.sum(elements)


#     def monte_carlo(self, candidate_count=3,mc_steps=1000,
#         temp_range_type='linear', temp_max=400, temp_min=0.,
#         threshold=10e-10):


#         def average_candicates(candidate_data, index_list):

#             #print([candidate_data[i] for i in index_list])

#             total = np.mean([candidate_data[i] for i in index_list], axis=0)
#             return total

#         # make the data flat.
#         exp_data = np.concatenate(self.experimental_array)
#         candidate_data = [np.concatenate(i) for i in self.candidate_arrays]

#         #set up some values
#         candidate_pool_size = len(candidate_data)

#         #make the temperature range
#         if temp_range_type == 'linear':
#             temp_range = np.linspace(temp_max, temp_min, mc_steps)

#         before_list = r.sample(range(0, candidate_pool_size), candidate_count)
#         before_average = average_candicates(candidate_data, before_list)
#         before_chi_square = self.chi_square(exp_data, before_average)


#         for step, temp in tqdm(zip(range(mc_steps), temp_range)):

#             index_to_change = r.randint(0,candidate_count-1)
#             #print(index_to_change, before_list, len(candidate_data))

#             a = None

#             while a == None:
#                 test = r.randint(0,candidate_pool_size-1)
#                 if test not in before_list:
#                     a = test

#             after_list = copy.deepcopy(before_list)
#             after_list[index_to_change] = a
#             #print('after_list', after_list)

#             after_average = average_candicates(candidate_data, after_list)
#             after_chi_square = self.chi_square(exp_data, after_average)
#             #print(before_chi_square, after_chi_square)

#             pass_probability = np.e**((before_chi_square - after_chi_square)/temp)
#             update_list = False

#             if pass_probability >= 1.:
#                 update_list = True
#             else:
#                 random_selector = r.random()
#                 if random_selector < pass_probability:
#                     update_list = True

#             if update_list == True:
#                 before_list = copy.deepcopy(after_list)
#                 before_chi_square = after_chi_square

#             if before_chi_square <= threshold:
#                 print('The accuracy threshold has been reached - breaking the loop!')
#                 print(f'The best chi square is {before_chi_square}')
#                 break


# class AnalyseTrr():
#     """A class to calculate properties of the trajectory note this
#     class requires gromacs and MDAnalysis"""
#     def __init__(self, tpr, gro, path_prefix):

#         self.gmx = 'gmx'
#         self.tpr = tpr
#         self.xtc = None
#         self.gro = gro
#         self.path_prefix = path_prefix


#         # the number of curves to use when fitting the internal correlation functions
#         self.curve_count = 3
#         # This is a dummy correlation time used to help fit the internal correlation times
#         # note that this should be seconds
#         self.dummy_tauc = 5e-6

#     def load_trr(self,):

#         if self.xtc == None:
#             uni = md.Universe(self.gro)
#         elif self.xtc != None and self.tpr != None:
#             uni = md.Universe(self.gro, self.xtc)
#         else:
#             print('Something went wrong ... not sure what to load in')
#         self.uni = uni

#     def make_ndx_file(self, atom_selection_pairs, index_name,supress=True):

#         indx = open(index_name,'w')
#         indx.write('[indx]\n')
#         for i,j in atom_selection_pairs:

#             a = self.uni.select_atoms(i)
#             b = self.uni.select_atoms(j)

#             #python numbering!
#             a_indx = a[0].ix+1
#             b_indx = b[0].ix+1

#             indx.write(f'{a_indx} {b_indx}\n')
#         indx.close()

#     def get_number_of_blocks(self,atom_names):

#         #get the number of blocks that have residue 1 and the atom names in them
#         # we subtract 1 because often the last block is not complete so we do not want to analyze it

#         atom1 = atom_names[0]
#         atom2 = atom_names[1]

#         files = glob.glob(f"{self.path_prefix}_rotacf/rotacf_block_*_1_{atom1}_1_{atom2}.xvg")
#         return len(files) - 1

#     def read_gmx_xvg(self,f):

#         x = []
#         y = []
#         #print('opening file ...')
#         fi = open(f)
#         for i in fi.readlines():
#             s = i.split()
#             x.append(float(s[0]))
#             y.append(float(s[1]))

#             # if maxtime != None:
#             #     if float(s[0]) > maxtime:
#             #         return np.array(x), np.array(y)
#         fi.close()
#         return np.array(x), np.array(y)

#     def calc_rotacf(self, indx_file, out_name, atom_selection_pairs, b=None, e=None, dt=None):
#         command = f'{self.gmx} rotacf -s {self.tpr} -f {self.xtc} -o {out_name}.xvg -n {indx_file} -P 2 -d -noaver -xvg none'
#         print('>>>>' + command)
#         if b != None:
#             command = command + f' -b {b} '
#         if e != None:
#             command = command + f' -e {e} '
#         if dt != None:
#             command = command + f' -dt {dt}'
#         print(f'Running the command: \n {command}')
#         os.system(command)

#         xvg = open(out_name+'.xvg')
#         time = []
#         total = []

#         data = []
#         time_check = False
#         for i in xvg.readlines():
#             s = i.split()


#             if s[0] == '&':
#                 check = True
#                 total.append(data)
#                 data = []
#             else:

#                 data.append(float(s[1]))

#                 if time_check == False:
#                     time.append(float(s[0]))

#         xvg.close()
#         total = np.array(total)

#         for indx, i in enumerate(atom_selection_pairs):
#             s0 = i[0].split()
#             s1 = i[1].split()
#             f = open(out_name+f'_{s0[1]}_{s0[4]}_{s1[1]}_{s1[4]}.xvg', 'w')

#             for ti, coeffi in zip(time, total[indx]):
#                 f.write(f'{ti} {coeffi}\n')

#             f.close()
#     def calc_rotacf_segments(self,indx_file, atom_selection_pairs, seg_blocks):

#         try:
#             os.mkdir(f'{self.path_prefix}_rotacf')
#         except FileExistsError:
#             pass

#         for indx, i in enumerate(seg_blocks):
#             out_name_curr = self.path_prefix+'_rotacf/rotacf' + f'_block_{indx}'
#             self.calc_rotacf(indx_file, out_name_curr, atom_selection_pairs, b=i[0], e=i[1])

#     def correlation_function(self, Params, x):

#         '''
#         An internal correlation function. This is the same one that is used
#         by Kresten in the absurder paper.
#         '''

#         total = Params['S_long']
#         for i in range(self.curve_count):
#             i = i+1
#             amp = 'amp_%i'%(i)
#             time = 'time_%i'%(i)

#             total = total + Params[amp]*(np.e**(-1*x/Params[time]))

#         return total

#     def fit_correlation_function(self,x,y, log_time_min=50, log_time_max=5000):
#         '''
#         This function fits an internal correlation function and also the corresponding spectral density
#         function. To do this we assume that the there is an overall isotropic tumbling so we can use a
#         a spectral desnity function we know. In effect this just acts as a numerical trick to help
#         speed up the fitting.

#         Note that the units of time here are those used in the gromacs rotacf. This is generally ps
#         '''

#         # here I have manually defined some variables
#         # because It didnt seem to want to play well with the Lmfit classes

#         curve_count = copy.deepcopy(self.curve_count)
#         dummy_tauc = copy.deepcopy(self.dummy_tauc)
#         correlation_function = self.correlation_function

#         def calc_jw(time, internal_corelations,):

#             '''
#             This function is used to calculate an numerical spectral density function
#             here we assume an overall isotropic tumbling. In reality this is just a way to focus the fitting
#             on a region of interest.

#             Here the spectral density is calculated numerically
#             '''

#             dt = time[1]-time[0]
#             tumbling = 0.2*(np.e**(-time/dummy_tauc))
#             total = tumbling * internal_corelations

#             j_fft = scipy.fft.fft(total)*dt*2
#             j_fft = scipy.fft.fftshift(j_fft)
#             j_fft_freq = scipy.fft.fftfreq(len(time), d=dt)
#             j_fft_freq = scipy.fft.fftshift(j_fft_freq)
#             j_fft_freq = np.pi*j_fft_freq*2
#             return j_fft_freq, j_fft

#         def spectral_density_fuction(params, omega):

#             total = "1"
#             for i in range(curve_count):
#                 i = i + 1

#                 #correction for fitting the correlation times in ps
#                 taui = params['time_%i'%(i)]
#                 ampi = params['amp_%i'%(i)]


#                 tau_eff = taui*dummy_tauc/(dummy_tauc+taui)
#                 term1 = ampi*tau_eff/(1+(omega*tau_eff)**2)
#                 term2 = params['S_long']*dummy_tauc/(1+(omega*dummy_tauc)**2)

#                 if type(total) == str:
#                     total = term1 + term2
#                 else:
#                     total = total + term1

#             return (2/5.)*total


#         def func2min(Params, x,y):

#             '''
#             The function we want to minimize. The fitting is carried out both in s and hz.
#             '''

#             correlation_diff =  y - correlation_function(Params, x)

#             x = x
#             dt = x[1]-x[0]
#             tumbling = 0.2*(np.e**(-x/dummy_tauc))
#             total = tumbling * y

#             j_fft = scipy.fft.fft(total)*dt*2
#             j_fft = scipy.fft.fftshift(j_fft)
#             j_fft = j_fft.real
#             j_fft_freq = scipy.fft.fftfreq(len(x), d=dt)
#             j_fft_freq = scipy.fft.fftshift(j_fft_freq)
#             j_fft_freq = np.pi*j_fft_freq*2

#             spec_difference = j_fft - spectral_density_fuction(Params, j_fft_freq)

#             all_diffs = np.concatenate([spec_difference, correlation_diff])
#             return all_diffs.flatten()


#         # create a set of Parameters
#         Params = Parameters()
#         s_guess = np.mean(y[int(len(y)*0.75):])
#         Params.add('S_long', value=s_guess, min=0, max=1)

#         log_times = np.geomspace(log_time_min, log_time_max, curve_count)
#         amp_list = []

#         #here we generate the parameters
#         for i, taui in zip(range(self.curve_count-1), log_times):
#             i = i+1

#             # create a set of Parameters
#             Params.add('amp_%i'%(i), value=1/curve_count, min=0, max=1.)
#             amp_list.append('amp_%i'%(i))
#             Params.add('time_%i'%(i), value=taui, min=0)
#             i = 1+1

#         #these final parameters are used to ensure the sums are all correct
#         expression = '1 - (' + ' + '.join(amp_list) + ' + S_long)'
#         Params.add('amp_%i'%(curve_count), expr=expression, min=0, max=1.)
#         Params.add('time_%i'%(curve_count), value=log_times[-1], min=0)

#         # do fit, here with the default leastsq algorithm

#         minner = Minimizer(func2min, Params, fcn_args=(x, y))
#         result = minner.minimize()
#         #report_fit(result)
#         return result

#     def fit_all_correlation_functions(self,
#         atom_names, time_threshold=10000,
#         log_time_min=50, log_time_max=5000):

#         atom1 = atom_names[0]
#         atom2 = atom_names[1]

#         try:
#             os.mkdir(f'{self.path_prefix}_fits')
#         except FileExistsError:
#             pass

#         try:
#             os.mkdir(f'{self.path_prefix}_fits')
#         except FileExistsError:
#             pass

#         try:
#             os.mkdir(f'{self.path_prefix}_fit_params')
#         except FileExistsError:
#             pass


#         block_count = self.get_number_of_blocks(atom_names)
#         manager = enlighten.get_manager()
#         blocks_pbar = manager.counter(total=block_count, desc="Blocks", unit="Ticks", color="red")
#         residues_pbar = manager.counter(total=len(self.uni.residues)*block_count, desc="residues", unit="Tocks", color="blue")

#         for block in range(block_count):

#             params_out = open(f'{self.path_prefix}_fit_params/{block}.dat', 'w')
#             params_out.write('# residue:atoms:S2:times:amps\n')
#             for i in self.uni.residues:
#                 name = f"{self.path_prefix}_rotacf/rotacf_block_{block}_{i.resid}_{atom1}_{i.resid}_{atom2}.xvg"

#                 #Sometimes we dont want to fit the whole Ci(t) as it will often plateau. This time is in ps
#                 x,y = self.read_gmx_xvg(name)
#                 y = y[x<time_threshold]
#                 x = x[x<time_threshold]

#                 #print('starting fit ...')
#                 result = self.fit_correlation_function(x,y,)
#                 values = result.params.valuesdict()
#                 x_model = np.linspace(0, max(x), 10000)
#                 y_model = self.correlation_function(values, x_model)

#                 #print('plotting...')
#                 plt.plot(x,y, label='raw acf')
#                 plt.plot(x_model, y_model, label='fit')
#                 plt.legend()
#                 plt.savefig(f'{self.path_prefix}_fits/rot_acf_{block}_{i.resid}_{atom1}.pdf')
#                 plt.close()
#                 residues_pbar.update()

#                 amps = [str(values['amp_%i'%(i+1)]) for i in range(self.curve_count)]
#                 amps = ','.join(amps)
#                 times = [str(values['time_%i'%(i+1)]) for i in range(self.curve_count)]
#                 times = ','.join(times)
#                 slong = values['S_long']
#                 line = f'{i.resid}:{atom1},{atom2}:{slong}:{amps}:{times}\n'
#                 params_out.write(line)
#                 params_out.flush()

#             blocks_pbar.update()
#         manager.stop()