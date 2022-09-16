

import data2ensembles as d2e
import data2ensembles.trrAnalysis
import numpy as np 
from multiprocessing import Pool

path = '/Users/lucassiemons/Desktop/important_stuff/Work_school_university/2021_dna_relaxometry/MD/duplex_simulation/amber14sb_parmbsc1/293.15/duplex_amber_293.15/unbiased_trrs/'
atom_names = [["C1'", "H1'"], ['C8', 'H8'], ['C5', 'H5'] ,['C6', 'H6'], ["C2'", "H2'"]]
#atom_names = [["C1'", "H1'"],]

def run(i):

	aligned_tag = 'amber_'+str(i)
	no_rot_tag = "amber_no_rotation_" +str(i)

	print('reading in fully aligned')


	fully_aligned = d2e.trrAnalysis.AnalyseTrr(path + 'run1.tpr', path + 'eq_dna_only.gro', aligned_tag)
	fully_aligned.curve_count = 6
	fully_aligned.xtc = path+f'rep_{i}_no_translation_rotation.xtc'

	fully_aligned.load_trr()
	fully_aligned.calc_cosine_angles(atom_names)
	#fully_aligned.make_ndx_file(atom_names, 'test.ndx')
	#fully_aligned.calc_rotacf('test.ndx', atom_names)

	print('reading in trans aligned')
	no_rot = d2e.trrAnalysis.AnalyseTrr(path + 'run1.tpr',path + 'eq_dna_only.gro', no_rot_tag)
	no_rot.xtc = path+f'rep_{i}_no_translation.xtc'
	no_rot.load_trr()
	#no_rot.make_ndx_file(atom_names, 'test.ndx')
	#no_rot.calc_rotacf('test.ndx', atom_names)

	print('getting dif tensor and later bits')
	#fully_aligned.extract_diffusion_rotacf(f'{aligned_tag}_rotacf', f'{no_rot_tag}_rotacf')
	print('fitting correlation funcs')
	#fully_aligned.fit_all_correlation_functions(atom_names, blocks=False)
	print('fitting dif tensor')
	#fully_aligned.fit_diffusion_tensor(atom_names, blocks=False)


	fully_aligned.fit_diffusion_to_r1_r2_hetnoe('../../../analysis/293_a2_dna_rates_17_08_2022/duplex_r1.dat', 
		'../../../analysis/293_a2_dna_rates_17_08_2022/duplex_r1_errors.dat', 
		'../../../analysis/293_a2_dna_rates_17_08_2022/duplex_r2_echo.dat', 
		'../../../analysis/293_a2_dna_rates_17_08_2022/duplex_r2_echo_errors.dat', 
		'../../../analysis/293_a2_dna_rates_17_08_2022/duplex_noe.dat', 
		'../../../analysis/293_a2_dna_rates_17_08_2022/duplex_noe_errors.dat'
		, f'{aligned_tag}_fit_params/internal_correlations_0.dat', 14.1,'c',error_filter=0.05,model='scale',) #scale_model='../hydro_nmr_diffusion_tensor.dat')# model='scale')

	# # fully_aligned.fit_diffusion_to_r1_r2_hetnoe('../../../analysis/293_a2_dna_rates_17_08_2022/duplex_r1.dat', 
	# 	'../../../analysis/293_a2_dna_rates_17_08_2022/duplex_r1_errors.dat', 
	# 	'../../../analysis/293_a2_dna_rates_17_08_2022/duplex_r2.dat', 
	# 	'../../../analysis/293_a2_dna_rates_17_08_2022/duplex_r2_errors.dat', 
	# 	'../../../analysis/293_a2_dna_rates_17_08_2022/duplex_reduced_noe.dat', 
	# 	'../../../analysis/293_a2_dna_rates_17_08_2022/duplex_reduced_noe_errors.dat'
	# 	, f'{aligned_tag}_fit_params/internal_correlations_0.dat', 14.1,'c', error_filter=0.05,model='scale', reduced_noe=True)

	# fully_aligned.calculate_r1_r2_hetnoe(atom_names, f'{aligned_tag}_diffusion_rotacf_fit/diffusion_tensor_fitted.dat'
	# 	, 14.1,'c', y='h', dna=True, write_out=True, prefix='fitted_')
    
	# fully_aligned.calculate_r1_r2_hetnoe(atom_names, f'../hydro_nmr_diffusion_tensor.dat'
	# 	, 14.1,'c', y='h', dna=True, write_out=True, prefix='fitted_')


	del fully_aligned
	#del no_rot


for i in [1,2,3,4,5]:#, 2,3,4,5]:
	run(i)

# if __name__ == '__main__':
#     with Pool(5) as p:
#         print(p.map(run, [2, 3,4,5,6]))