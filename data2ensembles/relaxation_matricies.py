from data2ensembles.utils import PhysicalQuantities
import data2ensembles.rates as rates

import numpy as np

PhysicalQuantities = PhysicalQuantities()

def c1_prime_relaxation_matrix(params, resid, 
		spectral_density, 
		fields, 
		rxy, 
		csa_name, 
		cos_ang, 
		csa_ang, 
		csa_params,
		PhysQ=PhysicalQuantities):

	'''
	this is the relaxation matrix for the C1' in DNA
	In this system we will consider 
	C1'
	H1'
	H2'
	H2''

	The interactions we will consider will be 

	auto relaxation 
	C1'z - Done 
	H1'z - Done 
	C1'zH1'z 

	H2'z
	H2''z

	NOes
	sC1'H1' - Done 
	H1'H2'
	H1'H2''

	
	
	The structure pf the matrix is

	0		0		0		0		0		0	 		0	 	0	 		0	 	0	
	0		pH1'	sC1'H1'	dH1'C1'	sH1pH2	0	 		sH1pH2	0	 		0	 	0
	0		sC1'H1'	pC1'	dC1'H1'	0		0	 		0	 	0	 		0	 	0
	0		dH1'C1'	dC1'H1'	pC1'H1'	0		sH1pH2 		0	 	sH1pH2		0	 	0
	0		sH1pH2	0		0		pH2'1	0	 		sH2H2	0	 		0	 	0		
	0		0		0		sH1pH2	0		PC1zH21z	0	 	0	 		0	 	0
	0		sH1pH2	0		0		sH2H2	0	 		pH2'2	0		 	0	 	0
	0		0		0		sH1pH2	0		0	 		0	 	PC1zH22z	0	 	0
	0		0		0		0		0		0	 		0	 	0	 		pH4' 	0
	0		0		0		0		0		0	 		0	 	0	 		0	 	0

	p denotes the auto-relaxation
	'''

	relaxation_matrix = np.zeros([10,10, len(fields)])
	#auto relaxation for pC1'z ========================
	key = (resid, "C1'", "H1'")
	p_C1p = rates.r1_YX(params[key], 
		spectral_density, 
		fields, 
		rxy[key], 
		csa_name[key], 
		'c', 
		y='h',
		cosine_angles= cos_ang[key], 
		csa_cosine_angles=csa_ang[key], 
		csa_params=csa_params[key],
		PhysQ=PhysQ)

	# print(f'Iz {p_C1p}')
	relaxation_matrix[2][2] = p_C1p

	#auto relaxation for pH1'z  ========================
	key = (resid, "H1'", "C1'")

	#print('====>>>', key)
	p_H1p = rates.r1_YX(params[key], 
		spectral_density, 
		fields, 
		rxy[key], 
		csa_name[key], 
		'h', 
		y='c',
		cosine_angles= cos_ang[key], 
		csa_cosine_angles=cos_ang[key], #csa_ang[key], 
		csa_params=None,
		PhysQ=PhysQ,
		model='axially symmetric')

	#add the dipoplar part for the H2' and H2'' vectors
	key = (resid, "H1'", "H2'1")
	p_H1p_H2p = rates.r1_YX_dipollar(params[key], 
		spectral_density, 
		fields, 
		rxy[key], 
		'h', 
		y='h',
		cosine_angles= cos_ang[key], 
		PhysQ=PhysQ)

	key = (resid, "H1'", "H2'2")
	p_H1p_H2pp = rates.r1_YX_dipollar(params[key], 
		spectral_density, 
		fields, 
		rxy[key], 
		'h', 
		y='h',
		cosine_angles= cos_ang[key], 
		PhysQ=PhysQ)

	relaxation_matrix[1][1] = (p_H1p + p_H1p_H2p + p_H1p_H2pp)

	# NOE between C1' and H1'  ========================
	key = (resid, "C1'", "H1'")
	sigma_c1p_h1p = rates.r1_reduced_noe_YX(params[key], 
		spectral_density, 
		fields,
		rxy[key], 
		'c', 
		y='h', 
		cosine_angles=cos_ang[key], 
		PhysQ=PhysQ)

	relaxation_matrix[1][2] = sigma_c1p_h1p
	relaxation_matrix[2][1] = sigma_c1p_h1p
	#print(f'C1p H1p NOE {sigma_c1p_h1p}', fields)
	# print(f'sigma {sigma_c1p_h1p}')
	#delta C1' H1'
	key = (resid, "C1'", "H1'")
	p_c1h1_delta = rates.delta_rate(params[key], 
		spectral_density, 
		fields, 
		rxy[key], 
		csa_name[key], 
		'c', 
		y='h',
		csa_cosine_angles=csa_ang[key], 
		csa_params=csa_params[key],
		PhysQ=PhysQ)

	relaxation_matrix[3][2] = p_c1h1_delta
	relaxation_matrix[2][3] = p_c1h1_delta
	# print(f'delta Iz {p_c1h1_delta}')
	#delta H1' C1'
	key = (resid,  "H1'", "C1'")
	p_h1c1_delta = rates.delta_rate(params[key], 
		spectral_density, 
		fields, 
		rxy[key], 
		csa_name[key], 
		'h', 
		y='c',
		csa_cosine_angles=cos_ang[key], # assume the csa is along the bond
		PhysQ=PhysQ, model='axially symmetric')

	relaxation_matrix[3][1] = p_h1c1_delta
	relaxation_matrix[1][3] = p_h1c1_delta
	# print(f'delta Sz {p_h1c1_delta}')
	# the two spin order for 2CzHz
	#delta C1' H1'
	
	# check in the two spin order term is the csa for Spin I or S? 
	key = (resid,  "C1'", "H1'")
	key_proton = (resid,  "H1'", "C1'")
	p_c1h1_2sp = rates.r_XzYz(params[key], 
		spectral_density, 
		fields, 
		rxy[key], 
		csa_name[key], 
		'c', 
		y='h',
		cosine_angles= cos_ang[key], # this is if the CSA is on the I spin 
		csa_cosine_angles=csa_ang[key], 
		csa_params=csa_params[key],
		PhysQ=PhysQ)
		# cosine_angles= cos_ang[key], # if the CSA is on the S spin
		# csa_cosine_angles=cos_ang[key], 
		# model='axially symmetric',
		# PhysQ=PhysQ)

	# print(f'IzSz {p_h1c1_delta + p_H1p_H2p + p_H1p_H2pp}')
	relaxation_matrix[3][3] = p_c1h1_2sp + p_H1p_H2p + p_H1p_H2pp

	# Now we add the terms for the interaction between H1' and the H2'1
	#auto relaxation for pH2'1z and pH2'2z ========================
	key = (resid, "H2'1", "H2'2")
	p_H2p_H2pp = rates.r1_YX_dipollar(params[key], 
		spectral_density, 
		fields, 
		rxy[key], 
		'h', 
		y='h',
		cosine_angles=cos_ang[key], 
		PhysQ=PhysQ)

	# might want to include a contribution from the H3'
	
	p_H2p = p_H1p_H2p + p_H2p_H2pp
	p_H2pp = p_H1p_H2pp + p_H2p_H2pp

	#print('...', p_H1p_H2p, p_H1p_H2pp, p_H2p_H2pp)

	relaxation_matrix[4][4] = p_H2p
	relaxation_matrix[6][6] = p_H2pp

	# add the NOEs between the H2'1 and H2'2
	sigma_h2p_h2pp = rates.r1_reduced_noe_YX(params[key], 
		spectral_density, 
		fields,
		rxy[key], 
		'h', 
		y='h', 
		cosine_angles=cos_ang[key], 
		PhysQ=PhysQ)

	relaxation_matrix[4][6] = sigma_h2p_h2pp
	relaxation_matrix[6][4] = sigma_h2p_h2pp

	#add the Noes to the H1' 
	key = (resid, "H1'", "H2'1")
	sigma_h1p_h2p = rates.r1_reduced_noe_YX(params[key], 
		spectral_density, 
		fields,
		rxy[key], 
		'h', 
		y='h', 
		cosine_angles=cos_ang[key], 
		PhysQ=PhysQ)

	relaxation_matrix[4][1] = sigma_h1p_h2p
	relaxation_matrix[1][4] = sigma_h1p_h2p

	key = (resid, "H1'", "H2'2")
	sigma_h1p_h2pp = rates.r1_reduced_noe_YX(params[key], 
		spectral_density, 
		fields,
		rxy[key], 
		'h', 
		y='h', 
		cosine_angles=cos_ang[key], 
		PhysQ=PhysQ)

	relaxation_matrix[6][1] = sigma_h1p_h2pp
	relaxation_matrix[1][6] = sigma_h1p_h2pp

	# # check in the two spin order term is the csa for Spin I or S? 
	# key = (resid,  "C1'", "H2'1")
	# key_proton = (resid,  "H2'1", "C1'")
	# p_c1h21_2sp = rates.r_YzXz(params[key], 
	# 	spectral_density, 
	# 	fields, 
	# 	rxy[key], 
	# 	csa_name[key], 
	# 	'c', 
	# 	y='h',
	# 	cosine_angles= cos_ang[key], 
	# 	csa_cosine_angles=cos_ang[key], 
	# 	model='axially symmetric',
	# 	PhysQ=PhysQ)

	# # print(f'IzSz {p_h1c1_delta + p_H1p_H2p + p_H1p_H2pp}')
	# relaxation_matrix[3][3] = p_c1h1_2sp + p_H1p_H2p + p_H1p_H2pp


	# check in the two spin order term is the csa for Spin I or S? 
	key = (resid,  "C1'", "H2'1")
	key_C1p = (resid, "C1'", "H1'")
	p_c1h21_2sp = rates.r_XzYz(params[key], 
		spectral_density, 
		fields, 
		rxy[key], 
		csa_name[key], 
		'c', 
		y='h',
		cosine_angles= cos_ang[key], # this is if the CSA is on the I spin 
		csa_cosine_angles=csa_ang[key], 
		csa_params=csa_params[key_C1p],
		PhysQ=PhysQ)
		# cosine_angles= cos_ang[key], # if the CSA is on the S spin
		# csa_cosine_angles=cos_ang[key], 
		# model='axially symmetric',
		# PhysQ=PhysQ)

	relaxation_matrix[5][5] = p_c1h21_2sp + p_H1p_H2pp
	#print(relaxation_matrix[5][5][0], p_c1h21_2sp[0], p_H1p_H2pp[0])

	# An NOE between then H2'1 and the H1' to transfer between 
	relaxation_matrix[5][3] = sigma_h1p_h2p
	relaxation_matrix[3][5] = sigma_h1p_h2p


	# check in the two spin order term is the csa for Spin I or S? 
	key = (resid,  "C1'", "H2'2")
	key_proton = (resid,  "H2'2", "C1'")
	p_c1h22_2sp = rates.r_XzYz(params[key], 
		spectral_density, 
		fields, 
		rxy[key], 
		csa_name[key], 
		'c', 
		y='h',
		cosine_angles= cos_ang[key], # this is if the CSA is on the I spin 
		csa_cosine_angles=csa_ang[key], 
		csa_params=csa_params[key_C1p],
		PhysQ=PhysQ)
		# cosine_angles= cos_ang[key], # if the CSA is on the S spin
		# csa_cosine_angles=cos_ang[key], 
		# model='axially symmetric',
		# PhysQ=PhysQ)

	#relaxation_matrix[7][7] = p_c1h22_2sp + p_H1p_H2pp
	# An NOE between then H2'1 and the H1' to transfer between 
	relaxation_matrix[7][3] = sigma_h1p_h2pp
	relaxation_matrix[3][7] = sigma_h1p_h2pp

	# relaxation for H4'
	key = (resid,  "H1'", "H4'")
	p_h4p = rates.r1_YX_dipollar(params[key], 
		spectral_density, 
		fields, 
		rxy[key], 
		'h', 
		y='h',
		cosine_angles=cos_ang[key], 
		PhysQ=PhysQ)

	relaxation_matrix[8][8] = p_h4p

	#noe 
	sigma_h1p_h4p = rates.r1_reduced_noe_YX(params[key], 
		spectral_density, 
		fields,
		rxy[key], 
		'h', 
		y='h', 
		cosine_angles=cos_ang[key], 
		PhysQ=PhysQ)

	relaxation_matrix[1][8] = sigma_h1p_h4p
	relaxation_matrix[8][1] = sigma_h1p_h4p

	return relaxation_matrix