from data2ensembles.utils import PhysicalQuantities
import data2ensembles.rates as rates
import numpy as np

'''
The code here could be abstracted a little ...
'''

PhysicalQuantities = PhysicalQuantities()

def c1_prime_relaxation_matrix(params,
		resid, 
		spectral_density, 
		fields, 
		rxy, 
		cos_ang, 
		csa_ang, 
		csa_params,
		restype,
		PhysQ=PhysicalQuantities):

	'''
	this is the relaxation matrix for the C1' in DNA

	
	The structure pf the matrix is
	relaxation operators: 
	E/2
	C1'z
	H1'z
	H2'1z
	H2'2z
	H3'z
	H4'z
	C1'zH1'z
	C1'zH1'z
	C1'zH2'1z
	C1'zH2'2z
	C1'zH3'z
	C1'zH4'z
	'''

	# make an emty matrix
	relaxation_matrix = np.zeros([8,8,  len(fields)])
	proton_dipolar = {}
	proton_dipolar_summed = {}
	protons = ["H1'", "H2'1", "H2'2",] #  "H3'", "H4'"]
	protons_passive = protons  + ["H5'1", "H5'2", 'H8', 'H6'] + ["H3'", "H4'"]
	proton_len = len(protons)
	proton_len_passive = len(protons_passive)
	tsp_index_add = proton_len + 2

	# calculate the contribution to the dipolar network
	for i in range(proton_len_passive):
		proton_i = protons_passive[i]
		for j in range(i+1, proton_len_passive):
			proton_j = protons_passive[j]

			#this is ugly, I should unify the keys
			key = (resid, f"{proton_i},{proton_j}")
			dist_key = (resid, proton_i , proton_j)
			cos_key = (resid, proton_i , resid, proton_j)
			#print(rxy[dist_key])]

			if key in params:
				dipolar_contribution = rates.r1_YX_dipollar(params[key], 
					spectral_density, 
					fields,
					rxy[dist_key], 
					'h',
					cosine_angles = cos_ang[cos_key])

				proton_dipolar[key] = dipolar_contribution

	# relaxation for Cz 
	key = (resid, "C1',H1'")
	cos_key = (resid, "C1'" , resid, "H1'")
	csa_atom_name = ("C1'", restype)
	auto_C1pz = rates.r1_YX(params[key], 
		spectral_density, 
		fields,
		PhysQ.bondlengths["C1'", "H1'"], 
		csa_atom_name, 
		'c', 
		y='h', 
		cosine_angles = cos_ang[cos_key], 
		csa_cosine_angles=csa_ang[cos_key], 
		csa_params=csa_params,
		PhysQ=PhysQ, 
		model='anisotropic')

	csa_atom_name = ("H1'", restype)
	auto_H1pz = rates.r1_YX(params[key], 
		spectral_density, 
		fields,
		PhysQ.bondlengths["C1'", "H1'"], 
		csa_atom_name, 
		'h', 
		y='c', 
		cosine_angles = cos_ang[cos_key], 
		csa_cosine_angles=cos_ang[cos_key],
		PhysQ=PhysQ, 
		model='axially symmetric')
	#print('auto H1', auto_H1pz)


	relaxation_matrix[1][1] = auto_C1pz
	relaxation_matrix[2][2] = auto_H1pz

	# add contributions from the proton dipolar network to the auto relaxation 
	# and the two spin order relaxation
	for indx, item in enumerate(protons):
		indx_auto = indx + 2
		indx_tsp = indx + tsp_index_add

		for i in proton_dipolar:
			if item in i[1]:
				relaxation_matrix[indx_auto][indx_auto] = relaxation_matrix[indx_auto][indx_auto] + proton_dipolar[i]
				relaxation_matrix[indx_tsp][indx_tsp] = relaxation_matrix[indx_tsp][indx_tsp] + proton_dipolar[i]
				
		
	# heteronuclear NOE 
	key = (resid, "C1',H1'")
	noe_c1p_h1p = rates.r1_reduced_noe_YX(params[key], 
		spectral_density, 
		fields,
		PhysQ.bondlengths["C1'", "H1'"], 
		'c', 
		y='h', 
		cosine_angles=cos_ang[cos_key], 
		PhysQ=PhysQ)

	relaxation_matrix[1][2] = noe_c1p_h1p
	relaxation_matrix[2][1] = noe_c1p_h1p

	# now do all the homonuclear NOEs
	for indx, item in enumerate(protons):
		indx_mod = indx + 2
		indx_tsp = indx + tsp_index_add


		for jndx in range(indx + 1, len(protons)):
			jndx_mod = jndx + 2
			jndx_tsp = jndx + tsp_index_add
			jtem = protons[jndx]


			#should unify the keys at some point ...
			dist_key = (resid, item , jtem)
			# print(dist_key, rxy[dist_key])
			cos_key = (resid, item , resid,  jtem)
			key = (resid, f"{item},{jtem}")

			if key in params:
				current_noe = rates.r1_reduced_noe_YX(params[key], 
							spectral_density, 
							fields,
							rxy[dist_key], 
							'h', 
							y='h', 
							cosine_angles=cos_ang[cos_key], 
							PhysQ=PhysQ)

				relaxation_matrix[indx_mod][jndx_mod] = current_noe
				relaxation_matrix[jndx_mod][indx_mod] = current_noe

				relaxation_matrix[indx_tsp][jndx_tsp] = current_noe
				relaxation_matrix[jndx_tsp][indx_tsp] = current_noe

			# print('current NOE', current_noe	)

	# We already have added the latice dipolar contribution to the two spin order 
	# so now we just the terms themselves 
	# this uses the csa of the C1'
	csa_atom_name = ("C1'", restype)
	csa_cos_key = (resid, "C1'" , resid, "H1'")
	for indx, item in enumerate(protons):

		indx_tsp = indx + tsp_index_add
		indx_auto = indx + 2

		key = (resid, f"C1',{item}")
		dist_key = (resid, "C1'" , item)
		cos_key = (resid, "C1'" , resid, item)
		if item == "H1'":
			current_dist = PhysQ.bondlengths["C1'", "H1'"]
		else:
			current_dist = rxy[dist_key]

		two_spin_order = rates.r_XzYz(params[key], 
		spectral_density, 
		fields, 
		current_dist, 
		csa_atom_name, 
		'c', 
		y='h',
		cosine_angles=cos_ang[cos_key], # this is if the CSA is on the I spin 
		csa_cosine_angles=csa_ang[csa_cos_key], 
		csa_params=csa_params,
		PhysQ=PhysQ)
		#print(two_spin_order)

		#print('>>>', item, relaxation_matrix[indx_tsp][indx_tsp][0])
		relaxation_matrix[indx_tsp][indx_tsp] = relaxation_matrix[indx_tsp][indx_tsp] + two_spin_order
		#print(relaxation_matrix[indx_tsp][indx_tsp][0])
		
		# now we add the cross correlated relaxation
		p_c1h_delta = rates.delta_rate(params[key], 
		spectral_density, 
		fields, 
		rxy[dist_key], 
		csa_atom_name, 
		'c', 
		y='h',
		csa_cosine_angles=csa_ang[csa_cos_key], # using the CSA of spin I 
		csa_params=csa_params,
		PhysQ=PhysQ)

		relaxation_matrix[1][indx_tsp] = p_c1h_delta
		relaxation_matrix[indx_tsp][1] = p_c1h_delta
		# print('p_c1h_delta', p_c1h_delta	)

	# print(relaxation_matrix.round(2)[:,:,0])
	return relaxation_matrix


def c5_relaxation_matrix(params,
		resid, 
		spectral_density, 
		fields, 
		rxy, 
		cos_ang, 
		csa_ang, 
		csa_params,
		restype,
		PhysQ=PhysicalQuantities):

	'''
	this is the relaxation matrix for the C1' in DNA

	
	The structure pf the matrix is
	relaxation operators: 
	E/2
	C5z
	H5z
	H6z
	H42z
	C5zH5z
	C5zH6z
	C5zH42z
	'''

	# make an emty matrix
	relaxation_matrix = np.zeros([8,8 ,  len(fields)])
	proton_dipolar = {}
	proton_dipolar_summed = {}
	protons = ["H5", "H42", "H6",] #  "H3'", "H4'"]
	protons_passive = protons
	proton_len = len(protons)
	proton_len_passive = len(protons_passive)
	tsp_index_add = proton_len + 2

	# calculate the contribution to the dipolar network
	for i in range(proton_len_passive):
		proton_i = protons_passive[i]
		for j in range(i+1, proton_len_passive):
			proton_j = protons_passive[j]

			#this is ugly, I should unify the keys
			key = (resid, f"{proton_i},{proton_j}")
			dist_key = (resid, proton_i , proton_j)
			cos_key = (resid, proton_i , resid, proton_j)
			#print(rxy[dist_key])]

			if key in params:
				dipolar_contribution = rates.r1_YX_dipollar(params[key], 
					spectral_density, 
					fields,
					rxy[dist_key], 
					'h',
					cosine_angles = cos_ang[cos_key])

				proton_dipolar[key] = dipolar_contribution

	# relaxation for Cz 
	key = (resid, "C5,H5")
	cos_key = (resid, "C5" , resid, "H5")
	csa_atom_name = ("C5", restype)
	auto_C1pz = rates.r1_YX(params[key], 
		spectral_density, 
		fields,
		PhysQ.bondlengths["C5", "H5"], 
		csa_atom_name, 
		'c', 
		y='h', 
		cosine_angles = cos_ang[cos_key], 
		csa_cosine_angles=csa_ang[cos_key], 
		csa_params=csa_params,
		PhysQ=PhysQ, 
		model='anisotropic')

	csa_atom_name = ("H5", restype)
	auto_H1pz = rates.r1_YX(params[key], 
		spectral_density, 
		fields,
		PhysQ.bondlengths["C5", "H5"], 
		csa_atom_name, 
		'h', 
		y='c', 
		cosine_angles = cos_ang[cos_key], 
		csa_cosine_angles=cos_ang[cos_key],
		PhysQ=PhysQ, 
		model='axially symmetric')
	#print('auto H1', auto_H1pz)


	relaxation_matrix[1][1] = auto_C1pz
	relaxation_matrix[2][2] = auto_H1pz

	# add contributions from the proton dipolar network to the auto relaxation 
	# and the two spin order relaxation
	for indx, item in enumerate(protons):
		indx_auto = indx + 2
		indx_tsp = indx + tsp_index_add

		for i in proton_dipolar:
			if item in i[1]:
				relaxation_matrix[indx_auto][indx_auto] = relaxation_matrix[indx_auto][indx_auto] + proton_dipolar[i]
				relaxation_matrix[indx_tsp][indx_tsp] = relaxation_matrix[indx_tsp][indx_tsp] + proton_dipolar[i]
				
		
	# heteronuclear NOE 
	key = (resid, "C5,H5")
	noe_c1p_h1p = rates.r1_reduced_noe_YX(params[key], 
		spectral_density, 
		fields,
		PhysQ.bondlengths["C5", "H5"], 
		'c', 
		y='h', 
		cosine_angles=cos_ang[cos_key], 
		PhysQ=PhysQ)

	relaxation_matrix[1][2] = noe_c1p_h1p
	relaxation_matrix[2][1] = noe_c1p_h1p

	# now do all the homonuclear NOEs
	for indx, item in enumerate(protons):
		indx_mod = indx + 2
		indx_tsp = indx + tsp_index_add


		for jndx in range(indx + 1, len(protons)):
			jndx_mod = jndx + 2
			jndx_tsp = jndx + tsp_index_add
			jtem = protons[jndx]


			#should unify the keys at some point ...
			dist_key = (resid, item , jtem)
			# print(dist_key, rxy[dist_key])
			cos_key = (resid, item , resid,  jtem)
			key = (resid, f"{item},{jtem}")

			if key in params:
				current_noe = rates.r1_reduced_noe_YX(params[key], 
							spectral_density, 
							fields,
							rxy[dist_key], 
							'h', 
							y='h', 
							cosine_angles=cos_ang[cos_key], 
							PhysQ=PhysQ)

				relaxation_matrix[indx_mod][jndx_mod] = current_noe
				relaxation_matrix[jndx_mod][indx_mod] = current_noe

				relaxation_matrix[indx_tsp][jndx_tsp] = current_noe
				relaxation_matrix[jndx_tsp][indx_tsp] = current_noe

			# print('current NOE', current_noe	)

	# We already have added the latice dipolar contribution to the two spin order 
	# so now we just the terms themselves 
	# this uses the csa of the C1'
	csa_atom_name = ("C5", restype)
	csa_cos_key = (resid, "C5" , resid, "H5")
	for indx, item in enumerate(protons):

		indx_tsp = indx + tsp_index_add
		indx_auto = indx + 2

		key = (resid, f"C5,{item}")
		dist_key = (resid, "C5" , item)
		cos_key = (resid, "C5" , resid, item)
		if item == "H5":
			current_dist = PhysQ.bondlengths["C5", "H5"]
		else:
			current_dist = rxy[dist_key]

		two_spin_order = rates.r_XzYz(params[key], 
		spectral_density, 
		fields, 
		current_dist, 
		csa_atom_name, 
		'c', 
		y='h',
		cosine_angles=cos_ang[cos_key], # this is if the CSA is on the I spin 
		csa_cosine_angles=csa_ang[csa_cos_key], 
		csa_params=csa_params,
		PhysQ=PhysQ)
		#print(two_spin_order)

		#print('>>>', item, relaxation_matrix[indx_tsp][indx_tsp][0])
		relaxation_matrix[indx_tsp][indx_tsp] = relaxation_matrix[indx_tsp][indx_tsp] + two_spin_order
		#print(relaxation_matrix[indx_tsp][indx_tsp][0])
		
		# now we add the cross correlated relaxation
		p_c1h_delta = rates.delta_rate(params[key], 
		spectral_density, 
		fields, 
		rxy[dist_key], 
		csa_atom_name, 
		'c', 
		y='h',
		csa_cosine_angles=csa_ang[csa_cos_key], # using the CSA of spin I 
		csa_params=csa_params,
		PhysQ=PhysQ)

		# relaxation_matrix[1][indx_tsp] = p_c1h_delta
		# relaxation_matrix[indx_tsp][1] = p_c1h_delta
		# print('p_c1h_delta', p_c1h_delta	)

	# print(relaxation_matrix.round(2)[:,:,0])
	return relaxation_matrix
