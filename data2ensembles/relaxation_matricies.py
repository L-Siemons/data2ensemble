
import data2ensembles.rates as rates

def c1_prime_relaxation_matrix(params, spectral_density, fields):

	'''
	this is the relaxation matrix for the C1' in DNA
	In this system we will consider 
	C1'
	H1'
	H2'
	H2''

	The interactions we will consider will be 

	auto relaxation 
	C1'z
	H1'z
	C1'zH1'z

	C2'z
	C2''z


	
	The structure pf the matrix is

	0		0		0		0		0		0	 	0	 	0	
	0		pH1'	0		0		0		0	 	0		0	
	0		0		pC1'	0		0		0	 	0	 	0		
	0		0		0		pC1'H1'	0		0	 	0	 	0		
	0		0		0		0		pH2'	0	 	0	 	0		
	0		0		0		0		0		pC1'H2' 0	 	0	
	0		0		0		0		0		0	 	pH2'' 	0	
	0		0		0		0		0		0	 	0	 	pC1'H2''	

	p denotes the auto-relaxation
	'''

	relaxation_matrix = np.array([8,8])


	#unpack the varriables 
	params_c1ph1p = params
	rxy_c1ph1p = rxy
	csa_name_c1p = csa_atom_name
	cos_ang_c1p = cosine_angles
	csa_ang_c1p = csa_cosine_angles

	p_C1prime = rates.r1_YX(params_c1ph1p, 
		spectral_density, 
		fields, 
		rxy, 
		csa_name_c1p, 
		'c', 
		cos_ang_c1p = cos_ang_c1p, 
		csa_cosine_angles=csa_ang_c1p, 
		csa_params=None,
		PhysQ=PhysicalQuantities



