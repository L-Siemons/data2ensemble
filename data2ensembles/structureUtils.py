import numpy as np
import MDAnalysis as md



def get_bond_vector(uni,atom1_resid, atom1_type, atom2_resid, atom2_type, unit_vec=False):

    select_atom2 = uni.select_atoms(f'resid {atom2_resid} and name {atom2_type}')[0]
    possition2 = select_atom2.position

    select_atom1 = uni.select_atoms(f'resid {atom1_resid} and name {atom1_type}')[0]
    possition1 = select_atom1.position

    diff = (possition1 - possition2)

    if unit_vec == False:
    	return diff
    else: 
    	return diff/np.linalg.norm(diff)

def cacl_angle(a, b):
    '''
    calculate the angle between two vectors
    '''


    inner = np.dot(a, b)
    norms = np.linalg.norm(a) * np.linalg.norm(b)

    cos = inner / norms
    rad = np.arccos(np.clip(cos, -1.0, 1.0))
    return rad