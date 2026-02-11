'''
This module has functions to determine diffusion from NMR data using an approach similar
https://sci-hub.hkvisa.net/10.1126/science.7754375
Long-Range Motional Restrictions in a Multi domain Zinc-Finger Protein from Anisotropic Tumbling

and 

Rotational diffusion anisotropy of proteins from 
simultaneous analysis of 15N and 13Cα nuclear spin relaxation
'''        

import numpy as np 
import data2ensembles.utils as utils

def q_model(cosine_angles, q):
            cosine_angles = np.array(cosine_angles)
            part_1 = np.matmul(q, cosine_angles)
            part_2 = np.matmul(cosine_angles, part_1)
            return part_2

def q_model_from_params(cosine_angles, params):
    # set up the matrix q
    dx = params['dx']
    dy = params['dy']
    dz = params['dz']

    qx = (dy + dz)/2
    qy = (dx + dz)/2
    qz = (dx + dy)/2

    q = np.zeros([3,3])
    q[0][0] = qx
    q[1][1] = qy
    q[2][2] = qz
    
    model = q_model(cosine_angles, q)
    return model

def model_all_residues(params, diso_local, cosine_angles):
    # determine the diffusions
    diffs = []
    model = []
    for i in diso_local:
        res_info = utils.get_atom_info_from_tag(i)
        resid, resname, atom1, atom2 = res_info
        key = (resid, atom1 , resid, atom2)

        current_cosine_angles = cosine_angles[key]
        current_model = q_model_from_params(current_cosine_angles, params)
        model.append(current_model)
        diffs.append(diso_local[i] - current_model)

    diffs = np.array(diffs) 
    model = np.array(model)
    return model, diffs

def quadratic_diffusion_resid(params, diso_local,cosine_angles):
    _, diffs = model_all_residues(params, diso_local, cosine_angles)
    return diffs