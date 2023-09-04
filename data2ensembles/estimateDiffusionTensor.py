
import MDAnalysis as md 
import data2ensembles as d2e
from lmfit import Minimizer, Parameters, report_fit
import numpy as np
from .utils import PhysicalQuantities
import data2ensembles.utils as utils
import data2ensembles.structureUtils as strucUtils
import matplotlib.pyplot as plt
import string
import data2ensembles.trrAnalysis as trrAnalysis
import random as r
import itertools
from tqdm import tqdm 

PhysQ = PhysicalQuantities()

def guess_from_local_tc(r1_file, r2_file, hetNoe_file,   
    r1_errors, r2_errors, hetNoe_errors, PDB, fields, PhysQ=PhysQ, 
    make_figs=True, fig_base='', error_filter=0.075, mc_steps=25, selected_atom_types=None):

    '''
    This function esimates the diffusion tensor from the per-residue isotropic correlation times 
    as described in https://sci-hub.hkvisa.net/10.1126/science.7754375
    The PDB file needs to aligned aling its principle axis

    please note that the atom names in the r1, r2 and hetnoe files need to match those used in the 
    PDB file

    '''
    
    def resid(params, cur_r1, cur_r2, cur_noe, csa_key, bondlength, fields, nucleus_type):

        y = np.array([cur_r1, cur_r2, cur_noe]).T
        spec_den = d2e.spectralDensity.J_iso_tauc_only

        model_r1 = d2e.rates.r1_YX(params, spec_den, fields, bondlength, csa_key, nucleus_type)
        model_r2 = d2e.rates.r2_YX(params, spec_den, fields, bondlength, csa_key, nucleus_type)
        #print(model_r2, cur_r2)
        model_noe = d2e.rates.noe_YX(params, spec_den, fields, bondlength, nucleus_type, model_r1)
        model = np.array([model_r1,model_r2, model_noe])
        return y - model

    def di_model(params, cosine_angles):
        '''
        Calculate di as defined in 
        Journal of Biomolecular NMR, 9 (1997) 287–298 287
        ESCOM
        '''
        ex,ey,ez = cosine_angles
        di = params['q11']*ex**2 + params['q22']*ey**2 + params['q33']*ez**2 
        di = di + 2*params['q12']*ex*ey + 2*params['q13']*ex*ez + 2*params['q23']*ey*ez
        return di 


    def diff_tensor_reisd(params, tc, angles,errors):
        '''
        Here we calculate the diffusion tensor using the quadratic approach.
        Journal of Biomolecular NMR, 9 (1997) 287–298 287
        ESCOM
        '''



        total = []
        for tci, angs, err in zip(tc, angles,errors):

            di_exp = 1/(6*tci)
            di_calc = di_model(params, angs)
            err = 1/(6*err)
            total.append((di_exp - di_calc)/err)

        return np.array(total)

    # read in the data
    r1, _ = d2e.utils.read_nmr_relaxation_rate(r1_file)
    r2, _ = d2e.utils.read_nmr_relaxation_rate(r2_file) 
    hetnoe, _ = d2e.utils.read_nmr_relaxation_rate(hetNoe_file) 

    r1_err, _ = d2e.utils.read_nmr_relaxation_rate(r1_errors)
    r2_err, _ = d2e.utils.read_nmr_relaxation_rate(r2_errors)
    hetnoe_err, _ = d2e.utils.read_nmr_relaxation_rate(hetNoe_errors) 
    
    #lists to store stuff we will calculate
    local_correlation_times = []
    local_correlation_times_err = []
    cosine_angles = []


    #structural info
    uni = md.Universe(PDB)
    p1, p2, p3 = uni.select_atoms('all').principal_axes() 
    
    print('From principal_axes(), ie we assume the diffusion tensor is aligned with the inertia tensor')
    print('P1: ',p1)
    print('P2: ',p2)
    print('P3: ',p3)
    # is [p1,p2,p3] right handed: 
    right_hand_status = utils.is_right_handed(np.array([p1,p2,p3]).T)
    print('These are right handed? ', right_hand_status)

    for i in tqdm(r1):
        check = False

        # get the atom info earlier in the funtion that I originally thought 
        atom1_letters, atom1_numbers, atom1_res_type, atom1_resid, atom1_type, \
        atom2_letters, atom2_numbers, atom2_res_type, atom2_resid, atom2_type = utils.get_atom_info_from_rate_key(i)
        nucleus_type = atom2_letters[1].lower()
        atom_full_names = (atom2_letters[1] + atom2_numbers[1], atom1_letters[1] + atom1_numbers[1])

        # Quite a few logic checks! 
        if i in r2: 
            if i in hetnoe:
                # do some sub selection 
                if selected_atom_types != None:
                    for type_ in selected_atom_types:

                        for atom in atom_full_names:
                            if type_ in atom:
                                check=True
                else:
                    check = True



        #do some error filtering
        if error_filter != None:
            if np.mean(r1_err[i]/r1[i]) > error_filter:
                check = False

            if np.mean(r2_err[i]/r2[i]) > error_filter:
                check = False

            if np.mean(hetnoe_err[i]/hetnoe[i]) > error_filter:
                check = False

        if check == True:

            #get the local correlation times but fitting r1, r2, and hetnoe 
            cur_r1 = r1[i]
            cur_r2 = r2[i]
            cur_noe = hetnoe[i]

            csa_key = (atom2_type, atom2_res_type)
            bond_lengths_key = (atom2_type, atom1_type)
            bond_length = PhysQ.bondlengths[bond_lengths_key]


            mc_tcis = []
            for _ in range(mc_steps):

                cur_r1 = r.gauss(r1[i], r1_err[i])
                cur_r2 = r.gauss(r2[i], r2_err[i])
                cur_noe = r.gauss(hetnoe[i], hetnoe_err[i])

                params = Parameters()
                params.add('tc', value=2e-9, min=0, vary=True)

                # do fit, here with the default leastsq algorithm
                minner = Minimizer(resid, params, fcn_args=(cur_r1, cur_r2, cur_noe, csa_key, bond_length, fields, nucleus_type))
                result = minner.minimize(method='powel')
                mc_tcis.append(result.params['tc'].value)

            local_correlation_times.append(np.mean(mc_tcis))
            local_correlation_times_err.append(np.std(mc_tcis))

            #now we get the cosine angles
            vec = strucUtils.get_bond_vector(uni,atom1_resid, atom1_type, atom2_resid, atom2_type)
            #calculate the cosine angles. I am not sure how these should be ordered with 
            # repect to p1,p2,p3
            mag = np.linalg.norm(vec)

            ex = np.dot(vec, p1)/mag
            ey = np.dot(vec, p2)/mag
            ez = np.dot(vec, p3)/mag
            cosine_angles.append([ex,ey,ez])

    # do a completely free fit 
    print('== Anisotropic Fit of Q ==')
    params = Parameters()
    params.add('q11', value=1/(6*5e-9), min=0, vary=True)
    params.add('q22', value=1/(6*5e-9), min=0, vary=True)
    params.add('q33', value=1/(6*5e-9), min=0, vary=True)

    params.add('q12', value=0, min=0, vary=False)
    params.add('q13', value=0, min=0, vary=False)
    params.add('q23', value=0, min=0, vary=False)
    minner = Minimizer(diff_tensor_reisd, params, fcn_args=(local_correlation_times, cosine_angles, local_correlation_times_err))
    anisotropic_result = minner.minimize(method='powell')
    report_fit(anisotropic_result)

    # do a completely free fit 
    print('== Sphereoid Fit of Q ==')
    params = Parameters()
    params.add('q11', value=1/(6*5e-9), min=0, vary=True)
    params.add('q22', value=1/(6*5e-9), min=0, vary=True, expr='q11')
    params.add('q33', value=1/(6*5e-9), min=0, vary=True)

    params.add('q12', value=0, min=0, vary=False)
    params.add('q13', value=0, min=0, vary=False)
    params.add('q23', value=0, min=0, vary=False)
    minner = Minimizer(diff_tensor_reisd, params, fcn_args=(local_correlation_times, cosine_angles, local_correlation_times_err))
    sphereoid_result = minner.minimize(method='powell')
    report_fit(sphereoid_result)

    # model selection
    if sphereoid_result.bic < anisotropic_result.bic:
        print('Selecting spheroid tensor')
        selected_model = sphereoid_result
    elif anisotropic_result.bic < sphereoid_result.bic:
        print('Selecting anisotropic tensor')
        selected_model = anisotropic_result

    #some stats
    selected_params = selected_model.params
    diso = (selected_params['q11'].value + selected_params['q22'].value + selected_params['q33'].value)/3
    dpq = 2*selected_params['q33'].value/(selected_params['q11'].value + selected_params['q22'].value)
    print('Diso: ', diso)
    print('D||/DT: ', dpq)
    print('tc iso: ',1/(6*diso))

    if make_figs ==True:

        plt.hist(local_correlation_times)
        plt.title('Local Correlation Times')
        plt.xlabel('tc')
        plt.ylabel('frequency')
        plt.savefig(fig_base+'tc_hist.pdf')
        plt.close()

        #write out the model
        x = []
        y = []
        model_errors = []
        for tci, angs in zip(local_correlation_times, cosine_angles):

            tci = tci
            tci_calc = 1/(6*di_model(selected_params, angs))
            x.append(tci)
            y.append(tci_calc)

        xerr = np.array(local_correlation_times_err)
        
        x = np.array(x)
        y = np.array(y)
        minimum = min([min(x), min(y)]) * 0.9
        maximum = max([max(x), max(y)]) * 1.1
        rmsd = np.sqrt(np.mean((x-y)**2))
        
        plt.plot([minimum, maximum-rmsd], [minimum, maximum-rmsd], c='C2')
        plt.plot([minimum, maximum+rmsd], [minimum, maximum+rmsd], c='C2')
        plt.plot([minimum, maximum], [minimum, maximum], c='C1')
        plt.errorbar(x,y, xerr=xerr, fmt='o',) #yerr=yerr)
        plt.xlabel('exp tci')
        plt.ylabel('calc tci')
        plt.savefig(fig_base+'calc_local_tci.pdf')

    return selected_model



### All this code doesnt work ........

# def fit_spheroid_to_r1_r2(r1_file, r2_file, r1_errors, r2_errors, PDB, fields, PhysQ=PhysQ, align_PDB=True, select_atoms='all', error_cutoff=0.075):
#     # https://link.springer.com/content/pdf/10.1023/A:1018631009583.pdf

#     def model(params, spec_den, fields, bond, csa, atom, ang):
#         model_r1 = d2e.rates.r1_YX(params, spec_den, fields, bond, csa, atom, cosine_angles=[ang])
#         model_r2 = d2e.rates.r2_YX(params, spec_den, fields, bond, csa, atom, cosine_angles=[ang])
#         return model_r2/model_r1, model_r1, model_r2
#     def residual(params, quotients, angles, fields,bond_lengths, csa_keys,atom_type, errors):
#         spec_den = d2e.spectralDensity.J_axial_symetric
#         total = []
#         for q, ang, bond, csa, atom, err in zip(quotients, angles, bond_lengths, csa_keys, atom_type,errors):
#             model_value, _,_ = model(params, spec_den, fields, bond, csa, atom, ang)
#             total.append((q-model_value)/err)

#         return np.array(total)

#     r1, _ = d2e.utils.read_nmr_relaxation_rate(r1_file)
#     r2, _ = d2e.utils.read_nmr_relaxation_rate(r2_file) 

#     r1_err, _ = d2e.utils.read_nmr_relaxation_rate(r1_errors)
#     r2_err, _ = d2e.utils.read_nmr_relaxation_rate(r2_errors) 


#     #get common keys
#     common_keys = list(set(r1.keys()).intersection(r2.keys()))
#     #select only some atoms 
#     if select_atoms=='all':
#         selected_atoms = common_keys
#     else:
#         print('Selecting only atoms in', select_atoms)
#         #probably a nicer/faster way to do this
#         selected_atoms = []
#         for i in common_keys:
#             for j in select_atoms:
#                 if j in i:
#                     selected_atoms.append(i)

#     #properties needed for the fit
#     quotients = []
#     angles = []
#     bond_length = []
#     csa_keys = []
#     atom_types = []
#     errors = []

#     uni = md.Universe(PDB)
#     p_axis = uni.select_atoms('all').principal_axes()
#     print('Is the axis system right handed? ', utils.is_right_handed(p_axis))
#     p_long = p_axis[2]
#     r1list = []
#     r2list = []

#     for key in selected_atoms:

#         # sort the quocients
#         q = r2[key]/r1[key]
#         if error_cutoff != None:
#             print(r2[key],r2_err[key], r1[key], r1_err[key])
#             error_check = False
#             error_mc_list = []
#             for _ in range(50):
#                 r1_mc = r.gauss(r1[key], r1_err[key])
#                 r2_mc = r.gauss(r2[key], r2_err[key])

#                 value = r2_mc/r1_mc
#                 error_mc_list.append(value)

#             mc_std = np.std(error_mc_list)
#             print(mc_std)
#             error_percent = mc_std/q
#             if error_percent < error_cutoff:
#                 errors.append(mc_std)
#                 quotients.append(q)
#                 r1list.append(r1[key])
#                 r2list.append(r2[key])
#                 error_check = True
#                 print('yay')
#             else:
#                 #print(key, ' did not pass error filtering')
#                 pass

#         else:    
#             #print(key, r1_errors)
#             error_check = True
#             errors.append(1)
#             quotients.append(q)
#             r1list.append(r1[key])
#             r2list.append(r2[key])

#         if error_check == True:
#             print('I got here?')
#             #get the bond angles
#             atom1_letters, atom1_numbers, atom1_res_type, atom1_resid, atom1_type, \
#             atom2_letters, atom2_numbers, atom2_res_type, atom2_resid, atom2_type = utils.get_atom_info_from_rate_key(key)
#             unit_vec = strucUtils.get_bond_vector(uni,atom1_resid, atom1_type, atom2_resid, atom2_type)

#             # so in once case if I flip the unit vector then I get a result that seems to make sense. 
#             # I think this might be related to how MDAnalysis stores the atom possitions [z,y,x] but I am not very sure 
#             # about this - at the moment it seems to be working ...
#             angle = strucUtils.cacl_angle(unit_vec, p_long)
#             print(angle)
#             angles.append(angle)

#             #get the bondlengths
#             bond_lengths_key = (atom2_type, atom1_type)
#             bond_length.append(PhysQ.bondlengths[bond_lengths_key])

#             #get the cse keys
#             csa_key = (atom2_type, atom2_res_type)
#             csa_keys.append(csa_key)
#             atom_types.append(atom2_type[0].lower())

#     plt.hist(angles, bins=50)
#     plt.show()

#     #now we Set up all the parameters for the fitting
#     #only fit these two
#     params = Parameters()

#     params.add('dperpendicular', value=1.1e7, vary=True, min=0)
#     params.add('dparallel'     , value=4.1e7, vary=True, min=0)

#     # these are held constant
#     params.add('S2_f',value=1,vary=False)
#     params.add('S2_s',value=1,vary=False)

#     params.add('tau_f',value=1,vary=False)
#     params.add('tau_s',value=1,vary=False)

#     # do fit, here with the default leastsq algorithm
#     minner = Minimizer(residual, params, fcn_args=(quotients, angles, 14.1,bond_length, csa_keys,atom_types, errors))
#     result = minner.minimize(method='powel')
#     report_fit(result)

#     params = result.params
#     diso = (params['dparallel'] + 2*params['dperpendicular'])/3
#     print('Diso, ', diso)
#     print('D||/D', params['dparallel']/ params['dperpendicular'])
#     tc = 1/(6*diso)
#     print("'tc': ", tc)

#     x = []
#     modelr1 = []
#     modelr2 = []
#     for q, ang, bond, csa, atom, err in zip(quotients, angles, bond_length, csa_keys, atom_types,errors):
#         spec_den = d2e.spectralDensity.J_axial_symetric
#         model_value, model_r1, model_r2 = model(params, spec_den, fields, bond, csa, atom, ang)
#         modelr1.append(model_r1)
#         modelr2.append(model_r2)
#         x.append(model_value)

#     x = np.array(x)
#     quotients = np.array(quotients)
#     squarediff = (x - quotients)**2

#     rmsd = np.sqrt(np.mean(squarediff))
#     print(rmsd)
#     minimum = min([min(quotients), min(x)])
#     maximum = max([max(quotients), max(x)])

#     plt.scatter(quotients, x)
#     plt.plot([minimum,maximum], [minimum,maximum], 'C1')
#     plt.plot([minimum,maximum], [minimum+rmsd,maximum+rmsd], c='C2')
#     plt.plot([minimum,maximum], [minimum-rmsd,maximum-rmsd], c='C2')
#     plt.show()

#     plt.scatter(r1list, modelr1)
#     plt.show()
    
#     plt.scatter(r2list, modelr2)
#     plt.show()
    
# def fit_anisotropic_to_r1_r2(r1_file, r2_file, r1_errors, r2_errors, PDB, fields, 
#                              PhysQ=PhysQ, align_PDB=True, select_atoms='all', mc_error_status=True):
#     # https://link.springer.com/content/pdf/10.1023/A:1018631009583.pdf

#     def model(params, spec_den, fields, bond, csa, atom, ang):
#         model_r1 = d2e.rates.r1_YX(params, spec_den, fields, bond, csa, atom, cosine_angles=[ang])
#         model_r2 = d2e.rates.r2_YX(params, spec_den, fields, bond, csa, atom, cosine_angles=[ang])
#         return model_r2/model_r1

#     def residual(params, quotients, angles, fields,bond_lengths, csa_keys,atom_type, errors):
#         spec_den = d2e.spectralDensity.J_anisotropic_mf
#         total = []
#         for q, ang, bond, csa, atom, err in zip(quotients, angles, bond_lengths, csa_keys, atom_type,errors):
#             model_value = model(params, spec_den, fields, bond, csa, atom, ang)
#             total.append((q-model_value)/err)
#         return np.array(total)

#     r1, _ = d2e.utils.read_nmr_relaxation_rate(r1_file)
#     r2, _ = d2e.utils.read_nmr_relaxation_rate(r2_file) 

#     if mc_error_status == True:
#         r1_err, _ = d2e.utils.read_nmr_relaxation_rate(r1_errors)
#         r2_err, _ = d2e.utils.read_nmr_relaxation_rate(r2_errors) 


#     #get common keys
#     common_keys = list(set(r1.keys()).intersection(r2.keys()))
#     #select only some atoms 
#     if select_atoms=='all':
#         selected_atoms = common_keys
#     else:
#         print('Selecting only atoms in', select_atoms)
#         #probably a nicer/faster way to do this
#         selected_atoms = []
#         for i in common_keys:
#             for j in select_atoms:
#                 if j in i:
#                     selected_atoms.append(i)

#     # #align the PDB to the axis of inertia 
#     # if align_PDB == True: 
#     #     s = PDB.split('.')
#     #     new_PDB_name = s[0]+'_aligned_to_inertia_tensor.'+s[-1]
#     #     print(new_PDB_name)
#     #     PrepPDB = trrAnalysis.PrepareReference(PDB)
#     #     PrepPDB.align_to_inertia_axis(outname=new_PDB_name)

#     # else:
#     #     new_PDB_name = PDB

#     uni = md.Universe(PDB)
#     p1, p2, p3 = uni.select_atoms('all').principal_axes() 
    
#     print('From principal_axes() ')
#     print('P1: ',p1)
#     print('P2: ',p2)
#     print('P3: ',p3)
#     # is [p1,p2,p3] right handed: 
#     right_hand_status = utils.is_right_handed(np.array([p1,p2,p3]).T)
#     print('These are right handed? ', right_hand_status)
#     axis = [p1,p2,p3]
#     axis_names =['p1', 'p2', 'p3']

#     for  i in itertools.permutations([0, 1,2], 3):
#         i1, i2, i3 = i
#         #properties needed for the fit
#         quotients = []
#         angles = []
#         angles_plot = []
#         bond_length = []
#         csa_keys = []
#         atom_types = []
#         errors = []
#         print(f'ex: {axis_names[i1]} ey: {axis_names[i2]} ez: {axis_names[i3]} ')
#         for key in selected_atoms:

#             # sort the quocients
#             q = r2[key]/r1[key]
            
#             # some error handling if we want
#             check = True
#             if mc_error_status == True:
#                 check = False
#                 r1_error_percent = r1_err[key]/r1[key]*100
#                 r2_error_percent = r2_err[key]/r2[key]*100

#                 if r1_error_percent < 8:
#                     if r2_error_percent < 8: 
#                         check = True

#             if check == True:
#                 if mc_error_status == True:
#                     # propergate error with MC estimation
#                     mc_error = []
#                     for _ in range(50):
#                         #print(r.gauss(r1[key], r1_err[key]),r.gauss(r2[key], r2_err[key]))
#                         value = r.gauss(r2[key], r2_err[key])/r.gauss(r1[key], r1_err[key])
#                         mc_error.append(value)

#                     errors.append(np.std(mc_error))
#                 else:
#                     #no error
#                     errors.append(1)
        
#                 quotients.append(q)

#                 #get the bond angles
#                 atom1_letters, atom1_numbers, atom1_res_type, atom1_resid, atom1_type, \
#                 atom2_letters, atom2_numbers, atom2_res_type, atom2_resid, atom2_type = utils.get_atom_info_from_rate_key(key)

#                 #calculate the bond vectors 
#                 vec = strucUtils.get_bond_vector(uni,atom1_resid, atom1_type, atom2_resid, atom2_type)
#                 #calculate the cosine angles. I am not sure how these should be ordered with 
#                 # repect to p1,p2,p3
#                 mag = np.linalg.norm(vec)

#                 ex = np.dot(vec, axis[i1])/mag
#                 ey = np.dot(vec, axis[i2])/mag
#                 ez = np.dot(vec, axis[i3])/mag

#                 angles.append([ez,ey,ex])
#                 angles_plot.append([np.arccos(a) for a in [ez,ey,ex]])
#                 #get the bondlengths
#                 bond_lengths_key = (atom2_type, atom1_type)
#                 bond_length.append(PhysQ.bondlengths[bond_lengths_key])

#                 #get the cse keys
#                 csa_key = (atom2_type, atom2_res_type)

#                 csa_keys.append(csa_key)
#                 atom_types.append(atom2_type[0].lower())

#         angles_plot = np.array(angles_plot).T
#         fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
#         # axs[0].scatter(angles_plot[0], quotients)
#         # axs[0].set_xlim(0,np.pi)
#         # axs[1].scatter(angles_plot[1], quotients)
#         # axs[1].set_xlim(0,np.pi)
#         # axs[2].scatter(angles_plot[2], quotients)
#         # axs[2].set_xlim(0,np.pi)
#         # plt.show()
#         #now we Set up all the parameters for the fitting
#         #only fit these two
#         params = Parameters()
#         params.add('dxx', value=3/4.1e-6, vary=True, min=0)
#         params.add('dyy', value=3/4.1e-6, vary=True, min=0)
#         params.add('dzz', value=3/4.1e-6, vary=True, min=0)

#         # these are held constant
#         params.add('S2_f',value=1,vary=False)
#         params.add('S2_s',value=1,vary=False)

#         params.add('tau_f',value=1,vary=False)
#         params.add('tau_s',value=1,vary=False)

#         # do fit, here with the default leastsq algorithm
#         minner = Minimizer(residual, params, fcn_args=(quotients, angles, 14.1,bond_length, csa_keys,atom_types, errors))
#         result = minner.minimize(method='powel')
#         report_fit(result)

#         params = result.params
#         diso = (params['dxx'] +params['dyy'] + params['dzz'])/3
#         print('Diso, ', diso)
#         tc = 1/(6*diso)
#         print("'tc': ", tc)
#         print('dzz/dxx', params['dzz'].value/params['dxx'].value)
#         print('dzz/dyy', params['dzz'].value/params['dyy'].value)
#         print('dyy/dxx', params['dyy'].value/params['dxx'].value)

#         x = []
#         for q, ang, bond, csa, atom, err in zip(quotients, angles, bond_length, csa_keys, atom_types,errors):
#             spec_den = d2e.spectralDensity.J_anisotropic_mf
#             model_value = model(params, spec_den, fields, bond, csa, atom, ang)
#             x.append(model_value)

#         x = np.array(x)
#         quotients = np.array(quotients)
#         squarediff = (x - quotients)**2

#         rmsd = np.sqrt(np.mean(squarediff))
#         print(rmsd)
#         minimum = min([min(quotients), min(x)])
#         maximum = max([max(quotients), max(x)])

#         # plt.scatter(quotients, x)
#         # plt.plot([minimum,maximum], [minimum,maximum], 'C1')
#         # plt.plot([minimum,maximum], [minimum+rmsd,maximum+rmsd], c='C2')
#         # plt.plot([minimum,maximum], [minimum-rmsd,maximum-rmsd], c='C2')

#         # plt.show()