
import numpy as np
from tqdm import tqdm
cimport numpy as np
np.import_array()

DTYPE = np.float
ctypedef np.float_t DTYPE_t

def calc_csa_spectral_density(selections, time_list, path, max_csa_ct_diff):


    cdef dict csa_ct_xx, csa_ct_yy, csa_ct_xy
    cdef float time_i,q,w,e,r
    cdef int time_array_len, mask_len
    cdef np.ndarray d11_i = np.zeros(3, dtype=DTYPE)
    cdef np.ndarray d22_i = np.zeros(3, dtype=DTYPE)
    cdef np.ndarray count, dts, idx
    
    csa_ct_xx = {}
    csa_ct_yy = {}
    csa_ct_xy = {}
    
    print('Calculating C(t) for CSA principle axis:')
    for i in tqdm(selections):

        current = np.array(selections[i])
        time_array = np.array(time_list[i], dtype=DTYPE)
        d11 = np.array(current[:,1].astype(float), dtype=DTYPE)
        d22 = np.array(current[:,2].astype(float), dtype=DTYPE)

        # this assumes that the smallest time point is time_array[0]
        # sanity check: are times in assending order, seems to be true!
        #print('Are times in assending order:', np.all(time_array[:-1] <= time_array[1:]))
        

        time_diffs = time_array - time_array[0]
        all_arrays, time_tot, cxx_tot, cyy_tot,  cxy_tot  = [],[],[],[],[]    
        time_array_len = len(time_array)

        for time_i, d11_i, d22_i in zip(time_array, d11, d22):

            # make the t0 array
            time_dim = np.abs(time_array-time_i)
            mask = (time_dim < max_csa_ct_diff)
            mask_len = sum(mask)


            d11_constant = np.zeros((mask_len, 3), dtype=DTYPE) + d11_i
            d22_constant = np.zeros((mask_len, 3), dtype=DTYPE) + d22_i

            # this einsum should give a 1D np array where the 
            # each entry is the dot product - I hope this works!

            #print('LOK HERE ')
            #print(d11[mask])
            #print(d11[mask].shape, d11_constant.shape)
            
            cxx = np.einsum('ij,ij->i',d11_constant,d11[mask], dtype=DTYPE)
            cyy = np.einsum('ij,ij->i',d22_constant,d22[mask], dtype=DTYPE)
            cxy = np.einsum('ij,ij->i',d11_constant,d22[mask], dtype=DTYPE)

            #cxx = np.array([np.dot(m,k) for m,k in zip(d11_constant,d11)])
            #cyy = np.array([np.dot(m,k) for m,k in zip(d22_constant,d22)])
            #cxy = np.array([np.dot(m,k) for m,k in zip(d11_constant,d22)])

            # now apply the P2 part 
            cxx = 1.5*cxx**2 - 0.5
            cyy = 1.5*cyy**2 - 0.5
            cxy = 1.5*cxy**2 - 0.5

            time_tot.append(time_dim[mask])
            cxx_tot.append(cxx)
            cyy_tot.append(cyy)
            cxy_tot.append(cxy)

        time_tot = np.concatenate(time_tot, axis=0, dtype=DTYPE)
        cxx_tot = np.concatenate(cxx_tot, axis=0, dtype=DTYPE)
        cyy_tot = np.concatenate(cyy_tot, axis=0, dtype=DTYPE)
        cxy_tot = np.concatenate(cxy_tot, axis=0, dtype=DTYPE)

        # so noe we want to sum according to the 
        dts, idx, count = np.unique(time_tot, return_counts=True, return_inverse=True)
        cur_ct_xx = np.bincount(idx, cxx_tot)/count
        cur_ct_yy = np.bincount(idx, cyy_tot)/count
        cur_ct_xy = np.bincount(idx, cxy_tot)/count

        # this assumes that the smallest time point is time_array[0]
        # sanity check: are times in assending order, seems to be true!
        # print('Are times in assending order:', np.all(dts[:-1] <= dts[1:]))
        
        # save the correlation functions.
        csa_ct_xx[i] = [dts, cur_ct_xx]
        csa_ct_yy[i] = [dts, cur_ct_yy]
        csa_ct_xy[i] = [dts, cur_ct_xy]

    # write out! 
    for i in csa_ct_xx:

        out_name = f'{path}_rotacf/csa_ct_{i[0]}_{i[1]}.xvg'
        f = open(out_name, 'w')
        f.write('#time xx yy xy\n')
        string = ''.join([ f'{q} {w} {e} {r}\n' for q,w,e,r in zip(csa_ct_xx[i][0], csa_ct_xx[i][1], csa_ct_yy[i][1], csa_ct_xy[i][1])])
        #print(string)
        f.write(string)
        f.close()

    return csa_ct_xx, csa_ct_yy, csa_ct_xy