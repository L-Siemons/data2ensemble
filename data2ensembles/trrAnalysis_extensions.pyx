import numpy as np 
import math
cimport numpy

cdef float theta
cdef int curve_count
cdef int i
ctypedef double DTYPE_x

def correlation_function(Params, numpy.ndarray[DTYPE_x, ndim=1] x, curve_count, theta):

    '''
    An internal correlation function. This is the same one that is used
    by Kresten in the absurder paper.
    '''

    total = 0
    for i in range(curve_count):
        i = i+1
        amp = 'amp_%i'%(i)
        time = 'time_%i'%(i)
        total = total + Params[amp]*(np.e**(-1*x/Params[time]))

    #this is for the vectors that are not aligned with eachother
    if theta != 0:
        total = (1.5*math.cos(theta)**2 - 0.5)*total

    total = total + Params['S_long']
    return total
