"""
Non Periodic-Boundary-Condition Translation Non-Invariance Estimators

Routines in this module:

#   Fourier Space Method

sigma(field1D, estimator_kind=1)                                    = 1st or 2nd kind biased sigma estimator computed in Fourier space
sigma_bias(pspec, estimator_kind=1)                                 = 1st or 2nd kind estimator bias
sigma_which_diagonal(field1D, diagonal=0)                           = finds the desired diagonal of the covariance matrix of field1D
sigma_bias_which_diagonal(pspec, diagonal=0, estimator_kind=1)      = finds the desired diagonal of the 1st or 2nd kind estimator matrix

#   Configuration Space Method
finite_geometric_series(N, n, m, l)                                 = finite geometric series function returns int
geometric_series_matrices(N)                                        = find all geometric matrices by varying n, m, l variables on finite_geometric_series method
sigma_cs(field1D, geometric_matrices, estimator_kind=1)             = 1st or 2nd kind biased sigma estimator computed in configuration space

"""

__all__ = ['sigma', 'sigma_bias', 'sigma_which_diagonal', 'sigma_bias_which_diagonal', 'finite_geometric_series', 'geometric_series_matrices', 'sigma_cs']



import functools

import numpy as np
from numpy.fft import fftshift, fftn, ifftshift, fftfreq, ifftn, fft, ifft




################################################################
#################### Fourier space approach ####################
################################################################


#   field1D must be a flattened cosmological field in a 1-D ndarray type
#   estimator_kind takes either 1 or 2, refers to the estimator you want to use to compute non-invariance
#   outputs the first or second kind estimator for the input field1D.
def sigma(field1D, estimator_kind=1):
    N = field1D.shape[0]
    count = np.zeros((N))
    ans = np.zeros((N),dtype='complex')

    if estimator_kind == 2:
        field1D = np.abs(field1D)**2

    elif estimator_kind != 1 or estimator_kind != 2:
        raise ValueError("Invalid estimator kind. Must be either 1 or 2.")
    
    for i in range(N):
        if i == 0:
            ans[i] = (1/N)*np.sum(sigma_which_diagonal(field1D, diagonal=0)[0])
            count[i] = 1
        
        else:
            ans[i] = (1/(N-i))*np.sum(sigma_which_diagonal(field1D, diagonal=(i))[0])
            count[i]=1 
                
    return ans/count


#   pspec refers to the field power spectrum and must also be flattened into a 1-D ndarray type
#   method used to compute either 1st or 2nd kind estimator bias given the field power spectrum
#   outputs the bias for that field given its power spectrum.
def sigma_bias(pspec, estimator_kind=1):
    N = pspec.shape[0]  
    ans = np.zeros((N),dtype='complex')

    if estimator_kind == 1:

        ans[0] = (1/N)*np.sum(sigma_bias_which_diagonal(pspec, diagonal=0)[0])

    elif estimator_kind == 2:

        for i in range(N):
            
            ans[i] = (1/(N-i))*np.sum(sigma_bias_which_diagonal(pspec, diagonal=i, estimator_kind=2)[0])

    else:
        raise ValueError("Invalid estimator kind. Must be either 1 or 2.")
              
    return ans


#   Computes the n-th diagonal cross correlation between the field with itself where n is
#   the corresponding covariance matrix diagonal
#   outputs an object containing a N - diagonal shaped ndarray.
def sigma_which_diagonal(field1D, diagonal=0):
    N = field1D.shape[0]
    ans = np.zeros((1),dtype='object')
    
    ans[0] = (1/2)*(field1D[diagonal:]*np.conjugate(field1D[:N-diagonal]) +
                           np.conjugate(field1D[diagonal:]*np.conjugate(field1D[:N-diagonal])))
    return ans


#   Find the n-th diagonal of the 1st or 2nd kind estimator bias
#   outputs an object continaing a N - diagona shaped ndarray.
def sigma_bias_which_diagonal(pspec, diagonal=0, estimator_kind=1):
    N = pspec.shape[0]
    ans = np.zeros((1),dtype='object')
    Id = np.identity(N)
    Ide = np.zeros((N,N))

    if estimator_kind == 1:
        ans[0] = pspec

    elif estimator_kind == 2:
        for j in range(N):
            Ide[:,j] = Id[:,-j]

        ans[0] = pspec[:N-diagonal]*pspec[diagonal:] + (pspec[:N-diagonal]**2)*(np.diagonal(Ide,offset=diagonal)**2) + (pspec[:N-diagonal]**2)*(np.diagonal(Id,offset=diagonal)**2)

    else:
        raise ValueError("Invalid estimator kind. Must be either 1 or 2.")
    
    return ans


######################################################################
#################### Configuration space approach ####################
######################################################################

#   Compute a finite geometric series of the following exponential. LaTeX script:
#   \sum_{p=0}^{N-n-1} e^{-\frac{2* \pi * i * p * (m - l)}{N}}
#   outputs the result of that sum given the parameters N, n, m and l
def finite_geometric_series(N, n, m, l):
    if m == l:
        return N - n
    
    elif n == 0:
        return 0
    
    
    ans = (1 - np.exp(-(1 - n/N) * 2 * np.pi * 1j * (m - l)))/(1 - np.exp(-2 * np.pi * 1j * (m - l) / N))
    return ans


#   Loop through variable (n, m, l) of finite_geometric_series method to generate a 3-D ndarray
#   first index corresponds to n variable, second index corresponds to m variable and third index corresponds to l variable
#   outputs all geometric matrices for every 0 <= n, m, l <= N - 1.
def geometric_series_matrices(N):

    geometric_matrices = np.zeros((N, N, N), dtype='complex')
    
    for n in range(N):
        for m in range(N):
            for l in range(N):

                geometric_matrices[n, m, l] = finite_geometric_series(N, n, m, l) * np.exp(2 * np.pi * 1j * l * n / N)

    return geometric_matrices


#   Computes 1st or 2nd kind sigma estimator in configurations space
#   outputs the desired estimator for that given field.
def sigma_cs(field1D, geometric_matrices, estimator_kind=1):

    N = field1D.shape[0]
    ans = np.zeros((N), dtype='complex')

    if estimator_kind == 1:
 

        for n in range(N):

            transformed_field = np.dot(geometric_matrices[n, :, :], field1D)
            ans[n] = (1 / (2*N - 2*n) ) * ( np.dot(field1D, transformed_field) + np.dot(field1D, np.transpose(np.conjugate(transformed_field))) )



    elif estimator_kind == 2:

        field1D = ifftn(np.abs(fftn(field1D))**2)
        field1D_conjugate = np.conjugate(field1D)


        for n in range(N):

            transformed_field = np.dot(geometric_matrices[n, :, :], field1D_conjugate)
            ans[n] = (1/(N-n)) * (np.dot(field1D, transformed_field))

    else:
        raise ValueError("Invalid estimator kind. Must be either 1 or 2.")

    
    return ans

