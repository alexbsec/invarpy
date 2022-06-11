"""
Periodic-Boundary-Condition Translation Non-Invariance Estimators

Routines in this module:

#   Fourier Space Method

sigma_which_diagonal(field1D, diagonal=0)                                               = finds the desired diagonal of the covariance matrix of field1D
sigma(field1D, estimator_kind=1, assume_invariance=False, field1D_spectrum=None)        = 1st or 2nd kind biased sigma estimator
sigma_bias_which_diagonal(pspec, diagonal=0, estimator_kind=1)                          = finds the desired diagonal of the bias matrix of field1D

#   Configuration Space Method
sigma_cs(field1D, estimator_kind=1)                                                     = 1st or 2nd kind biased sigma estimator

"""

__all__ = ['sigma', 'sigma_which_diagonal', 'sigma_bias_which_diagonal', 'sigma_cs']



import functools

import numpy as np
from numpy.fft import fftshift, fftn, ifftshift, fftfreq, ifftn, fft, ifft




################################################################
#################### Fourier space approach ####################
################################################################




def sigma_which_diagonal(field1D, diagonal=0, estimator_kind=1):
    """
    Compute, assuming Python's periodic boundary condition, the diagonal of the covariance matrix of a given field. 
    
    Parameters
    ----------
    pspec : one-dimensiona complex ndarray
            Input one-dimensional ndarray corresponding 
            to the cosmological field Fourier transformed.
    diagonal : int, default=0
               the desired diagonal of the covariance matrix to be
               computed.
    Returns
    -------
    ans : One-index object containing a complex ndarray
          Returns the desired diagonal of the covariance matrix of that
          Fourier transformed field input.

    Raises
    ------
    ValueError
        If 'estimator_kind' is not equal to either 1 or 2 (int).

    """

    N = field1D.shape[0]
    ans = 0

    if estimator_kind == 1:

        for i in range(-diagonal, N - diagonal):

            variable = (field1D[i] * np.conjugate(field1D[i + diagonal]))
            ans += (1/2) * (variable + np.conjugate(variable))

    elif estimator_kind == 2:

        for i in range(-diagonal, N - diagonal):

            ans += (field1D[i] * np.conjugate(field1D[i + diagonal]))

    else:
        raise ValueError("Invalid estimator kind. Must be either 1 or 2.")
    return ans/N


def sigma(field1D, estimator_kind=1, assume_invariance=False, field1D_spectrum=None):
    """

    Compute the desired kind of the biased sigma estimator, given a field, using Fourier space method. 
    
    Parameters
    ----------
    field1D : one-dimensiona complex ndarray
              Input one-dimensional ndarray corresponding 
              to the cosmological field Fourier transformed.
    estimator_kind : int, two-choices, default=1
                     This sets the estimator kind to be computed. 
                     Must be either 1 (1st kind) or 2 (2nd kind).
    assume_invariance : bool, default=False
                        If true, compute bias of desired estimator_kind.
    field_spectrum : One-dimensional real ndarray, default=None
                     Add this parameter only if you set assume_invariance=true.
                     Feed it with the field power spectrum.

    Returns
    -------
    ans : one-dimensional complex ndarray
          Returns the desired kind of the biased sigma estimator, using Fourier space method, or
          if assume_invariance=True, returns the bias of that sigma estimator kind.

    Raises
    ------
    ValueError
        If 'estimator_kind' is not equal to either 1 or 2 (int).

    TypeError
        if 'assume_invariance' is set to True, but no ndarray power spectrum is assigned to field_spectrum variable.

    """

    import numpy
    
    N = field1D.shape[0]

    ans = np.zeros((N), dtype='complex')

    sigma_bd_fn = sigma_which_diagonal

    if assume_invariance == True:

        if type(field1D_spectrum) != numpy.ndarray:

            raise TypeError("Expected field1D_spectrum variable to be " + str(type(field1D.shape)) + " type. Got " + str(type(assume_invariance)) + " type instead.")
        
        sigma_bd_fn = sigma_bias_which_diagonal
        field1D = field1D_spectrum

    if estimator_kind == 1:
         

        for n in range(N):

            ans[n] = sigma_bd_fn(field1D, diagonal=n)

    elif estimator_kind == 2:

        field1D = np.abs(field1D)**2

        if assume_invariance == True:

            field1D = field1D_spectrum


        for n in range(N):

            ans[n] = sigma_bd_fn(field1D, diagonal=n, estimator_kind=2)

    else:
        raise ValueError("Invalid estimator kind. Must be either 1 or 2.")

    return ans


def sigma_bias_which_diagonal(pspec, diagonal=0, estimator_kind=1):

    """

    Compute the diagonal of the bias matrix, given the field power spectrum, using Fourier space method. 
    
    Parameters
    ----------
    pspec : one-dimensiona complex ndarray
            Input one-dimensional ndarray corresponding 
            to the cosmological field Fourier transformed.
    diagonal : int, default=0
               the desired diagonal of the covariance matrix to be
               computed.
    estimator_kind : int, two-choices, default=1
                     This sets the estimator kind to be computed. 
                     Must be either 1 (1st kind) or 2 (2nd kind).

    Returns
    -------
    ans : One-index object containing a complex ndarray
          Returns the desired diagonal of the covariance matrix of that
          Fourier transformed field input.

    Raises
    ------
    ValueError
        If 'estimator_kind' is not equal to either 1 or 2 (int).

    """

    N = pspec.shape[0]
    ans = 0 
    Id = np.identity(N)
    Ide = np.zeros((N,N))

    if estimator_kind == 1:

        if diagonal == 0:

            for loop in range(-diagonal, N - diagonal):

                ans += pspec[loop]

        else:

            ans = 0

        return ans/N

    
    elif estimator_kind == 2:

        for j in range(N):

            Ide[:,j] = Id[:,-j]


        for loop in range(-diagonal, N - diagonal):

            ans += pspec[loop] * pspec[loop + diagonal] + (pspec[loop]**2) * (Ide[loop, loop + diagonal])**2 + (pspec[loop]**2) * (Id[loop, loop + diagonal])**2


        return ans

    else:
        ValueError("Invalid estimator kind. Must be either 1 or 2.")


  

######################################################################
#################### Configuration space approach ####################
######################################################################



def sigma_cs(field1D, estimator_kind=1):

    """

    Compute the desired kind of the biased sigma estimator, given a field, using configuration space method. 
    
    Parameters
    ----------
    field1D : one-dimensiona complex ndarray
              Input one-dimensional ndarray corresponding 
              to the cosmological field Fourier transformed.
    estimator_kind : int, two-choices, default=1
                     This sets the estimator kind to be computed. 
                     Must be either 1 (1st kind) or 2 (2nd kind).

    Returns
    -------
    ans : one-dimensional complex ndarray
          Returns the desired kind of the biased sigma estimator, using configuration space method.

    Raises
    ------
    ValueError
        If 'estimator_kind' is not equal to either 1 or 2 (int).

    """

    N = field1D.shape[0]

    if estimator_kind == 1:

        input_field = fftn(field1D**2)

        ans = (1/(2*N)) * (input_field + np.conjugate(input_field))

    elif estimator_kind == 2:

        rho_fft = np.abs( fftn(field1D) )**2
        rho = ifftn(rho_fft)
        input_field = fftn( rho**2 )

        ans = (1/N)*input_field

    else:
        raise ValueError("Invalid estimator kind. Must be either 1 or 2.")

    return ans
