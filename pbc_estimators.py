################################################################
#################### Fourier space approach ####################
################################################################

def sigma_which_diagonal_boundary_conditioned(field1D, diagonal=0):

    N = field1D.shape[0]
    ans = 0

    for i in range(-diagonal, N - diagonal):

        variable = (field1D[i] * np.conjugate(field1D[i + diagonal]))
        ans += (1/2) * (variable + np.conjugate(variable))

    return ans/N


def sigma_estimator_boundary_conditioned(field1D, estimator_kind=1, assume_invariance=False, field1D_spectrum=None):

    import numpy
    
    N = field1D.shape[0]
    field1D_estimator_kind2 = np.abs(field1D)**2
    ans = np.zeros((N), dtype='complex')

    sigma_bd_fn = sigma_which_diagonal_boundary_conditioned


    if assume_invariance == True:

        if type(field1D_spectrum) != numpy.ndarray:
            raise TypeError("Expected field1D_spectrum variable to be " + str(type(field1D.shape)) + " type. Got " + str(type(assume_invariance)) + " type instead.")


        sigma_bd_fn = sigma2_inv_which_diagonal_boundary_conditioned
        estimator_kind = 2
        field1D_estimator_kind2 = field1D_spectrum

    
    if estimator_kind == 1:

        for n in range(N):

            ans[n] = sigma_bd_fn(field1D, diagonal=n)

    elif estimator_kind == 2: 



        for n in range(N):

            ans[n] = sigma_bd_fn(field1D_estimator_kind2, diagonal=n)

    else:
        raise ValueError("Invalid estimator kind. Must be either 1 or 2.")

    return ans


def sigma2_inv_which_diagonal_boundary_conditioned(pspec, diagonal=0):

    N = pspec.shape[0]
    ans = 0 
    Id = np.identity(N)
    Ide = np.zeros((N,N))
    for j in range(N):
        Ide[:,j] = Id[:,-j]

    for loop in range(-diagonal, N - diagonal):

        ans += pspec[loop] * pspec[loop + diagonal] + (pspec[loop]**2) * (Ide[loop, loop + diagonal])**2 + (pspec[loop]**2) * (Id[loop, loop + diagonal])**2


    return ans/N
  

######################################################################
#################### Configuration space approach ####################
######################################################################

def sigma_estimator_boundary_conditioned_configuration_space(field1D, estimator_kind=1):



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
