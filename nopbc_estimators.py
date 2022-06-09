import numpy as np
from numpy.fft import fftshift, fftn, ifftshift, fftfreq, ifftn, fft, ifft


################################################################
#################### Fourier space approach ####################
################################################################

def sigma_shreded(flat_field, bins, nd=1):
    N = flat_field.shape[0]
    count = np.zeros((bins))
    ans = np.zeros((bins),dtype='complex')
    
    for i in range(bins):
        if i == 0:
            ans[i] = (1/N)*np.sum(sigma1_which_diagonal(flat_field, diagonal=0)[0])
            count[i] = 1
            
        elif N == bins:
            ans[i] = (1/(N-i))*np.sum(sigma1_which_diagonal(flat_field, diagonal=(i))[0])
            count[i]=1 
        else:
            for j in range(i,i+nd):
                ans[i] += (1/(N-(i+j-1)))*np.sum(sigma1_which_diagonal(flat_field, diagonal=(i+j-1))[0])
                count[i]+=1

                
    return ans/count

def sigma2_inv_shreded(flat_field, bins, nd=1):
    N = flat_field.shape[0]
    count = np.zeros((bins))
    
    ans = np.zeros((bins),dtype='complex')
    
    for i in range(bins):
        if i == 0:
            ans[i] = (1/N)*np.sum(sigma2_inv_which_diagonal(flat_field, diagonal=0)[0])
            count[i] = 1
            
        elif N == bins:
            ans[i] = (1/(N-i))*np.sum(sigma2_inv_which_diagonal(flat_field, diagonal=(i))[0])
            #print(i,sigma1_which_diagonal(flat_field, diagonal=(i)), ans[i])
            count[i]=1
            #if i == 1:
               # print((sigma2_inv_which_diagonal(flat_field, diagonal=(i))))
        else:
            for j in range(i,i+nd):
                
                ans[i] += (1/(N-(i+j-1)))*np.sum(sigma2_inv_which_diagonal(flat_field, diagonal=(i+j-1))[0])
                count[i]+=1

                
    return ans/count

def sigma1_which_diagonal(box1d,diagonal=0):
    N = box1d.shape[0]
    ans = np.zeros((1),dtype='object')
    
    ans[0] = (1/2)*(box1d[diagonal:]*np.conjugate(box1d[:N-diagonal]) +
                           np.conjugate(box1d[diagonal:]*np.conjugate(box1d[:N-diagonal])))
    return ans

def sigma2_inv_which_diagonal(pspec, diagonal=0):
    N = pspec.shape[0]
    ans = np.zeros((1),dtype='object')
    Id = np.identity(N)
    Ide = np.zeros((N,N))
    for j in range(N):
        Ide[:,j] = Id[:,-j]

    ans[0] = pspec[:N-diagonal]*pspec[diagonal:] + (pspec[:N-diagonal]**2)*(np.diagonal(Ide,offset=diagonal)**2) + (pspec[:N-diagonal]**2)*(np.diagonal(Id,offset=diagonal)**2)
    
    return ans


######################################################################
#################### Configuration space approach ####################
######################################################################


def finite_geometric_series(N, n, m, l):
    if m == l:
        return N - n
    
    elif n == 0:
        return 0
    
    
    ans = (1 - np.exp(-(1 - n/N) * 2 * np.pi * 1j * (m - l)))/(1 - np.exp(-2 * np.pi * 1j * (m - l) / N))
    return ans

def geometric_series_matrices(N):

    ans = np.zeros((N, N, N), dtype='complex')
    
    for n in range(N):
        for m in range(N):
            for l in range(N):

                ans[n, m, l] = finite_geometric_series(N, n, m, l) * np.exp(2 * np.pi * 1j * l * n / N)

    return ans


def sigma_through_geometric_series(field1D, geometric_matrices, estimator_kind=1):

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