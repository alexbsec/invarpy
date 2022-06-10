"""
Common statistics

Routines in this module:

spectrum_estimate(field1D, pixel_size=1, shift_origin=False)	= power spectrum estimate from a 1D cosmological scalar field
split_spectrum(field1D, pixel_size=1, origin_shift=False)		= split spectra of an even length 1D cosmological scalar field 

"""

__all__ = ['spectrum_estimate', 'split_spectrum']


import numpy as np
from numpy.fft import fftshift, fftn, ifftshift, fftfreq, ifftn, fft, ifft




def spectrum_estimate(field1D, pixel_size=1, shift_origin=False):
	"""

	"""

	N = field1D.shape[0]

	vol_factor = N * pixel_size

	if shift_origin == True:

		field1D = ifftshift(field1D)

		field_fft = pixel_size * fftshift(fftn(field1D))

		ans = np.abs(field_fft)**2 / vol_factor

	else:

		field_fft = pixel_size * fftn(field1D)

		ans = np.abs(field_fft)**2 / vol_factor

	return ans




def split_spectrum(field1D, pixel_size=1, origin_shift=False):
	"""

	"""

	N = field1D.shape[0]

	if N % 2 != 0:
		raise ValueError("Length of field1D ndarray must be even.")

	n = int(N/2)

	field_L, field_R = field1D[:n], field1D[n:]

	pspec_L = spectrum_estimate(field_L, pixel_size=pixel_size, origin_shift=origin_shift)
	pspec_R = spectrum_estimate(field_R, pixel_size=pixel_size, origin_shift=origin_shift)

	return pspec_L, pspec_R




	
