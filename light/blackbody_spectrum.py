import numpy as np
import phys_constants as constants

'''
This file will generate a random wavelength from the Black Body distribution
'''


def planck_distribution(wavelength,temperature):
    '''

    :param wavelength: nanometers
    :param temperature: Kelvin
    :return: spectral density per unit area of the black body
    '''

    p = 2.0*(2.0*np.pi)*(constants.hbarc)*(constants.c) # (ev*nm) * m/s
    p = (p/wavelength**5)  # (1/nm**5)


    z = (2.0*np.pi*constants.hbarc)/(wavelength*constants.kB*temperature)
    p = p/(np.exp(z)-1.0)


    return p