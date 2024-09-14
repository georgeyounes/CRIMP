#############################################################
# Simple implementation of the Z2 and H-test statistics to
# search for periodicity. Inputs are a time array, frequency
# array, and the number of harmonics. It outputs an array of
# the power at each frequency. Check Buccheri et al. (1983)
# for more details on the Ztest and Jagar et al. (1989) for
# more details on the Htest, and many more recent literature.
# Now, it supports the 2d_Z^2 test in the method "twod_ztest"
# which requires an addition array input of frequency
# derivatives
#
# In a future iteration, 2d-Z^2 and more periodicity search
# methods may be implemented
#
# Warning: these methods are not optimized for speed, and are
# best utilized if you have a rough idea where the spin frequency
# (frequency-derivative) of your object-of-interest is.
#############################################################

import sys
import numpy as np
import pandas as pd

sys.dont_write_bytecode = True


class PeriodSearch:
    """
            A class to perform simple periodicity searches

            Attributes
            ----------
            time : numpy.ndarray
                array of photon arrival times
            freq : numpy.ndarray
                array of frequencies
            nbrHarm : int
                number of harmonics for the Z^2-test or H-test

            Methods
            -------
            ztest(): calculate Z^2 power at each frequency
            twod_ztest(freq_dot): calculate Z^2 power at each pair of frequency, frequency derivative
            htest(): calculate H power at each frequency
            """

    def __init__(self, time, freq, nbrHarm: int = 2):
        """
        Constructs the necessary attribute for the Phases object.

        :param time: array of photon arrival times
        :type time: numpy.ndarray
        :param freq: array of frequencies
        :type freq: numpy.ndarray
        :param nbrHarm: number of harmonics for the Z^2-test or H-test
        :type nbrHarm: int
        """
        self.time = time
        self.freq = freq
        self.nbrHarm = nbrHarm

    #################################################################
    def ztest(self):
        """
        Calculate Z^2-test power
        :return: array of Z^2-power
        :rtype: numpy.ndarray
        """
        Z2pow = np.zeros(len(self.freq))
        n = len(self.time)

        for jj in range(len(self.freq)):
            Z2pow[jj] = np.sum([((np.sum(np.cos(2 * (kk + 1) * np.pi * self.freq[jj] * self.time))) ** 2 +
                                 (np.sum(np.sin(2 * (kk + 1) * np.pi * self.freq[jj] * self.time))) ** 2) for kk in
                                range(0, self.nbrHarm)]) * (2.0 / n)

        return Z2pow

    def twod_ztest(self, freq_dot):
        """
        Calculate 2-d Z^2-test power
        :param freq_dot: array of frequency derivatives
        :type freq_dot: numpy.ndarray
        :return: Z2pow_2d, array of Z^2-power for each pair of freq,freq_dot
        :rtype: numpy.ndarray
        :return: Z2pow_2d_df, Pandas dataframe of Z^2-power for each pair of freq,freq_dot
        :rtype: pandas.DataFrame

        """
        Z2pow_2d = np.zeros((len(self.freq) * len(freq_dot), 3))  # I need to find a way to preallocate here
        mm = 0
        n = len(self.time)

        for ll in range(0, len(freq_dot)):
            for jj in range(len(self.freq)):
                # Formula derived from Strohmayer & Markwardt 1999, ApJ, 516L, 81S
                # \nu(t) = \phi(0) + \nu_0 t + \int_0^t \nu^\prime(t) dt
                # \int_0^t \nu^\prime(t) dt = \delta\phi + \delta\nu t + \frac{1}{2} \dot{\nu} t^2.
                z2nHarm = np.sum([(np.sum(np.cos(2 * (kk + 1) * np.pi *
                                                 (self.freq[jj] * self.time +
                                                  (0.5 * (-1 * 10 ** freq_dot[ll]) * self.time ** 2)))) ** 2) +
                                  (np.sum(np.sin(2 * (kk + 1) * np.pi * (self.freq[jj] * self.time +
                                                                         (0.5 * (-1 * 10 ** freq_dot[ll]) *
                                                                          self.time ** 2)))) ** 2)
                                  for kk in range(0, self.nbrHarm)]) * (2.0 / n)

                Z2pow_2d[mm, :] = [self.freq[jj], freq_dot[ll], z2nHarm]
                mm += 1  # counter

        Z2pow_2d_df = pd.DataFrame(Z2pow_2d, columns=['Freq', 'Freq_dot', 'Z2pow'])

        return Z2pow_2d, Z2pow_2d_df

    #################################################################
    def htest(self):
        """
        Calculate H-test power
        :return: array of H-power
        :rtype: numpy.ndarray
        """
        Hpow = np.zeros(len(self.freq))
        n = len(self.time)
        Z2nHarm = np.zeros(self.nbrHarm)
        for jj in range(len(self.freq)):
            for kk in range(0, self.nbrHarm):
                Z2nHarm[kk] = ((np.sum(np.cos(2 * (kk + 1) * np.pi * self.freq[jj] * self.time))) ** 2 +
                               (np.sum(np.sin(2 * (kk + 1) * np.pi * self.freq[jj] * self.time))) ** 2) * (2.0 / n)
            Z2nHarm.cumsum()
            Hpow[jj] = np.amax([(Z2nHarm[ll] - 4 * ll + 4) for ll in range(0, self.nbrHarm)])

        return Hpow
