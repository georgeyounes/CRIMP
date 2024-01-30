#############################################################
# Simple implementation of the Z2 and H-test statistics to
# search for periodicity. Inputs are a time array, frequency
# array, and the number of harmonics. It outputs an array of
# the power at each frequency. Check Buccheri et al. (1983)
# for more details on the Ztest and Jagar et al. (1989) for
# more details on the Htest, and many more recent literature.
#
# In a future iteration, 2d-Z^2 and more periodicity search
# methods may be implemented
#############################################################

import sys
import numpy as np

sys.dont_write_bytecode = True


class PeriodSearch:
    """
            A class to operate on a fits event file

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
    def ztest(self):  # Ztest implmentation - could be made more efficient
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

    #################################################################
    def htest(self):  # Htest implmentation - could be made more efficient
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
