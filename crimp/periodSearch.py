#############################################################
# Simple implementation of the Z2 and H-test statistics to
# search for periodicity. Inputs are a time array, frequency
# array, and the number of harmonics. It outputs an array of
# the power at each frequency. Check Buccheri et al. (1983)
# for more details on the Ztest and Jagar et al. (1989) for
# more details on the Htest, and many more recent literature.
#
# In a future iteration, I may implement other tests, e.g.,
# 2d-Ztest
#############################################################
import argparse
import sys
import numpy as np

sys.dont_write_bytecode = True


#######################################
# Class to search for periodic signal #
#######################################

class PeriodSearch:
    def __init__(self, time, freq, nbrHarm: int = 2):
        self.time = time
        self.freq = freq
        self.nbrHarm = nbrHarm

    #################################################################
    def ztest(self):  # Ztest implmentation - could be made more efficient

        Z2pow = np.zeros(len(self.freq))
        n = len(self.time)

        for jj in range(len(self.freq)):
            Z2pow[jj] = np.sum([((np.sum(np.cos(2 * (kk + 1) * np.pi * self.freq[jj] * self.time))) ** 2 +
                                 (np.sum(np.sin(2 * (kk + 1) * np.pi * self.freq[jj] * self.time))) ** 2) for kk in
                                range(0, self.nbrHarm)]) * (2.0 / n)

        return Z2pow

    #################################################################
    def htest(self):  # Htest implmentation - could be made more efficient

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


##########################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Class to search for periodic signal - methods allowed are ztest and htest")
    args = parser.parse_args()
