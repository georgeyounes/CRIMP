####################################################################################
# calcPhase.py is a module that calculates phases of an array of TIME (MJD) instances
# using a .par file. It reads the .par file with the readTimMod scripts. As a reminder,
# this script can manage glitches, wave functions and the frequency and its derivatives
# up to F12; it does not accomodate IFUNC, binary systems, proper motion, or parallax.
# These will be implemented in future versions, likely in that respective order
#
# Make this into a class
####################################################################################

import sys
import numpy as np

from math import factorial

# Custom modules
from crimp import readTimMod

sys.dont_write_bytecode = True


##############################################################################################
# Script that calculates pulsar rotational phases using a time array and a .par timing model #
##############################################################################################


class Phases:
    """
        A class to calcualte the phases of a time array

        Attributes
        ----------
        timeMJD : float
            time array in modified julian day
        timMod : str
            timing model, i.e., .par file

        Methods
        -------
        taylorExpansion():
            calculates the phases according to the taylor expansion of the phase evolution (F0, F1, F2, etc.)
        glitches():
            calculates the phases according to glitches
        waves():
            calculates the phases according to "waves" used to whiten noise and align ToAs
        """

    def __init__(self, timeMJD, timMod):
        """
        Constructs all the necessary attributes for the Phases object.

        Parameters
        ----------
            timeMJD : float
                time array in modified julian day
            timMod : str
                timing model, i.e., .par file
        """
        self.timeMJD = timeMJD
        self.timMod = timMod
        self.timModParam = readTimMod(self.timMod)

    def taylorExpansion(self):
        """
        Method to calculate phases from the spin parameters, i.e., taylor expansion of the phase evolution
        :return: phases_te
        :rtype: float
        """
        t0mjd = self.timModParam["PEPOCH"]
        phases_te = 0
        for nn in range(1, 14):
            phases_te += (1 / factorial(nn)) * self.timModParam["F" + str(nn - 1)] * (
                    (self.timeMJD - t0mjd) * 86400) ** nn

        return phases_te

    def glitches(self):
        """
        Method to calculate phases from glitches in spin evolution
        :return: phases_gl_all
        :rtype: float
        """
        nbrGlitches = len([gg for glKey, gg in self.timModParam.items() if glKey.startswith('GLEP_')])
        phases_gl_all = 0  # initializing the jump in phase due to all glitches combined

        for jj in range(1, nbrGlitches + 1):

            glep = self.timModParam["GLEP_" + str(jj)]
            # Creating boolean list based on whether any times are after glitch
            timesAfterGlitch = (self.timeMJD >= glep)

            # If any time instance in the array timeMJD is after the glitch,
            # calculate phase jumps according to the glitch model
            if timesAfterGlitch.any():
                glph = self.timModParam["GLPH_" + str(jj)]
                glf0 = self.timModParam["GLF0_" + str(jj)]
                glf1 = self.timModParam["GLF1_" + str(jj)]
                glf2 = self.timModParam["GLF2_" + str(jj)]
                glf0d = self.timModParam["GLF0D_" + str(jj)]
                gltd = self.timModParam["GLTD_" + str(jj)]
                # Calculating phases
                # Here we calculate phase jumps according to each glitch model for all time column,
                # then we multiply by 0 if times are < glep and 1 if times are > glep using the boolean list created above
                phases_gl = (glph + (glf0 * ((self.timeMJD - glep) * 86400)) +
                             (0.5 * glf1 * ((self.timeMJD - glep) * 86400) ** 2) +
                             ((1 / 6) * glf2 * ((self.timeMJD - glep) * 86400) ** 3) +
                             (glf0d * (gltd * 86400) * (1 - np.exp(
                                 -((self.timeMJD - glep) * 86400) / (gltd * 86400))))) * timesAfterGlitch

                phases_gl_all += phases_gl

        return phases_gl_all

    def waves(self):
        """
        Method to calculate phases from "waves", which is used to whiten noise and align ToAs
        :return: phases_waves_all
        :rtype: float
        """
        nbrWaves = np.array([ww for wvKey, ww in self.timModParam.items() if wvKey.startswith('WAVE')])
        phases_waves_all = 0  # initializing the noise in phase due to all waves combined

        if nbrWaves.size:
            waveEpoch = self.timModParam["WAVEEPOCH"]
            waveFreq = self.timModParam["WAVE_OM"]

            for jj in range(1, len(nbrWaves) - 1):
                phases_waves_all += ((self.timModParam["WAVE" + str(jj)]["A"] * np.sin(jj * waveFreq * (self.timeMJD - waveEpoch))) +
                                     (self.timModParam["WAVE" + str(jj)]["B"] * np.cos(jj * waveFreq * (self.timeMJD - waveEpoch))))
        return phases_waves_all


def calcPhase(timeMJD, timMod):
    """
    Function that adds all above phase calculations
    :param timeMJD: array of times
    :type timeMJD: float
    :param timMod: timing model, .par file
    :type timMod: str
    returns
            - totalphases (float): phases (normalised by 2pi) \n
            - cycleFoldedPhases (float): cycle folded phases [0,1)
    """
    phases = Phases(timeMJD, timMod)
    phases_te = phases.taylorExpansion()
    phases_gl_all = phases.glitches()
    phases_waves_all = phases.waves()

    totalphases = phases_te + phases_gl_all + phases_waves_all
    cycleFoldedPhases = totalphases - np.floor(totalphases)

    return totalphases, cycleFoldedPhases
