#################################################################
# A simple script to derive the ephemerides (only frequency at
# the moment) at a given MJD based on a timing solution. Currently,
# this script takes into account taylor expansion of the phase
# evolution and a random number of glitches - binary motion is
# not included
# 
# Input:
# 1- Tmjd: time at which to derive frequency
# 2- timeMod: timing model (.par file)
#
# Return:
# 1- ephemerides: dictionary of Tmjd and corresponding rotational
# frequency
#################################################################

import sys
import numpy as np
from math import factorial

# Custom modules
from crimp.readtimingmodel import readtimingmodel

sys.dont_write_bytecode = True


def ephemeridesAtTmjd(Tmjd, timMod):
    """
    Function that provides the spin frequency at the input MJD according
    to a timing model (.par file)

    :param Tmjd: time
    :type Tmjd: float
    :param timMod: .par file
    :type timMod: str
    :return: ephemerides
    :rtype: dict
    """
    timModParam = readtimingmodel(timMod)
    t0mjd = timModParam["PEPOCH"]

    #############################
    # Taking into account the taylor expansion terms to the frequency evolution
    freqAtTmjd_te = timModParam["F0"]
    for nn in range(1, 13):
        freqAtTmjd_te += (1 / factorial(nn)) * timModParam["F" + str(nn)] * ((Tmjd - t0mjd) * 86400) ** nn

    ##################################
    # Taking into account the glitches
    nbrGlitches = len([gg for glKey, gg in timModParam.items() if glKey.startswith('GLEP_')])
    freqAtTmjd_gl = 0  # initializing the jump in frequency due to all glitches combined

    for jj in range(1, nbrGlitches + 1):
        glep = timModParam["GLEP_" + str(jj)]
        # Creating boolean list based on whether any times are after glitch
        timesAfterGlitch = (Tmjd >= glep)

        # If Tmjd is after the glitch, calculate frequency shift according to the glitch model
        if timesAfterGlitch.any():
            glf0 = timModParam["GLF0_" + str(jj)]
            glf1 = timModParam["GLF1_" + str(jj)]
            glf2 = timModParam["GLF2_" + str(jj)]
            glf0d = timModParam["GLF0D_" + str(jj)]
            gltd = timModParam["GLTD_" + str(jj)]
            # Here we calculate frequency shift according to each glitch model for all time column,
            # then we multiply by 0 if times are < glep and 1 if times are > glep using the boolean list created above
            freqAtTmjd_gl += (glf0 + (glf1 * ((Tmjd - glep) * 86400)) + (0.5 * glf2 * ((Tmjd - glep) * 86400) ** 2) +
                              (glf0d * np.exp(-((Tmjd - glep) * 86400) / (gltd * 86400)))) * timesAfterGlitch

    ######################################
    # Adding all frequency-shifts together
    freqAtTmjd = freqAtTmjd_te + freqAtTmjd_gl

    ephemerides = {'Tmjd': Tmjd, 'freqAtTmjd': freqAtTmjd}

    return ephemerides
